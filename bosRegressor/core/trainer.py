import os
import os.path as osp
import numpy as np

import torch
import wandb
from aitviewer.models.smpl import SMPLLayer
from loguru import logger
from tqdm import tqdm

from bosRegressor.utils.misc_utils import copy2cpu as c2c
from bosRegressor.losses.build_losses import min_signed_distance_to_convex_hull
from bosRegressor.utils import constants
from humos.utils.fk import ForwardKinematicsLayer
from humos.utils.mesh_utils import smplh_breakdown


def collate_batch(item, train_feats, device):
    # item has "motion_x_dict" keys which has multiple keys. Filter out the keys that are in train_feats. Each key has value of shape (batch_size, frames, features), shuffle along the frames axis and combine frames and batch dimension to get (batch_size*frames, features)

    batch = {}
    import ipdb;
    ipdb.set_trace()
    for k, v in item["motion_x_dict"].items():
        if k in train_feats:
            # shuffle v along the frames axis
            v = v[:, torch.randperm(v.shape[1]), ...].to(device).float()
            # combine frames and batch dimension
            v = v.reshape(-1, v.shape[-1])
            batch[k] = v
    import ipdb;
    ipdb.set_trace()
    return batch


class BoSTrainer:
    def __init__(self, hparams, model, criterion, optimizer, normalizer, biomechanical_evaluator):
        self.device = hparams.DEVICE
        self.train_feats = hparams.TRAINING.TRAIN_FEATS
        self.hparams = hparams
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.normalizer = normalizer
        self.biomechanical_evaluator = biomechanical_evaluator

        # Get smpl body models
        self.bm_male = SMPLLayer(model_type="smplh", gender="male", device=self.device)
        self.bm_female = SMPLLayer(model_type="smplh", gender="female", device=self.device)
        # Get fk models
        self.fk_male = ForwardKinematicsLayer(constants.SMPL_PATH, gender="male",
                                              num_joints=constants.SMPLH_BODY_JOINTS, device=self.device)

    def construct_input(self, data):
        """
        Flatten the input data and create motion_features "x" and identity_features "identity"
        """
        # get unflatten sizes for each features
        self.unflat_feat_sizes = {k: v.shape if isinstance(v, torch.Tensor) else -1 for k, v in data.items()}
        # flatten each feature
        data = {k: v.view(v.shape[0], v.shape[1], -1) if isinstance(v, torch.Tensor) and len(v.shape) > 2 else v for
                k, v
                in data.items()}

        # Take out x-y translation from all frames to put all poses at the center
        # data['trans'][..., :2] = 0.0

        # get sizes of each feature
        self.flat_feat_sizes = {k: v.shape[-1] if isinstance(v, torch.Tensor) else -1 for k, v in data.items()}
        features = torch.cat([data[k].float() for k in self.train_feats], dim=-1)
        data['x'] = features

        identity_feats = ["betas", "gender"]
        identity_features = torch.cat([data[k].float() for k in identity_feats], dim=-1)
        data['identity'] = identity_features
        return data

    def deconstruct_input(self, data, identity):
        """
        Get original features from "x" and "identity" and unflatten them
        """
        # split features according to sequentially
        split_indxs = [self.flat_feat_sizes[k] for k in self.train_feats]

        # split the features
        data = torch.split(data, split_indxs, dim=-1)
        # make dict again
        data = {k: v for k, v in zip(self.train_feats, data)}

        data["betas"] = identity[:, :, :-1]
        data["gender"] = identity[:, :, -1]

        # unflatten each feature
        data = {k: v.view(*self.unflat_feat_sizes[k]) if isinstance(v, torch.Tensor) else v for k, v in
                data.items()}
        return data

    def run_smpl_fk(self, data, skinning=True):
        gender = data["identity"][:, 0, -1]
        smpl_params = smplh_breakdown(data, fk=self.fk_male)

        m_idx = torch.nonzero(gender == 1).squeeze(-1)
        f_idx = torch.nonzero(gender == -1).squeeze(-1)

        m_bs, f_bs = len(m_idx) if m_idx.dim() > 0 else 0, len(f_idx) if f_idx.dim() > 0 else 0
        m_fr, f_fr = data["betas"].shape[1], data["betas"].shape[1]

        if m_bs > 0:
            # split male
            male_params = {k: v[m_idx] for k, v in smpl_params.items()}
            # squeeze parameter values in batch dimension
            male_params = {k: v.view(m_bs * m_fr, -1) for k, v in male_params.items() if k != 'gender'}
            # run smpl fk
            m_verts, m_joints = self.bm_male(poses_body=male_params['pose_body'],
                                             betas=male_params['betas'],
                                             poses_root=male_params['root_orient'],
                                             trans=male_params['trans'],
                                             skinning=skinning)
            # exclude hand joints
            m_joints = m_joints[:, :constants.SMPLH_BODY_JOINTS, :]
            # unsqueeze back the batch dimension
            m_verts, m_joints = (m_verts.view(m_bs, m_fr, -1, 3) if m_verts is not None else m_verts,
                                 m_joints.view(m_bs, m_fr, -1, 3))

        if f_bs > 0:
            # split female
            female_params = {k: v[f_idx] for k, v in smpl_params.items()}
            # squeeze parameter values in batch dimension
            female_params = {k: v.view(f_bs * f_fr, -1) for k, v in female_params.items() if k != 'gender'}
            # run smpl fk
            f_verts, f_joints = self.bm_female(poses_body=female_params['pose_body'],
                                               betas=female_params['betas'],
                                               poses_root=female_params['root_orient'],
                                               trans=female_params['trans'],
                                               skinning=skinning)
            # exclude hand joints
            f_joints = f_joints[:, :constants.SMPLH_BODY_JOINTS, :]
            # unsqueeze back the batch dimension
            f_verts, f_joints = (f_verts.view(f_bs, f_fr, -1, 3) if f_verts is not None else f_verts,
                                 f_joints.view(f_bs, f_fr, -1, 3))

        if m_bs > 0 and f_bs > 0:
            # join the f_verts and m_verts according to m_idx and f_idx
            if m_verts is not None and f_verts is not None:
                verts = torch.zeros_like(torch.concatenate((m_verts, f_verts), dim=0))
                verts[m_idx] = m_verts
                verts[f_idx] = f_verts
            else:
                verts = None

            joints = torch.zeros_like(torch.concatenate((m_joints, f_joints), dim=0))
            joints[m_idx] = m_joints
            joints[f_idx] = f_joints
        elif m_bs > 0:
            verts = m_verts
            joints = m_joints
        elif f_bs > 0:
            verts = f_verts
            joints = f_joints
        else:
            print("CHECK WHY ERROR HERE")
            import ipdb;
            ipdb.set_trace()

        # # Visualize the meshes
        # # save the mesh as an obj file
        # for i, vert in enumerate(verts[0]):
        #     import trimesh
        #     body_mesh = trimesh.Trimesh(vertices=c2c(vert), faces=c2c(self.bm_male.faces),
        #                                 vertex_colors=np.tile([255, 200, 200, 255], (6890, 1)))
        #     out_folder = f'./debug_mesh/0/'
        #     os.makedirs(out_folder, exist_ok=True)
        #     body_mesh.export(os.path.join(out_folder, f'body_mesh_{i:04d}.obj'))
        #     print(os.path.join(out_folder, f'body_mesh_{i:04d}.obj'))
        # print('------------------')
        #
        # # Visualize the meshes
        # # save the mesh as an obj file
        # for i, vert in enumerate(verts[1]):
        #     import trimesh
        #     body_mesh = trimesh.Trimesh(vertices=c2c(vert), faces=c2c(self.bm_male.faces),
        #                                 vertex_colors=np.tile([255, 200, 200, 255], (6890, 1)))
        #     out_folder = f'./debug_mesh/1/'
        #     os.makedirs(out_folder, exist_ok=True)
        #     body_mesh.export(os.path.join(out_folder, f'body_mesh_{i:04d}.obj'))
        #     print(os.path.join(out_folder, f'body_mesh_{i:04d}.obj'))
        # print('------------------')

        return verts, joints

    def get_hull_verts(self, joints, vertices):
        ## sample a random point on the mesh at a radius of 1 m from the CoM
        # setup biomechanical evaluator
        self.biomechanical_evaluator.joints = joints
        self.biomechanical_evaluator.vertices = vertices
        self.biomechanical_evaluator.init_per_part_buffers()
        self.biomechanical_evaluator.calculate_mass_per_vert()
        # compute com
        self.biomechanical_evaluator.calculate_com()
        # compute base of support
        status = self.biomechanical_evaluator.calculate_bos()
        if not status:
            return None, None
        # get hull vertices
        hull_verts = self.biomechanical_evaluator.hull_verts

        # project com to ground
        ground_coms = self.biomechanical_evaluator.coms
        ground_coms[:, :, 2] = 0.0
        return hull_verts, ground_coms

    def get_input_features(self, data, points):
        features = torch.cat([data[k].float() if k not in 'trans' else data[k][:, [-1]].float() for k in self.train_feats], dim=-1)
        features = torch.cat((features, data['identity']), dim=-1)
        features = torch.cat((features, points), dim=-1)
        return features

    def train(self, train_loader, skinning=True, epoch=0):
        self.model.train()
        # Training loop
        for iter, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            motion_x_dict_A = batch["motion_x_dict"]
            # Send to device
            for k, v in motion_x_dict_A.items():
                if type(v) == torch.Tensor:
                    motion_x_dict_A[k] = v.to(self.device).float()
            motion_x_dict_A = self.construct_input(motion_x_dict_A)

            ref_motions_A = motion_x_dict_A["x"]
            identity_A = motion_x_dict_A["identity"]  # these include betas + gender

            # unnormalize the motion features
            ref_motions_un_A = self.normalizer.inverse(self.deconstruct_input(ref_motions_A, identity_A))
            ref_motions_un_A = self.construct_input(ref_motions_un_A)

            # Run SMPLH FK to get the joints
            ref_verts_A, ref_joints_A = self.run_smpl_fk(ref_motions_un_A, skinning)

            # get hull vertices
            bs, f, _, _ = ref_verts_A.shape
            hull_verts_A = []
            ground_coms_A = []

            for b in range(bs):
                hull_vert, ground_coms = self.get_hull_verts(ref_joints_A[b], ref_verts_A[b])
                if hull_vert is None:
                    continue
                hull_verts_A.append(hull_vert)
                ground_coms_A.append(ground_coms)
                # Todo: collect bs where hull_verts were returned
                print("Todo: collect bs where hull_verts were returned")
                import ipdb; ipdb.set_trace()

            if len(hull_verts_A) == 0:
                continue
            # repeat the last hull_vert to match the batch size
            if len(hull_verts_A) < bs:
                # Todo: collect indices where hull_verts was returned
                print("Todo: collect indices where hull_verts was returned")
                import ipdb; ipdb.set_trace()

            ground_coms_A = torch.stack(ground_coms_A, dim=0)


            # Reduce the batch size to match the number of hull_verts
            ground_coms_A = ground_coms_A[:new_bs]
            ref_verts_A = ref_verts_A[:new_bs]
            ref_joints_A = ref_joints_A[:new_bs]
            for k, v in motion_x_dict_A.items():
                if len(v.shape) > 1:
                    motion_x_dict_A[k] = v[:new_bs]


            # sample bs * f points
            # sample 1000 points
            radius = 1
            r = torch.rand(new_bs, f, 1).to(self.device) * radius
            theta = torch.rand(new_bs, f, 1).to(self.device) * 2 * np.pi
            try:
                x = ground_coms_A[:, :, :, 0] + r * torch.cos(theta)
                y = ground_coms_A[:, :, :, 1] + r * torch.sin(theta)
                z = torch.zeros_like(x)
            except:
                import ipdb;
                ipdb.set_trace()
            points = torch.cat((x, y, z), dim=-1)

            # combine batch and frames dimension
            ref_verts_A = ref_verts_A.view(new_bs * f, -1, 3)
            ref_joints_A = ref_joints_A.view(new_bs * f, -1, 3)
            for k, v in motion_x_dict_A.items():
                if len(v.shape) > 1:
                    motion_x_dict_A[k] = v.view(new_bs * f, -1)
            points = points.view(new_bs * f, 3)
            hull_verts_A = [item for sublist in hull_verts_A for item in sublist]

            # shuffle along frame dimension
            shuffle_idx = torch.randperm(new_bs*f)
            ref_verts_A = ref_verts_A[shuffle_idx, ...]
            ref_joints_A = ref_joints_A[shuffle_idx, ...]
            for k, v in motion_x_dict_A.items():
                if len(v.shape)>1:
                    motion_x_dict_A[k] = v[shuffle_idx, ...]
            points = points[shuffle_idx, ...]
            hull_verts_A = [hull_verts_A[i] for i in shuffle_idx]
            input = self.get_input_features(motion_x_dict_A, points)
            model_output = self.model(input)

            _, min_signed_distance = min_signed_distance_to_convex_hull(points[:, None, :], hull_verts_A)
            # compute loss
            loss = self.criterion(model_output.squeeze(), min_signed_distance)
            # zero out the loss whenever min_signed_distance is 0
            loss[min_signed_distance == 0] = 0.0
            loss = loss.mean() * 1000

            wandb.log({"train_loss": loss.item()})

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log loss
            logger.info(f'Epoch: {epoch}, Loss: {loss.item()}')

    def validate(self, val_loader, skinning=True, epoch=0):
        if epoch % self.hparams.TRAINING.CHECKPOINT_EPOCHS == 0:
            self.model.eval()
            with torch.no_grad():
                total_val_loss = 0.0
                iter_count = 0
                # Val loop
                for iter, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                    motion_x_dict_A = batch["motion_x_dict"]
                    # Send to device
                    for k, v in motion_x_dict_A.items():
                        if type(v) == torch.Tensor:
                            motion_x_dict_A[k] = v.to(self.device).float()
                    motion_x_dict_A = self.construct_input(motion_x_dict_A)

                    ref_motions_A = motion_x_dict_A["x"]
                    identity_A = motion_x_dict_A["identity"]  # these include betas + gender

                    # unnormalize the motion features
                    ref_motions_un_A = self.normalizer.inverse(self.deconstruct_input(ref_motions_A, identity_A))
                    ref_motions_un_A = self.construct_input(ref_motions_un_A)

                    # Run SMPLH FK to get the joints
                    ref_verts_A, ref_joints_A = self.run_smpl_fk(ref_motions_un_A, skinning)

                    # get hull vertices
                    bs, f, _, _ = ref_verts_A.shape
                    hull_verts_A = []
                    ground_coms_A = []

                    for b in range(bs):
                        hull_vert, ground_coms = self.get_hull_verts(ref_joints_A[b], ref_verts_A[b])
                        if hull_vert is None:
                            continue
                        hull_verts_A.append(hull_vert)
                        ground_coms_A.append(ground_coms)

                    if len(hull_verts_A) == 0:
                        continue
                    # repeat the last hull_vert to match the batch size
                    if len(hull_verts_A) < bs:
                        new_bs = len(hull_verts_A)

                    ground_coms_A = torch.stack(ground_coms_A, dim=0)

                    # Reduce the batch size to match the number of hull_verts
                    ground_coms_A = ground_coms_A[:new_bs]
                    ref_verts_A = ref_verts_A[:new_bs]
                    ref_joints_A = ref_joints_A[:new_bs]
                    for k, v in motion_x_dict_A.items():
                        if len(v.shape) > 1:
                            motion_x_dict_A[k] = v[:new_bs]

                    # sample bs * f points
                    # sample 1000 points
                    radius = 1
                    r = torch.rand(new_bs, f, 1).to(self.device) * radius
                    theta = torch.rand(new_bs, f, 1).to(self.device) * 2 * np.pi
                    x = ground_coms_A[:, :, :, 0] + r * torch.cos(theta)
                    y = ground_coms_A[:, :, :, 1] + r * torch.sin(theta)
                    z = torch.zeros_like(x)
                    points = torch.cat((x, y, z), dim=-1)

                    # combine batch and frames dimension
                    ref_verts_A = ref_verts_A.view(new_bs * f, -1, 3)
                    ref_joints_A = ref_joints_A.view(new_bs * f, -1, 3)
                    for k, v in motion_x_dict_A.items():
                        if len(v.shape) > 1:
                            motion_x_dict_A[k] = v.view(new_bs * f, -1)
                    points = points.view(new_bs * f, 3)
                    hull_verts_A = [item for sublist in hull_verts_A for item in sublist]

                    # shuffle along frame dimension
                    shuffle_idx = torch.randperm(new_bs*f)
                    ref_verts_A = ref_verts_A[shuffle_idx, ...]
                    ref_joints_A = ref_joints_A[shuffle_idx, ...]
                    for k, v in motion_x_dict_A.items():
                        if len(v.shape)>1:
                            motion_x_dict_A[k] = v[shuffle_idx, ...]
                    points = points[shuffle_idx, ...]
                    hull_verts_A = [hull_verts_A[i] for i in shuffle_idx]
                    input = self.get_input_features(motion_x_dict_A, points)
                    model_output = self.model(input)

                    _, min_signed_distance = min_signed_distance_to_convex_hull(points[:, None, :], hull_verts_A)
                    # compute loss
                    loss = self.criterion(model_output.squeeze(), min_signed_distance)
                    # zero out the loss whenever min_signed_distance is 0
                    loss[min_signed_distance == 0] = 0.0
                    loss = loss.mean() * 1000
                    total_val_loss += loss.item()
                    iter_count += 1

                wandb.log({"val_loss": total_val_loss / iter_count})

            # save model
            out_dir = os.path.join(self.hparams.OUTPUT_DIR, self.hparams.EXP_NAME)
            os.makedirs(out_dir, exist_ok=True)
            if epoch % self.hparams.TRAINING.CHECKPOINT_EPOCHS == 0:
                torch.save(self.model.state_dict(),
                           osp.join(out_dir, f'{self.hparams.EXP_NAME}_BoSDistModel_epoch_{epoch}.pth'))
                print(f'Saved model at epoch {epoch} to {out_dir}')

            # early stopping
            if epoch >= self.hparams.TRAINING.NUM_EARLY_STOP:
                logger.info('Early stopping')
                torch.save(self.model.state_dict(),
                           osp.join(out_dir, f'{self.hparams.EXP_NAME}_BoSDistModel_FINAL.pth'))
                print(f'Saved model at epoch {epoch} to {out_dir}')
                import sys; sys.exit(0)










def fit(hparams, train_loader, val_loader, model, biomechanical_evaluator, criterion, optimizer):
    # Training loop
    for epoch in range(hparams.TRAINING.NUM_EPOCHS):
        for iter, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            bs = len(batch["motion_x_dict"]["trans"])
            motion_x_dict_A = batch["motion_x_dict"]
            # Send to device
            for k, v in motion_x_dict_A.items():
                if type(v) == torch.Tensor:
                    item[k] = v.to(hparams.DEVICE).float()
            # motion_x_dict_A =

            # forward pass
            model.train()

            # run smpl forward
            # set up smpl forward
            smplx_output = body_model(betas=betas,
                                      transl=transl,
                                      global_orient=poses[:, :3],
                                      body_pose=poses[:, 3:66])
            vertices = smplx_output.vertices

            ## sample a random point on the mesh at a radius of 1 m from the CoM
            # setup biomechanical evaluator
            biomechanical_evaluator.joints = smplx_output.joints
            biomechanical_evaluator.vertices = smplx_output.vertices
            biomechanical_evaluator.init_per_part_buffers()
            biomechanical_evaluator.body_pose.data = poses[:, 3:66]
            biomechanical_evaluator.calculate_mass_per_vert()
            # compute com
            biomechanical_evaluator.calculate_com()
            # compute base of support
            status = biomechanical_evaluator.calculate_bos()
            if not status:
                continue
            # get hull vertices
            hull_verts = biomechanical_evaluator.hull_verts

            # project com to ground
            coms = biomechanical_evaluator.coms
            coms[:, :, 2] = 0.0
            # sample 1000 points
            n_points = 1
            radius = 1
            r = torch.rand(batch_size, n_points).to(hparams.DEVICE) * radius
            theta = torch.rand(batch_size, n_points).to(hparams.DEVICE) * 2 * np.pi
            x = coms[:, :, 0] + r * torch.cos(theta)
            y = coms[:, :, 1] + r * torch.sin(theta)
            z = torch.zeros_like(x)
            points = torch.stack((x, y, z), dim=-1)

            # combine points and vertices
            for point_iter in range(n_points):
                # input = torch.cat((vertices, points[:, [point_iter], :]), dim=1).reshape(batch_size, -1)
                input = torch.cat((poses[:, :66], transl, points[:, point_iter, :]), dim=1).reshape(batch_size, -1)
                input = torch.cat((input, betas), dim=1).reshape(batch_size, -1)

                model_output = model(input)
                _, min_signed_distance = min_signed_distance_to_convex_hull(points[:, [point_iter], :], hull_verts)
                # compute loss
                loss = criterion(model_output.squeeze(), min_signed_distance)
                # zero out the loss whenever min_signed_distance is 0
                loss[min_signed_distance == 0] = 0.0
                loss = loss.mean() * 1000

                wandb.log({"train_loss": loss.item()})

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # log loss
                logger.info(f'Epoch: {epoch}, Loss: {loss.item()}')

        # save model
        out_dir = os.path.join(hparams.OUTPUT_DIR, hparams.EXP_NAME)
        os.makedirs(out_dir, exist_ok=True)
        if epoch % hparams.TRAINING.CHECKPOINT_EPOCHS == 0:
            torch.save(model.state_dict(),
                       osp.join(out_dir, f'{hparams.EXP_NAME}_BoSDistModel_epoch_{epoch}.pth'))
            print(f'Saved model at epoch {epoch} to {out_dir}')

        # early stopping
        if epoch >= hparams.TRAINING.NUM_EARLY_STOP:
            logger.info('Early stopping')
            torch.save(model.state_dict(),
                       osp.join(out_dir, f'{hparams.EXP_NAME}_bos_dist_model_final.pth'))
            print(f'Saved model at epoch {epoch} to {out_dir}')
            break

        #        Evaluate after every 5 epochs
        if epoch % hparams.TRAINING.CHECKPOINT_EPOCHS == 0:
            model.eval()
            with torch.no_grad():
                total_val_loss = 0.0
                iter_count = 0
                for iter, item in tqdm(enumerate(val_loader), total=len(val_loader)):
                    for k, v in item.items():
                        if type(v) == torch.Tensor:
                            item[k] = v.to(hparams.DEVICE).float()

                    poses = item['poses'][:, :66]  # TODO: check if you only want the body parameters
                    betas = item['betas']
                    transl = item['trans']
                    # Make x and y translation 0
                    transl[:, 0] = 0.0
                    transl[:, 1] = 0.0
                    batch_size = poses.shape[0]

                    pnames = item['pname']

                    # run smplx forward
                    # set up smpl forward
                    smplx_output = body_model(betas=betas,
                                              transl=transl,
                                              global_orient=poses[:, :3],
                                              body_pose=poses[:, 3:66])
                    vertices = smplx_output.vertices

                    ## sample a random point on the mesh at a radius of 1 m from the CoM
                    # setup biomechanical evaluator
                    biomechanical_evaluator.joints = smplx_output.joints
                    biomechanical_evaluator.vertices = smplx_output.vertices
                    biomechanical_evaluator.init_per_part_buffers()
                    biomechanical_evaluator.body_pose.data = poses[:, 3:66]
                    biomechanical_evaluator.calculate_mass_per_vert()
                    # compute com
                    biomechanical_evaluator.calculate_com()
                    # compute base of support
                    status = biomechanical_evaluator.calculate_bos()
                    if not status:
                        continue
                    # compute zmp
                    # biomechanical_evaluator.calculate_zmps()
                    # get hull vertices
                    hull_verts = biomechanical_evaluator.hull_verts

                    # project com to ground
                    coms = biomechanical_evaluator.coms
                    coms[:, :, 2] = 0.0
                    # sample 1000 points
                    n_points = 1
                    radius = 1
                    r = torch.rand(batch_size, n_points).to(hparams.DEVICE) * radius
                    theta = torch.rand(batch_size, n_points).to(hparams.DEVICE) * 2 * np.pi
                    x = coms[:, :, 0] + r * torch.cos(theta)
                    y = coms[:, :, 1] + r * torch.sin(theta)
                    z = torch.zeros_like(x)
                    points = torch.stack((x, y, z), dim=-1)

                    # combine points and vertices
                    for point_iter in range(n_points):
                        # input = torch.cat((vertices, points[:, [point_iter], :]), dim=1).reshape(batch_size, -1)
                        input = torch.cat((poses[:, :66], transl, points[:, point_iter, :]), dim=1).reshape(batch_size,
                                                                                                            -1)
                        input = torch.cat((input, betas), dim=1).reshape(batch_size, -1)
                        model_output = model(input)
                        _, min_signed_distance = min_signed_distance_to_convex_hull(points[:, [point_iter], :],
                                                                                    hull_verts)
                        # compute loss
                        loss = criterion(model_output.squeeze(), min_signed_distance)
                        # zero out the loss whenever min_signed_distance is 0
                        loss[min_signed_distance == 0] = 0.0
                        loss = loss.mean() * 1000
                        total_val_loss += loss.item()
                        iter_count += 1

                wandb.log({"val_loss": total_val_loss / iter_count})
