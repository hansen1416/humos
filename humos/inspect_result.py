import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as sRot

from humos.utils import constants
from humos.utils.fk import ForwardKinematicsLayer
from humos.utils.mesh_utils import smplh_breakdown

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fk_male = ForwardKinematicsLayer(
    constants.SMPL_PATH,
    gender="male",
    num_joints=constants.SMPLH_BODY_JOINTS,
    device=device,
)

d = torch.load("./debug_pred/000000_B_giv_A.pt", map_location="cpu")


def add_batch_dim(pred_un: dict) -> dict:
    out = {}
    for k, v in pred_un.items():
        if torch.is_tensor(v):
            out[k] = v.unsqueeze(0)  # prepend batch dim
        else:
            out[k] = v
    return out


pred_un = d["pred_un"]

pred_un_batched = add_batch_dim(pred_un)

print("=============================================")
print(pred_un_batched.keys())

for k, v in pred_un_batched.items():
    if torch.is_tensor(v):
        print(k, v.shape, v.dtype)


# print(pred_un_batched["betas"].shape)

for k, v in pred_un_batched.items():
    print(k)
    print(v.shape)


smpl_params = smplh_breakdown(pred_un_batched, fk=fk_male)

print(smpl_params.keys())

smpl_params = {k: v[0] for k, v in smpl_params.items()}

for k, v in smpl_params.items():
    print(k)
    print(v.shape)

# torch.save(smpl_params, "./debug_pred_un/000000_B_giv_A_aa.pt")

""""
Those keys are the **canonical SMPLH-control representation** HUMOS trains on (after `deconstruct_input` rehydrates the flattened feature vector and re-attaches identity). Interpreting each item in your dump:
 
* **`root_orient_6d_root_rel`** — shape **[T=200, 6]**
  Per-frame **global/root orientation** of the body, represented in **6D rotation representation** (the “two-column” 6D rep popularized to avoid discontinuities vs. Euler/quaternion).
  “`root_rel`” indicates it is expressed in the model’s chosen root-relative convention (in this repo, this field is the root orientation channel).
  This is the orientation for the SMPLH root joint (pelvis/root). 

* **`pose_6d_root_rel`** — shape **[T=200, J=21, 6]**
  Per-frame **local joint rotations** for the body (excluding hands), again in **6D rotation rep**.
  Here **21** matches the SMPLH “body joints” count used in this code path (`constants.SMPLH_BODY_JOINTS` appears in FK usage). So this tensor is the pose for those 21 joints, each as a 6D rotation, for each of 200 frames. 

* **`trans`** — shape **[T=200, 3]**
  Per-frame **root translation** (x,y,z) in the dataset/model coordinate system. This is the global motion trajectory.

* **`betas`** — shape **[T=200, 10]**
  The **body shape parameters** (SMPL betas). In your current pipeline they are replicated over time (constant across frames, but stored as [T,10] for convenience).
  In `deconstruct_input`, these are taken from `identity[:, :, :-1]`, so they reflect the identity conditioning you passed in (A or B). 

* **`gender`** — shape **[T=200, 1]**, dtype float64
  A **gender code** per frame (again time-replicated). In this repo it is typically **+1 for male** and **-1 for female** (see how `run_smpl_fk()` splits with `gender == 1` and `gender == -1`). 
  The float64 here is incidental (likely coming from how it was constructed/loaded); semantically it is just a scalar indicator.


"""


"""

### What is “6D rotation representation”?

The 6D representation encodes a rotation matrix (R \in \mathbb{R}^{3\times 3}) using its **first two columns**:

[
a = R_{:,0} \in \mathbb{R}^3,\quad b = R_{:,1} \in \mathbb{R}^3 ;;\Rightarrow;; [a; b] \in \mathbb{R}^6
]

To recover a valid rotation, you **orthonormalize** (Gram–Schmidt):

1. (u = \frac{a}{|a|})
2. (v = \frac{b - (u^\top b)u}{|b - (u^\top b)u|})
3. (w = u \times v)
4. (R = [u; v; w])

So the network outputs 6 numbers, and you project them onto SO(3) deterministically.

### Why 6D is preferred in motion models

* **Continuous and stable** over the rotation space (no angle wrap-around like Euler).
* **No hard constraint** during regression (unlike unit quaternions), yet you still get a valid rotation after projection.
* Empirically yields **smoother training** and fewer flips/jitter in generated motion.



"""
