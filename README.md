# HUMOS: Human Motion Model Conditioned on Body Shape [ECCV 2024]

> Code repository for the paper:  
> [**HUMOS: Human Motion Model Conditioned on Body Shape**](https://carstenepic.github.io/humos/)  
> [Shashank Tripathi](https://sha2nkt.github.io/), [Omid Taheri](https://otaheri.github.io/), [Christoph Lassner](https://christophlassner.de/), [Michael J. Black](https://ps.is.mpg.de/person/black), [Daniel Holden](https://theorangeduck.com/), [Carsten Stoll](https://carstenstoll.github.io/)<br />
> *European Conference on Computer Vision (ECCV), 2024*

[![arXiv](https://img.shields.io/badge/arXiv-2309.15273-00ff00.svg)](https://arxiv.org/abs/2409.03944)  [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://carstenepic.github.io/humos/)     

[//]: # ([![Open In Colab]&#40;https://colab.research.google.com/assets/colab-badge.svg&#41;]&#40;https://colab.research.google.com/drive/1fTQdI2AHEKlwYG9yIb2wqicIMhAa067_?usp=sharing&#41;  [![Hugging Face Spaces]&#40;https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue&#41;]&#40;https://huggingface.co/spaces/ac5113/DECO&#41;)

![teaser](website/static/teaser/teaser_humos_flat.gif)

[[Project Page](https://carstenepic.github.io/humos/)] [[Paper](https://arxiv.org/abs/2409.03944)] [[Video](https://www.youtube.com/watch?v=yLXX7TxBA4o)] [[Poster](https://www.dropbox.com/scl/fi/nxtj4svwe5dcfvaffou0u/ECCV2024_HUMOS_Poster_v2.pdf?rlkey=3cku1bxgio9ec7o4bumetqiu7&e=1&st=un1ub1c9&dl=0)] [[License]()] [[Contact](mailto:shashank.tripathi123@gmail.com)]

[//]: # (## News :triangular_flag_on_post:)

[//]: # ()
[//]: # (- [2024/05/28] :eight_pointed_black_star: Damon object-wise contacts are released in SMPL and SMPL-X format. Please refer [here]&#40;#damon-data-description&#41; for details. )

[//]: # (- [2024/01/31] The DAMON contact labels in SMPL-X format have been released. [This]&#40;#convert-damon&#41; is the conversion script.)

[//]: # (- [2023/10/12] The [huggingface demo]&#40;https://huggingface.co/spaces/ac5113/DECO&#41; has been released.)

[//]: # (- [2023/10/10] The [colab demo]&#40;https://colab.research.google.com/drive/1fTQdI2AHEKlwYG9yIb2wqicIMhAa067_?usp=sharing&#41; has been released. Huggingface demo coming soon...)

## Installation and Setup

(Coming soon)

[//]: # (git clone --recursive https://github.com/......)

[//]: # (git submodule update --init --recursive)

[//]: # ()
[//]: # (1. mamba create -n humos_p310 python=3.10)

[//]: # (2. mamba activate humos_p310)

[//]: # (3. mamba install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 -c pytorch)

[//]: # (4. Install Aitviewer from source)

[//]: # (   5. )

[//]: # (5. pip install -r reqruiements.txt # for packages not supported by poetry)

[//]: # ()
[//]: # (### Download the SMPL+H model)

[//]: # (Go to the [MANO website]&#40;https://mano.is.tue.mpg.de/download.php&#41;, register and go to the Download tab.)

[//]: # ()
[//]: # (- Click on "Models & Code" to download ``mano_v1_2.zip`` and place it in the folder ``body_models/smplh/``.)

[//]: # (- Click on "Extended SMPL+H model" to download ``smplh.tar.xz`` and place it in the folder ``body_models/smplh/``.)

[//]: # ()
[//]: # (The next step is to extract the archives, merge the hands from ``mano_v1_2`` into the ``Extended SMPL+H models``, and remove any chumpy dependency.)

[//]: # (All of this can be done using with the following commands.)

[//]: # ()
[//]: # ()
[//]: # (```bash)

[//]: # (bash humos/prepare/smplh.sh)

[//]: # (```)

[//]: # ()
[//]: # (This will create ``SMPLH_FEMALE.npz``, ``SMPLH_MALE.npz``, ``SMPLH_NEUTRAL.npz`` inside the ``body_models/smplh`` folder.)

[//]: # ()
[//]: # (1. First, clone the repo. Then, we recommend creating a clean [conda]&#40;https://docs.conda.io/&#41; environment, activating it and installing torch and torchvision, as follows:)

[//]: # (```shell)

[//]: # (git clone https://github.com/sha2nkt/deco.git)

[//]: # (cd deco)

[//]: # (conda create -n deco python=3.9 -y)

[//]: # (conda activate deco)

[//]: # (pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117)

[//]: # (```)

[//]: # (Please adjust the CUDA version as required.)

[//]: # ()
[//]: # (2. Install PyTorch3D from source. Users may also refer to [PyTorch3D-install]&#40;https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md&#41; for more details.)

[//]: # (However, our tests show that installing using ``conda`` sometimes runs into dependency conflicts.)

[//]: # (Hence, users may alternatively install Pytorch3D from source following the steps below.)

[//]: # (```shell)

[//]: # (git clone https://github.com/facebookresearch/pytorch3d.git)

[//]: # (cd pytorch3d)

[//]: # (pip install .)

[//]: # (cd ..)

[//]: # (```)

[//]: # ()
[//]: # (3. Install the other dependancies and download the required data.)

[//]: # (```bash)

[//]: # (pip install -r requirements.txt)

[//]: # (sh fetch_data.sh)

[//]: # (```)

[//]: # ()
[//]: # (4. Please download [SMPL]&#40;https://smpl.is.tue.mpg.de/&#41; &#40;version 1.1.0&#41; and [SMPL-X]&#40;https://smpl-x.is.tue.mpg.de/&#41; &#40;v1.1&#41; files into the data folder. Please rename the SMPL files to ```SMPL_FEMALE.pkl```, ```SMPL_MALE.pkl``` and ```SMPL_NEUTRAL.pkl```. The directory structure for the ```data``` folder has been elaborated below:)

[//]: # ()
[//]: # (```)

[//]: # (├── preprocess)

[//]: # (├── smpl)

[//]: # (│   ├── SMPL_FEMALE.pkl)

[//]: # (│   ├── SMPL_MALE.pkl)

[//]: # (│   ├── SMPL_NEUTRAL.pkl)

[//]: # (│   ├── smpl_neutral_geodesic_dist.npy)

[//]: # (│   ├── smpl_neutral_tpose.ply)

[//]: # (│   ├── smplpix_vertex_colors.npy)

[//]: # (├── smplx)

[//]: # (│   ├── SMPLX_FEMALE.npz)

[//]: # (│   ├── SMPLX_FEMALE.pkl)

[//]: # (│   ├── SMPLX_MALE.npz)

[//]: # (│   ├── SMPLX_MALE.pkl)

[//]: # (│   ├── SMPLX_NEUTRAL.npz)

[//]: # (│   ├── SMPLX_NEUTRAL.pkl)

[//]: # (│   ├── smplx_neutral_tpose.ply)

[//]: # (├── weights)

[//]: # (│   ├── pose_hrnet_w32_256x192.pth)

[//]: # (├── J_regressor_extra.npy)

[//]: # (├── base_dataset.py)

[//]: # (├── mixed_dataset.py)

[//]: # (├── smpl_partSegmentation_mapping.pkl)

[//]: # (├── smpl_vert_segmentation.json)

[//]: # (└── smplx_vert_segmentation.json)

[//]: # (```)

[//]: # (<a name="damon-data-description"></a>)

[//]: # (### Download the DAMON dataset)

[//]: # ()
[//]: # (⚠️ Register account on the [DECO website]&#40;https://deco.is.tue.mpg.de/register.php&#41;, and then use your username and password to login to the _Downloads_ page.)

[//]: # ()
[//]: # (Follow the instructions on the _Downloads_ page to download the DAMON dataset. The provided metadata in the `npz` files is described as follows: )

[//]: # (- `imgname`: relative path to the image file)

[//]: # (- `pose` : SMPL pose parameters inferred from [CLIFF]&#40;https://github.com/huawei-noah/noah-research/tree/master/CLIFF&#41;)

[//]: # (- `transl` : SMPL root translation inferred from [CLIFF]&#40;https://github.com/huawei-noah/noah-research/tree/master/CLIFF&#41;)

[//]: # (- `shape` : SMPL shape parameters inferred from [CLIFF]&#40;https://github.com/huawei-noah/noah-research/tree/master/CLIFF&#41;)

[//]: # (- `cam_k` : camera intrinsic matrix inferred from [CLIFF]&#40;https://github.com/huawei-noah/noah-research/tree/master/CLIFF&#41;)

[//]: # (- `polygon_2d_contact`: 2D contact annotation from [HOT]&#40;https://hot.is.tue.mpg.de/&#41;)

[//]: # (- `contact_label`: 3D contact annotations on the SMPL mesh)

[//]: # (- `contact_label_smplx`: 3D contact annotation on the SMPL-X mesh)

[//]: # (- `contact_label_objectwise`: 3D contact annotations split into separate object labels on the SMPL mesh)

[//]: # (- `contact_label_smplx_objectwise`: 3D contact annotations split into separate object labels on the SMPL-X mesh)

[//]: # (- `scene_seg`: path to the scene segmentation map from [Mask2Former]&#40;https://github.com/facebookresearch/Mask2Former&#41;)

[//]: # (- `part_seg`: path to the body part segmentation map)

[//]: # ()
[//]: # (The order of values is the same for all the keys. )

[//]: # ()
[//]: # (<a name="convert-damon"></a>)

[//]: # (#### Converting DAMON contact labels to SMPL-X format &#40;and back&#41;)

[//]: # ()
[//]: # (To convert contact labels from SMPL to SMPL-X format and vice-versa, run the following command)

[//]: # (```bash)

[//]: # (python reformat_contacts.py \)

[//]: # (    --contact_npz datasets/Release_Datasets/damon/hot_dca_trainval.npz \)

[//]: # (    --input_type 'smpl')

[//]: # (```)

[//]: # ()
[//]: # (## Run demo on images)

[//]: # (The following command will run DECO on all images in the specified `--img_src`, and save rendering and colored mesh in `--out_dir`. The `--model_path` flag is used to specify the specific checkpoint being used. Additionally, the base mesh color and the color of predicted contact annotation can be specified using the `--mesh_colour` and `--annot_colour` flags respectively. )

[//]: # (```bash)

[//]: # (python inference.py \)

[//]: # (    --img_src example_images \)

[//]: # (    --out_dir demo_out)

[//]: # (```)

[//]: # ()
[//]: # (## Training and Evaluation)

[//]: # ()
[//]: # (We release 3 versions of the DECO model:)

[//]: # (<ol>)

[//]: # (    <li> DECO-HRNet &#40;<em> Best performing model </em>&#41; </li>)

[//]: # (    <li> DECO-HRNet w/o context branches </li>)

[//]: # (    <li> DECO-Swin </li>)

[//]: # (</ol>)

[//]: # ()
[//]: # (All the checkpoints have been downloaded to ```checkpoints```. )

[//]: # (However, please note that versions 2 and 3 have been trained solely on the RICH dataset. <br>)

[//]: # (We recommend using the first DECO version.)

[//]: # ()
[//]: # (Please download the actual DAMON dataset from the website and place it in ```datasets/Release_Datasets``` following the instructions given.)

[//]: # ()
[//]: # (### Evaluation)

[//]: # (To run evaluation on the DAMON dataset, please run the following command:)

[//]: # ()
[//]: # (```bash)

[//]: # (python tester.py --cfg configs/cfg_test.yml)

[//]: # (```)

[//]: # ()
[//]: # (### Training)

[//]: # (The config provided &#40;```cfg_train.yml```&#41; is set to train and evaluate on all three datasets: DAMON, RICH and PROX. To change this, please change the value of the key ```TRAINING.DATASETS``` and ```VALIDATION.DATASETS``` in the config &#40;please also change ```TRAINING.DATASET_MIX_PDF``` as required&#41;. <br>)

[//]: # (Also, the best checkpoint is stored by default at ```checkpoints/Other_Checkpoints```.)

[//]: # (Please run the following command to start training of the DECO model:)

[//]: # ()
[//]: # (```bash)

[//]: # (python train.py --cfg configs/cfg_train.yml)

[//]: # (```)

[//]: # ()
[//]: # (### Training on custom datasets)

[//]: # ()
[//]: # (To train on other datasets, please follow these steps:)

[//]: # (1. Please create an npz of the dataset, following the structure of the datasets in ```datasets/Release_Datasets``` with the corresponding keys and values.)

[//]: # (2. Please create scene segmentation maps, if not available. We have used [Mask2Former]&#40;https://github.com/facebookresearch/Mask2Former&#41; in our work.)

[//]: # (3. For creating the part segmentation maps, this [sample script]&#40;https://github.com/sha2nkt/deco/blob/main/utils/get_part_seg_mask.py&#41; can be referred to.)

[//]: # (4. Add the dataset name&#40;s&#41; to ```train.py``` &#40;[these lines]&#40;https://github.com/sha2nkt/deco/blob/d5233ecfad1f51b71a50a78c0751420067e82c02/train.py#L83&#41;&#41;, ```tester.py``` &#40;[these lines]&#40;https://github.com/sha2nkt/deco/blob/d5233ecfad1f51b71a50a78c0751420067e82c02/tester.py#L51&#41;&#41; and ```data/mixed_dataset.py``` &#40;[these lines]&#40;https://github.com/sha2nkt/deco/blob/d5233ecfad1f51b71a50a78c0751420067e82c02/data/mixed_dataset.py#L17&#41;&#41;, according to the body model being used &#40;SMPL/SMPL-X&#41;)

[//]: # (5. Add the path&#40;s&#41; to the dataset npz&#40;s&#41; to ```common/constants.py``` &#40;[these lines]&#40;https://github.com/sha2nkt/deco/blob/d5233ecfad1f51b71a50a78c0751420067e82c02/common/constants.py#L19&#41;&#41;.)

[//]: # (6. Finally, change ```TRAINING.DATASETS``` and ```VALIDATION.DATASETS``` in the config file and you're good to go!)

[//]: # ()
[//]: # (## Citing)

[//]: # (If you find this code useful for your research, please consider citing the following paper:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@InProceedings{tripathi2023deco,)

[//]: # (    author    = {Tripathi, Shashank and Chatterjee, Agniv and Passy, Jean-Claude and Yi, Hongwei and Tzionas, Dimitrios and Black, Michael J.},)

[//]: # (    title     = {{DECO}: Dense Estimation of {3D} Human-Scene Contact In The Wild},)

[//]: # (    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision &#40;ICCV&#41;},)

[//]: # (    month     = {October},)

[//]: # (    year      = {2023},)

[//]: # (    pages     = {8001-8013})

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (### License)

[//]: # ()
[//]: # (See [LICENSE]&#40;LICENSE&#41;.)

[//]: # ()
[//]: # (### Acknowledgments)

[//]: # ()
[//]: # (We sincerely thank Alpar Cseke for his contributions to DAMON data collection and PHOSA evaluations, Sai K. Dwivedi for facilitating PROX downstream experiments, Xianghui Xie for his generous help with CHORE evaluations, Lea Muller for her help in initiating the contact annotation tool, Chun-Hao P. Huang for RICH discussions and Yixin Chen for details about the HOT paper. We are grateful to Mengqin Xue and Zhenyu Lou for their collaboration in BEHAVE evaluations, Joachim Tesch and Nikos Athanasiou for insightful visualization advice, and Tsvetelina Alexiadis for valuable data collection guidance. Their invaluable contributions enriched this research significantly. We also thank Benjamin Pellkofer for help with the website and IT support. This work was funded by the International Max Planck Research School for Intelligent Systems &#40;IMPRS-IS&#41;.)

[//]: # ()
[//]: # (### Contact)

[//]: # ()
[//]: # (For technical questions, please create an issue. For other questions, please contact `deco@tue.mpg.de`.)

[//]: # ()
[//]: # (For commercial licensing, please contact `ps-licensing@tue.mpg.de`.)

[//]: # ()
[//]: # (### Acknowlegements)

[//]: # ()
[//]: # (TEMOS)
