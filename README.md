# HUMOS: Human Motion Model Conditioned on Body Shape [ECCV 2024]

> Code repository for the paper:  
> [**HUMOS: Human Motion Model Conditioned on Body Shape**](https://carstenepic.github.io/humos/)  
> [Shashank Tripathi](https://sha2nkt.github.io/), [Omid Taheri](https://otaheri.github.io/), [Christoph Lassner](https://christophlassner.de/), [Michael J. Black](https://ps.is.mpg.de/person/black), [Daniel Holden](https://theorangeduck.com/), [Carsten Stoll](https://carstenstoll.github.io/)<br />
> *European Conference on Computer Vision (ECCV), 2024*

[![arXiv](https://img.shields.io/badge/arXiv-2309.15273-00ff00.svg)](https://arxiv.org/abs/2409.03944)  [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://carstenepic.github.io/humos/)     

[//]: # ([![Open In Colab]&#40;https://colab.research.google.com/assets/colab-badge.svg&#41;]&#40;https://colab.research.google.com/drive/1fTQdI2AHEKlwYG9yIb2wqicIMhAa067_?usp=sharing&#41;  [![Hugging Face Spaces]&#40;https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue&#41;]&#40;https://huggingface.co/spaces/ac5113/DECO&#41;)

![teaser](website/static/teaser/teaser_humos_flat.gif)

[[Project Page](https://carstenepic.github.io/humos/)] [[Paper](https://arxiv.org/abs/2409.03944)] [[Video](https://www.youtube.com/watch?v=yLXX7TxBA4o)] [[Poster](https://www.dropbox.com/scl/fi/nxtj4svwe5dcfvaffou0u/ECCV2024_HUMOS_Poster_v2.pdf?rlkey=3cku1bxgio9ec7o4bumetqiu7&e=1&st=un1ub1c9&dl=0)] [[License]()] [[Contact](mailto:shashank.tripathi123@gmail.com)]

## News :triangular_flag_on_post:

- [2024/12/10] Released training and inference code for HUMOS. DEMO code coming soon...

## Installation and Setup

1. First, clone the repo. Then, we recommend creating a clean [conda](https://docs.conda.io/) environment, activating it and installing torch and torchvision, as follows:
```shell
git clone --recursive https://github.com/sha2nkt/humos_website_backend.git
git submodule update --init --recursive
cd humos_website_backend
conda create -n humos_p310 python=3.10
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```
2. Install AitViewer from our custom fork
```shell
cd aitviewer_humos
pip install -e .
cd ..
```
3. Install the other dependencies
```shell
pip install -r requirements.txt
pip install -e .
```

### Download pretrained model checkpoints

This script downloads two model checkpoints:
1. Pretrained HUMOS auto-encoder -- used to initialize the HUMOS cycle-consistent training runs
2. Final HUMOS model -- used for demo and inference
```shell
sh fetch_data.sh
```


### Download the SMPL model

Go to the [SMPL website](https://smpl.is.tue.mpg.de/download.php), register and go to the Download tab.


- Click on "Download version 1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs)" to download and place the files in the folder ``body_models/smpl/``.

### Download the SMPL+H model

Go to the [MANO website](https://mano.is.tue.mpg.de/download.php), register and go to the Download tab.


- Click on "Models & Code" to download ``mano_v1_2.zip`` and place it in the folder ``body_models/smplh/``.

- Click on "Extended SMPL+H model" to download ``smplh.tar.xz`` and place it in the folder ``body_models/smplh/``.


The next step is to extract the archives, merge the hands from ``mano_v1_2`` into the ``Extended SMPL+H models``, and remove any chumpy dependency.

All of this can be done using with the following commands.

```shell
bash humos/prepare/smplh.sh
```


This will create ``SMPLH_FEMALE.npz``, ``SMPLH_MALE.npz``, ``SMPLH_NEUTRAL.npz`` inside the ``body_models/smplh`` folder.

The resulting structure for the ```body_models``` directory should look like this:

```
.
├── prepare
│   ├── merge_smplh_mano.py
│   └── smplh.sh
├── smpl
│   ├── female
│   │   ├── model.npz
│   │   ├── model.pkl
│   │   └── smpl_female.bvh
│   ├── male
│   │   ├── model.npz
│   │   ├── model.pkl
│   │   └── smpl_male.bvh
│   └── neutral
│       ├── model.npz
│       └── model.pkl
└── smplh
    ├── female
    │   └── model.npz
    ├── male
    │   └── model.npz
    ├── neutral
    │   └── model.npz
    ├── SMPLH_FEMALE.npz
    ├── SMPLH_FEMALE.pkl
    ├── SMPLH_MALE.npz
    ├── SMPLH_MALE.pkl
    ├── SMPLH_NEUTRAL.npz
 ```

### Preparing the AMASS dataset

1. Please run all blocks of ```humos/prepare/raw_pose_processing_humos.ipynb``` in a jupyter notebook.
2. Clean treadmill sequences from BMLrub (BML_NTroje) and MPI_HDM05 by running
```python humos/prepare/clean_amass_data.py```
3. Extract HUMOS features using
```python humos/prepare/compute_3dfeats.py --fps 20```
4. Process text annotations by removing paths that don't exist
```python humos/prepare/process_text_annotations.py```
5. Get dataset mean of all the input features
```python humos/prepare/motion_stats.py```
6. Get random body shapes - used for inference
```python humos/prepare/sample_body_shapes.py```

## Run inference on HUMOS

```shell
python humos/test.py --cfg humos/configs/cfg_template_test.yml
```

## Run HUMOS training
```shell
python humos/train.py --cfg humos/configs/cfg_template.yml
```


## Citing

If you find this code useful for your research, please consider citing the following papers:


```bibtex
@InProceedings{tripathi2024humos,
    author    = {Tripathi, Shashank and Taheri, Omid and Lassner, Christoph and Black, Michael J. and Holden, Daniel and Stoll, Carsten},
    title     = {{HUMOS}: Human Motion Model Conditioned on Body Shape},
    booktitle = {European Conference on Computer Vision},
    organization = {Springer},
    year      = {2025},
    pages     = {133--152},
}
```

Several parts of this code are heavily derived from the [IPMAN](https://ipman.is.tue.mpg.de/) and [TMR](https://mathis.petrovich.fr/tmr/). Please also consider citing this work:

```bibtex
@inproceedings{tripathi2023ipman,
    title     = {{3D} Human Pose Estimation via Intuitive Physics},
    author    = {Tripathi, Shashank and M{\"u}ller, Lea and Huang, Chun-Hao P. and Taheri Omid and Black, Michael J. and Tzionas, Dimitrios},
    booktitle = {Conference on Computer Vision and Pattern Recognition ({CVPR})},
    pages = {4713--4725},
    year = {2023},
    url = {https://ipman.is.tue.mpg.de}
}
```
```bibtex
@inproceedings{petrovich23tmr,
    title     = {{TMR}: Text-to-Motion Retrieval Using Contrastive {3D} Human Motion Synthesis},
    author    = {Petrovich, Mathis and Black, Michael J. and Varol, G{\"u}l},
    booktitle = {International Conference on Computer Vision ({ICCV})},
    year      = {2023}
}
```


### License


See [LICENSE](LICENSE).


### Acknowledgments


We sincerely thank Tsvetelina Alexiadis, Alpar Cseke, Tomasz Niewiadomski, and Taylor McConnell for facilitating the perceptual study, and Giorgio Becherini for his help with the Rokoko baseline. We are grateful to Iain Matthews, Brian Karis, Nikos Athanasiou, Markos Diomataris, and Mathis Petrovich for valuable discussions and advice. Their invaluable contributions enriched this research significantly.

### Contact


For technical questions, please create an issue. For other questions, please contact `shashank.tripathi123@gmail.com`.
