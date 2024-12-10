#!/bin/bash
echo "Extraction of the archives"
echo

mkdir -p body_models/smplh
cd body_models/smplh
mkdir tmp
cd tmp

tar xfv ../smplh.tar.xz
unzip ../mano_v1_2.zip

cd ../../../
echo
echo "Done!"
echo
echo "Clean and merge models"
echo

python body_models/prepare/merge_smplh_mano.py --smplh-fn body_models/smplh/tmp/male/model.npz --mano-left-fn body_models/smplh/tmp/mano_v1_2/models/MANO_LEFT.pkl --mano-right-fn body_models/smplh/tmp/mano_v1_2/models/MANO_RIGHT.pkl --output-folder body_models/smplh/
python body_models/prepare/merge_smplh_mano.py --smplh-fn body_models/smplh/tmp/mano_v1_2/models/SMPLH_male.pkl --mano-left-fn body_models/smplh/tmp/mano_v1_2/models/MANO_LEFT.pkl --mano-right-fn body_models/smplh/tmp/mano_v1_2/models/MANO_RIGHT.pkl --output-folder body_models/smplh/

python body_models/prepare/merge_smplh_mano.py --smplh-fn body_models/smplh/tmp/female/model.npz --mano-left-fn body_models/smplh/tmp/mano_v1_2/models/MANO_LEFT.pkl --mano-right-fn body_models/smplh/tmp/mano_v1_2/models/MANO_RIGHT.pkl --output-folder body_models/smplh/
python body_models/prepare/merge_smplh_mano.py --smplh-fn body_models/smplh/tmp/mano_v1_2/models/SMPLH_female.pkl --mano-left-fn body_models/smplh/tmp/mano_v1_2/models/MANO_LEFT.pkl --mano-right-fn body_models/smplh/tmp/mano_v1_2/models/MANO_RIGHT.pkl --output-folder body_models/smplh/

python body_models/prepare/merge_smplh_mano.py --smplh-fn body_models/smplh/tmp/neutral/model.npz --mano-left-fn body_models/smplh/tmp/mano_v1_2/models/MANO_LEFT.pkl --mano-right-fn body_models/smplh/tmp/mano_v1_2/models/MANO_RIGHT.pkl --output-folder body_models/smplh/

echo
echo "Done!"
echo
echo "Deleting tmp files"
rm -rf body_models/smplh/tmp/
echo
echo "Done!"

echo "Extracting SMPLH npz"
cd body_models/smplh
tar xfv ./smplh.tar.xz
cd ../../
