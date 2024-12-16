#!/bin/bash

# Script that fetches checkpoints and other necessary data for inference

# Download pretrained model from humos autoencoder-mode training. This initializes the cycle-consistency training run
wget https://download.is.tue.mpg.de/humos/humos/q6zbv2tu/checkpoints/latest-epoch=1599.ckpt --trust-server-names
mkdir -p logs/humos/q6zbv2tu/checkpoints
mv latest-epoch=1599.ckpt logs/humos/q6zbv2tu/checkpoints

# Download final HUMOS model for demo and inference runs
wget https://download.is.tue.mpg.de/humos/humos/5bhgscl8/checkpoints/latest-epoch=199.ckpt --trust-server-names
mkdir -p logs/humos/5bhgscl8/checkpoints/
mv latest-epoch=199.ckpt logs/humos/5bhgscl8/checkpoints/