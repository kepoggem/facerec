#!/usr/bin/env sh

align_data_path=/home/kepoggem/dcnn/datasets/lfw/lfw-deepfunneled
model_prefix=/home/kepoggem/dcnn/datasets/vggface2/checkpoints/vgg19_checkpoints/vgg19
epoch=38
# evaluate on lfw
python lfw.py --lfw-align $align_data_path --model-prefix $model_prefix --epoch $epoch