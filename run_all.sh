#!/bin/bash

mkdir -p ILSVRC2012_img_val
tar -xvf ILSVRC2012_img_val.tar -C ILSVRC2012_img_val

conda activate vitis-ai-tensorflow

python images_to_tfrec.py \
  --image_dir      ILSVRC2012_img_val \
  --label_file     val.txt \
  --img_shard      1100 \
  --tfrec_base     val_imagenet \
  --tfrec_dir      tf_rec \
  --encode

