# Convert images to TensorFlow TFRecords format

One of the most common formats for use with the TensorFlow tf.data API is TFRecords. This repository contains a script file (images_to_tfrec.py) that converts a folder of images and a text file of ground truth labels into a number of TFRecord files.

It is specifically set up for the ImageNet2012 validation dataset but could easily be applied to any folder of images.

+ Tested with TensorFlow 2.3

## How to use the script

1. Clone this repository.

2. Download the Imagenet validation set from [Academic Torrents](https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5) or some other source and place the ILSVRC2012_img_val.tar file in the cloned repository.

3. These command will untar the downloaded validation set to a folder called val_images:

```shell
mkdir val_images
tar -xvf ILSVRC2012_img_val.tar -C val_images --skip-old-files
```

..this may take a while as there are 50,000 images to untar - make sure you have sufficient disk space.


4. Download and untar the validation labels file:

```shell
wget -c --no-clobber http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
tar -xvf caffe_ilsvrc12.tar.gz --skip-old-files val.txt
```

5. Run the make_tfrec.py script which will create 10 tfrecord files in a folder called 'tfrecords'. The files will be named data_0.tfrecord to data_9.tfrecord.

```python
python -u images_to_tfrec.py \
   --image_dir    val_images \
   --label_file   val.txt \
   --img_shard    5000 \
   --tfrec_base   data \
   --tfrec_dir    tfrecords \
   --num_images   0
```

Arguments for make_tfrec.py:

| Argument                |  Type  | Default |Description|
|-------------------------|:------:|:-------:|:------------------------------------------------------ |
|`--image_dir` or `-dir`  | string |val_images|Name and path of folder that contains images to be converted |
|`--label_file` or `-l`   | string |val.txt|Name and path of text file containing ground truth labels |
|`--img_shard` or `-s`    | integer|5000|Number of images and labels in each tfrecord file |
|`--tfrec_base` or `-tfb` | string |data|Base name of tfrecord files |
|`--tfrec_dir` or `-tfdir`| string |tfrecords|Name and path of folder where tfrecord files are saved to |
|`--num_images` or `-n`   | integer|0|Total number of images to be converted. 0 means convert all |


The number of tfrecord files created will depend upon the nunber of images per tfrecord file (`--img_shard`) and the total number of images converted (`--num_images`). To convert all images, leave the `--num_images` at its default setting of 0.

If `--num_images` is not an exact multiple of `--img_shard` then the last tfrecord file will contain less images than the others and this will be reported by the script, for example:

```shell
Last TFRecord file will have 56 images.
```

## How the script works

xxx

