# Convert JPEG images to TensorFlow TFRecords format

One of the most common formats for use with the TensorFlow tf.data API is TFRecords. This repository contains a script file (images_to_tfrec.py) that converts a folder of JPEG encoded images and a text file of ground truth labels into a number of TFRecord files.

It is specifically set up for the ImageNet2012 validation dataset but could easily be applied to any folder of images.

The script runs faster than many other similar examples and will produce tfrecord files which occupy he same amount fo disk spce as the original images.

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

5. Running the images_to_tfrec.py script like this will create 10 tfrecord files in a folder called 'tfrecords'. The files will be named data_0.tfrecord to data_9.tfrecord. Each tfrecord file will contain 5000 images and labels:

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

The script expects that the label file has a image file name and an integer label on each line:

```
ILSVRC2012_val_00000001.JPEG 65
ILSVRC2012_val_00000002.JPEG 970
ILSVRC2012_val_00000003.JPEG 230
ILSVRC2012_val_00000004.JPEG 809
ILSVRC2012_val_00000005.JPEG 516
ILSVRC2012_val_00000006.JPEG 57
ILSVRC2012_val_00000007.JPEG 334
ILSVRC2012_val_00000008.JPEG 415
ILSVRC2012_val_00000009.JPEG 674
ILSVRC2012_val_00000010.JPEG 332
.
```

The labels file will be read and each line split into a file name and a label:

```python
def _create_images_labels(label_file):
  ''' create lists of image filenames and their labels '''  
  f= open(label_file,'r')
  linesList = f.readlines()
  f.close()
  labels_list=[]
  fileNames_list=[]
  for line in linesList:
    fileName, label = line.split()
    labels_list.append(int(label.strip()))
    fileNames_list.append(fileName.strip())
  return labels_list, fileNames_list
```

The list returned by this function maintain the order as per the labels file so that the images and labels are packed into the tfrecord files in the same order as they appear in the labels file.

The most import function is the one that creates each tfrecord file:

```python
def write_tfrec(tfrec_filename, image_dir, img_list, label_list):
  ''' write TFRecord file '''

  with tf.io.TFRecordWriter(tfrec_filename) as writer:

    for i in range(len(img_list)):
      filePath = os.path.join(image_dir, img_list[i])

      # read the JPEG source file into a tf.string
      image = tf.io.read_file(filePath)

      # get the shape of the image from the JPEG file header
      image_shape = tf.io.extract_jpeg_shape(image, output_type=tf.dtypes.int32, name=None)

      # features dictionary
      feature_dict = {
        'label' : _int64_feature(int(label_list[i])),
        'height': _int64_feature(image_shape[0]),
        'width' : _int64_feature(image_shape[1]),
        'chans' : _int64_feature(image_shape[2]),
        'image' : _bytes_feature(image)
      }

      # Create Features object
      features = tf.train.Features(feature = feature_dict)

      # create Example object
      tf_example = tf.train.Example(features=features)

      # serialize Example object into TfRecord file
      writer.write(tf_example.SerializeToString())

  return
  ```
  
  Note how each JPEG file is first read into a variable of type tf.string:
  
  ```python
  image = tf.io.read_file(filePath)
  ```
  
  
  The shape of the original image is read from the JPEG file header:
  
   ```python
  image_shape = tf.io.extract_jpeg_shape(image, output_type=tf.dtypes.int32, name=None)
  ```
  
  We need the shape as the image will be in serial string format when it is unpacked from the TFRecord and will need to be reshaped before being input to a model.  
  Reading just the header is much faster than decoding the entire JPEG image to get the shape and this is the key to the speed of this script.
  
  



