# Convert images to TensorFlow TFRecords format


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

5. Run the make_tfrec.py script which will

```python
python -u make_tfrec.py \
   --image_dir    val_images \
   --label_file   val.txt \
   --img_shard    5000 \
   --tfrec_base   data \
   --tfrec_dir    tfrecords \
   --num_images   0
```

--this will create 10 tfrecord files in a folder called 'tfrecords'. The files will be named data_0.tfrecord to data_9.tfrecord.


Arguments for make_tfrec.py:

| Argument              | Description                                                    |
|---------------------- | -------------------------------------------------------------- |
|`--image_dir` or `-dir`| name and path of folder that contains images to be converted   |
|`--label_file` or `-l` | Name of input function used in calibration pre-processing      |
|`--output_dir`         | Name of the output folder where the quantized models are saved |
|`--input_nodes`        | Name(s) of the input nodes                                     |
|`--output_nodes`       | Name(s) of the output nodes                                    |
|`--input_shapes`       | Shape(s) of the input nodes                                    |
|`--calib_iter`         | Number of calibration iterations                               |


