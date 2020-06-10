'''
Convert images to Tensorflow TFRecords format files (shards).
Will convert a folder of images to one or more TFRecord files.

Author: Mark Harvey
Date  : 2nd June 2020
'''


'''
Arguments:
  --image_dir  : Path to folder where images are stored. This folder should only contain image files.
               : Default is image_dir
  --img_shard  : Number of images in each shard. If the number of images in --image_dir is not an 
               : exact multiple of shard_size, the last shard file will contain less than img_shard images.
               : If img_shard is > than number of images in img_dir, all images will be converted.
               : Default is 100.
  --label_file : Full path of a text file which provides labels for each image.
               : If a file of label is provided it must be a text file with one line per image file.
               : Each line must be of the format:  image_file_name    integer label..for example: bus001.png 12
               : Default is val.txt
  --tfrec_dir  : Path to folder where TFRecord files are written.
               : Default is tfrec_dir
  --encode     : If enabled, will encode the images to JPEG format before writing to TFRecords file.
               : Default is disabled (i.e. no encoding)
  --tfrec_base : Base file name for TFRecords files. Each TFRecord file will be names with this base and an index.
               : Default is dataset.

Returns:
  One or more TFRecord files.
'''


import os
import argparse
import cv2
import shutil

import tensorflow as tf
from progressbar import ProgressBar


_divider = '-------------------------------------'

def _bytes_feature(value):
  '''Returns a bytes_list from a string / byte'''
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  '''Returns a float_list from a float / double'''
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  ''' Returns an int64_list from a bool / enum / int / uint '''
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _list_images(image_dir):
  ''' Create sorted list of images '''
  list_images = os.listdir(image_dir)
  list_images.sort()
  return list_images


def _calc_num_shards(img_list, img_shard):
  ''' calculate number of shards'''
  last_shard =  len(img_list) % img_shard
  if last_shard != 0:
    num_shards =  (len(img_list) // img_shard) + 1
  else:
    num_shards =  (len(img_list) // img_shard)
  return last_shard, num_shards


def _check_labels_file(label_file, img_list):
  ''' check there is one label for every image '''  
  f= open(label_file,'r')
  linesList = f.readlines()
  linesList.sort()
  f.close()
  labels_list=[]
  for i in range(len(linesList)):
    fileName, label = linesList[i].split()
    if (fileName != img_list[i]):
      raise SystemExit('Error in labels file, mismatch at line #',str(i))
    else:
      labels_list.append(int(label))

  return labels_list
  

def write_tfrec(tfrec_filename, image_dir, img_list, label_list, encode):
  ''' write TFRecord file '''

  with tf.io.TFRecordWriter(tfrec_filename) as writer:

    for i in range(len(img_list)):
      filePath = os.path.join(image_dir, img_list[i])
      image = cv2.imread(filePath)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # if encode = True, use JPEG encoding
      if (encode):
        _,im_buf_arr = cv2.imencode('.jpg', image)
        image_tf = _bytes_feature(im_buf_arr.tostring())
        is_jpeg=1
      else:
        image_tf = _bytes_feature(image.tostring())
        is_jpeg=0

      # features dictionary
      feature_dict = {
        'height'  : _int64_feature(image.shape[0]),
        'width'   : _int64_feature(image.shape[1]),
        'chans'   : _int64_feature(image.shape[2]),
        'label'   : _int64_feature(int(label_list[i])),
        'is_jpeg' : _int64_feature(is_jpeg),
        'image'   : image_tf
      }

      # Create Features object
      features = tf.train.Features(feature = feature_dict)

      # create Example object
      tf_example = tf.train.Example(features=features)

      # serialize Example object into TfRecord file
      writer.write(tf_example.SerializeToString())

  return





def gen_tfrec(image_dir, label_file, tfrec_dir, img_shard, encode, tfrec_base):

  # make destination directory if necessary
  if os.path.exists(tfrec_dir):
    shutil.rmtree(tfrec_dir, ignore_errors=True)
    print('Deleted contents of',tfrec_dir)

  os.mkdir(tfrec_dir)
  print('Directory',tfrec_dir,'created')

  # make a list of all image files in source folder
  all_images = _list_images(image_dir)
  print('Found',len(all_images),'image files in',image_dir)

  # check labels file if specified
  all_labels = _check_labels_file(label_file, all_images)
  print('Found',len(all_labels),'labels in',label_file)

  # calculate how many shards we will generate and number of images in last shard
  last_shard, num_shards = _calc_num_shards(all_images, img_shard)
  print (num_shards,'TFRecord files will be created.')
  if last_shard != 0:
    print ('Last TFRecord file will have',last_shard,'images.')

  # create TFRecord files (shards)
  start = 0
  progress = ProgressBar()

  for i in progress(range(num_shards)):

    tfrec_filename = tfrec_base+'_'+str(i)+'.tfrecord'
    write_path = os.path.join(tfrec_dir, tfrec_filename)

    if (i == num_shards-1):
      write_tfrec(write_path, image_dir, all_images[start:], all_labels[start:], encode)
    else:
      end = start + img_shard
      write_tfrec(write_path, image_dir, all_images[start:end], all_labels[start:end], encode)
      start = end

  return



# only used if script is run as 'main' from command line
def main():
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()

  ap.add_argument('-dir', '--image_dir',
                  type=str,
                  default='image_dir',
                  help='Path to folder that contains images. Default is image_dir')
  ap.add_argument('-l', '--label_file',
                  type=str,
                  default='val.txt',
                  help='Full path of label file. Default is val.txt')
  ap.add_argument('-s', '--img_shard',
                  type=int,
                  default=100,
                  help='Number of images per shard. Default is 100') 
  ap.add_argument('-tfb', '--tfrec_base',
                  type=str,
                  default='dataset',
                  help='Base file name for TFRecord files. Default is dataset') 
  ap.add_argument('-tfdir', '--tfrec_dir',
                  type=str,
                  default='tfrec_dir',
                  help='Path to folder for saving TFRecord files. Default is tfrec_dir') 
  ap.add_argument('-e', '--encode',
                  action='store_true',
                  help='Encode image files to JPEG format before writing to TFRecord file. Default is disabled i.e. no encoding')  


  
  args = ap.parse_args()  
  
  print (_divider)
  print ('Command line options:')
  print (' --image_dir  : ', args.image_dir)
  print (' --label_file : ', args.label_file)
  print (' --img_shard  : ', args.img_shard)
  print (' --tfrec_dir  : ', args.tfrec_dir)
  print (' --encode     : ', args.encode)
  print (' --tfrec_base : ', args.tfrec_base)
  print (_divider)


  gen_tfrec(args.image_dir, args.label_file, args.tfrec_dir, args.img_shard, args.encode, args.tfrec_base)


if __name__ == '__main__':
  main()


