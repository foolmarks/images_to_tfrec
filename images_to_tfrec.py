
'''
Convert images to Tensorflow TFRecords format files (shards).
Will convert a folder of images to one or more TFRecord files.
'''

'''
Author: Mark Harvey, Dec 2020
'''


import os
import argparse
import shutil
from tqdm import tqdm

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf


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


def _calc_num_shards(img_list, img_shard):
  ''' calculate number of shards'''
  last_shard =  len(img_list) % img_shard
  if last_shard != 0:
    num_shards =  (len(img_list) // img_shard) + 1
  else:
    num_shards =  (len(img_list) // img_shard)
  return last_shard, num_shards


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
  

def write_tfrec(tfrec_filename, image_dir, img_list, label_list):
  ''' write TFRecord file '''

  with tf.io.TFRecordWriter(tfrec_filename) as writer:

    for i in range(len(img_list)):
      filePath = os.path.join(image_dir, img_list[i])

      # read the JPEG source file into a tf string
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



def make_tfrec(image_dir, img_shard, tfrec_base, label_file, tfrec_dir, num_images):

  # make destination directory
  os.makedirs(tfrec_dir, exist_ok=True)
  print('Directory',tfrec_dir,'created',flush=True)

  # make lists of images and their labels
  all_labels, all_images = _create_images_labels(label_file)
  print('Found',len(all_labels),'images and labels in',label_file)

  if (num_images != 0 and num_images < len(all_images)):
    all_images = all_images[:num_images]
    all_labels = all_labels[:num_images]
    print('Using',num_images,'images..')
  else:
    print('Using',len(all_labels),'images..')

  # calculate how many shards we will generate and number of images in last shard
  last_shard, num_shards = _calc_num_shards(all_images, img_shard)
  print (num_shards,'TFRecord files will be created.')
  if (last_shard>0):
    print ('Last TFRecord file will have',last_shard,'images.')

  # create TFRecord files (shards)
  start = 0

  for i in tqdm(range(num_shards)):

    tfrec_filename = tfrec_base+'_'+str(i)+'.tfrecord'
    write_path = os.path.join(tfrec_dir, tfrec_filename)

    if (i == num_shards-1):
      write_tfrec(write_path, image_dir, all_images[start:], all_labels[start:])
    else:
      end = start + img_shard
      write_tfrec(write_path, image_dir, all_images[start:end], all_labels[start:end])
      start = end

  return



# only used if script is run as 'main' from command line
def main():
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-dir', '--image_dir',   type=str, default='val_images',help='Path to folder that contains images. Default is val_images')
  ap.add_argument('-s',   '--img_shard',   type=int, default=5000,        help='Number of images per shard. Default is 100') 
  ap.add_argument('-tfb', '--tfrec_base',  type=str, default='data',      help='Base file name for TFRecord files. Default is data') 
  ap.add_argument('-l',   '--label_file',  type=str, default='val.txt',   help='Imagenet validation set ground truths file. Default is val.txt') 
  ap.add_argument('-tfdir', '--tfrec_dir', type=str, default='tfrecords', help='Path to folder for saving TFRecord files. Default is tfrecords')  
  ap.add_argument('-n',   '--num_images',  type=int, default=0,           help='Number of images to convert - 0 means convert all. Default is 100')  
  args = ap.parse_args()  
  
  print (_divider)
  print ('Command line options:')
  print (' --image_dir  : ', args.image_dir)
  print (' --img_shard  : ', args.img_shard)
  print (' --tfrec_base : ', args.tfrec_base)
  print (' --label_file : ', args.label_file)
  print (' --tfrec_dir  : ', args.tfrec_dir)
  print (' --num_images : ', args.num_images)
  print (_divider)


  make_tfrec(args.image_dir, args.img_shard, args.tfrec_base, args.label_file, args.tfrec_dir, args.num_images)


if __name__ == '__main__':
  main()

