import numpy as np
import os
import glob
import argparse
import shutil

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf

from dataset_utils import parser

_divider = '-------------------------------------'

def input_fn(tfrec_dir, batchsize, height, width):
    '''
    Dataset creation and augmentation
    '''
    tfrecord_files = tf.data.Dataset.list_files('{}/data_65.tfrecord'.format(tfrec_dir),shuffle=False)
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batchsize, drop_remainder=True)
    return dataset


def test_tfrec(tfrec_dir, image_dir):

  height=224
  width=224
  
  # make dataset
  test_dataset = input_fn(tfrec_dir, 1, height, width)

  # make destination directory for images
  if os.path.exists(image_dir):
    shutil.rmtree(image_dir, ignore_errors=True)
    print('Deleted',image_dir)

  os.mkdir(image_dir)
  print('Directory',image_dir,'created')

  i = 0
  mylist = []
  for tfr in test_dataset:

    img = tf.reshape(tfr[0], [tfr[0].shape[1],tfr[0].shape[2],tfr[0].shape[3]] )

    img_png = tf.io.encode_png(img)
    filepath =  os.path.join(image_dir, str(i)+'_image.png' )
    tf.io.write_file(filepath, img_png)
    mylist.append(tfr[1][0].numpy())
    i += 1
    
    if i==10:
      break
  
  print(mylist)

  return


def main():
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-dir',  '--image_dir',type=str, default='img_test',   help='Path to folder for saving images. Default is img_test')
  ap.add_argument('-tfdir','--tfrec_dir', type=str, default='tfrec_train', help='Path to folder containg TFRecord files. Default is tfrecords')  
  args = ap.parse_args()  
  
  print (_divider)
  print ('Command line options:')
  print (' --tfdir     : ', args.tfrec_dir)
  print (' --image_dir : ', args.image_dir)
  print (_divider)


  test_tfrec(args.tfrec_dir, args.image_dir)


if __name__ == '__main__':
  main()
  
  
