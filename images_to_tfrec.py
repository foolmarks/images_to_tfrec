'''
Convert images to Tensorflow TFRecords format files (shards).
Will convert a folder of images to one or more TFRecord files.

Author: Mark Harvey
Date  : 2nd June 2020
'''

'''
TO DO:
- Add optional preprocessing functions.
- Add test/train splitting
'''

'''
Arguments:
  --image_dir  : Path to folder where images are stored.
               : Default is image_dir
  --img_shard  : Number of images in each shard. If the number of images in --image_dir is not an 
               : exact multiple of shard_size, the last shard file will contain less than img_shard images.
               : If img_shard is > than number of images in img_dir, all images will be converted.
               : Default is 100.
  --label_file : Full path of a text file which provides labels for each image.
               : If a file of label is provided it must be a text file with one line per image file.
               : Each line must be of the format:  image_file_name    integer label..for example: bus001.png 12
               : If no label file is provided, then the images will be converted to TFRecord format without labels.
               : If an image is not included in the label file, then it will not be converted to TFRecords and its 
               : full path and name will be logged.
               : Default is empty string (i.e. no label file)
  --encode     : Will encode the images to a one of the following formats before writing to the shard file: JPEG, PNG, BITMAP
               : Default is empty string (i.e. no encoding)

Returns:
  One or more TFRecord files.
'''

'''
REPORTING:
shard_size:   report actual size of last shard
label file: if image is not in file, exclude from tfrec
          : add to a list and report to a log/console.
img_shard  : Report #images in last shard.
           :  img_shard is > than number of images in img_dir, all images will be converted.
'''

import os
import argparse




def _list_images(image_dir):
    '''
    Create sorted list of images
    '''
    return (os.listdir(image_dir)).sort()







def gen_tfrec(image_dir, label_file, img_shard, encode):




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
                  default='',
                  help='Full path of label file. Default is empty string, i.e no label file')
  ap.add_argument('-s', '--img_shard',
                  type=int,
                  default=100,
                  help='Number of images per shard. Default is 100') 
  ap.add_argument('-e', '--encode',
                  type=str,
                  default='jpg',
                  choices=['png','jpg','jpeg','bmp','bitmap','PNG','JPG','JPEG','BMP','BITMAP'],
                  help='Encode image files - valid choices are png,jpg,jpeg,bmp,bitmap,PNG,JPG,JPEG,BMP,BITMAP. Default is jpg')  

  
  args = ap.parse_args()  
  
  print ('Command line options:')
  print (' --image_dir  : ', args.image_dir)
  print (' --label_file : ', args.label_file)
  print (' --img_shard  : ', args.img_shard)
  print (' --encode     : ', args.encode)


  gen_tfrec(args.image_dir, args.label_file, args.img_shard, args.encode)


if __name__ == '__main__':
  main()


