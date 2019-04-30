"""
Create TFRecord from Images

This file is used to convert two directories containing the depth map, ground truth sdf, scale and quaternion
respectively into two TFRecord files (training and validation).
This is used to prepare the data as well as to facilitate the training and validation processes.

A structure of a dataset should be as follows:
- A directory containing the depth maps.
- A directory containing the ground truth sdf.
- A directory containing the ground truth scale and quaternion.
- A .txt file listing the file names for the training objects.
- A .txt file listing the file names for the validation objects.

Usage: python create_tf.record.py --config config/dataset_config.json

"""

import os
import sys
import cv2
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils import dataset_util
from utils.generic_util import parse_args

args = parse_args()
# Enable only if validation data exists
VALIDATION_EXISTS = args.VALIDATION_EXISTS
# Path to the directory which will store the TFRecord train file
TRAIN_TF_RECORD_NAME = args.TRAIN_TF_RECORD_NAME
# Path to the directory which will store the TFRecord validation file
VAL_TF_RECORD_NAME = args.VAL_TF_RECORD_NAME
# Path to the file containing the list of training data
TRAIN_DATA_LIST_NAME = args.TRAIN_DATA_LIST_NAME
# Path to the file containing the list of validation data
VAL_DATA_LIST_NAME = args.VAL_DATA_LIST_NAME
# Path to the directory containing the depth maps
IMAGE_DIR = args.IMAGE_DIR
FOREGROUND_DIR = args.FOREGROUND_DIR
# Path to the directory containing the coordinates and the corresponding sdf values
SDF_DIR = args.SDF_DIR
# Output height and width of depth maps
OUTPUT_HEIGHT = args.OUTPUT_HEIGHT
OUTPUT_WIDTH = args.OUTPUT_WIDTH
# The number of different views for a single object
NUM_POINTS = args.NUM_POINTS


def load_image(filename, output_size):
    """
    Read a depth map(.png) from a filename, change it to gray-scale image and resize it to the required output size

    :param filename: a string representing the file name

    :param output_size: a tuple [height, width]

    :return: Depth map after the above operations performed
    """
    img = Image.open(filename)
    img = np.array(img)
    img = cv2.resize(img, (output_size[1], output_size[0]), interpolation=cv2.INTER_CUBIC)
    return img


def load_npy(filename):
    """
    Read a numpy array from .npy

    :param filename: a string representing the file name

    :return: corresponding numpy array
    """
    return np.load(filename)


def dict_to_tf_example(depth_map_path, normal_map_path, foreground_map_path, point_path, sdf_path):
    """
    Convert gray-scale depth map to tf.Example proto.
    ----
    Args:
    ----
      xxx_path: Path to a corresponding data.
    ----
    Returns:
    ----
      example: The converted tf.Example.
    """
    # Reading depth map, scale, quaternion, points and sdf
    depth_map = load_image(depth_map_path, (OUTPUT_HEIGHT, OUTPUT_WIDTH)).astype(np.uint8)
    normal_map = load_image(normal_map_path, (OUTPUT_HEIGHT, OUTPUT_WIDTH)).astype(np.uint8)
    foreground_map = load_image(foreground_map_path, (OUTPUT_HEIGHT, OUTPUT_WIDTH)).astype(np.uint8)

    # flatten here is necessary, otherwise the multi-dimensional ndarray cannot be stored in tf records
    points = np.squeeze(load_npy(point_path))[0:NUM_POINTS, :].flatten()
    sdf = load_npy(sdf_path)[0:NUM_POINTS, :].flatten()

    # Create the TFRecord example
    example = tf.train.Example(features=tf.train.Features(feature={
        'depth_map/height': dataset_util.int64_feature(depth_map.shape[0]),
        'depth_map/width': dataset_util.int64_feature(depth_map.shape[1]),
        'depth_map/encoded': dataset_util.bytes_feature(tf.compat.as_bytes(depth_map.tostring())),
        'normal_map/height': dataset_util.int64_feature(normal_map.shape[0]),
        'normal_map/width': dataset_util.int64_feature(normal_map.shape[1]),
        'normal_map/encoded': dataset_util.bytes_feature(tf.compat.as_bytes(normal_map.tostring())),
        'foreground_map/height': dataset_util.int64_feature(foreground_map.shape[0]),
        'foreground_map/width': dataset_util.int64_feature(foreground_map.shape[1]),
        'foreground_map/encoded': dataset_util.bytes_feature(tf.compat.as_bytes(foreground_map.tostring())),
        'points': dataset_util.float_list_feature(points),
        'sdf': dataset_util.float_list_feature(sdf)}))
    return example


def create_tf_record(output_filename,
                     image_dir, foreground_dir,
                     sdf_dir,
                     examples):
    """Creates a TFRecord file from examples.
    ----
    Args:
    ----
      output_filename: Path to where output file is saved.

      xxx_dir: Directory where image files are stored.

      examples: Examples to parse and save to tf record.

    """
    # Create a TFRecordWriter
    writer = tf.python_io.TFRecordWriter(output_filename)
    # Start reading the images one by one and iterate in this loop
    for idx, example in tqdm(enumerate(examples)):
        found = False
        # create an item for a single view of an object
        depth_map_path, normal_map_path, foreground_map_path, point_path, sdf_path = "", "", "", "", ""

        if not os.path.exists(depth_map_path):
            depth_map_path = os.path.join(image_dir, example + '_r_45_depth.jpg0001.jpg')
        if not os.path.exists(normal_map_path):
            normal_map_path = os.path.join(image_dir, example + '_r_45_normal.jpg0001.jpg')
        if not os.path.exists(foreground_map_path):
            foreground_map_path = os.path.join(foreground_dir, example + '_r_45_foreground.jpg0001.jpg')
        if not os.path.exists(point_path):
            point_path = os.path.join(sdf_dir, example + '_points.npy')
        if not os.path.exists(sdf_path):
            sdf_path = os.path.join(sdf_dir, example + '_sdf.npy')

        # Break when everything is correct!
        if os.path.exists(depth_map_path) and os.path.exists(normal_map_path) and os.path.exists(foreground_map_path) and \
                os.path.exists(point_path) and os.path.exists(sdf_path):
            found = True

        # Try to create the TFRecord example. If it can't be done, ignore the example.
        try:
            if found:
                tf_example = dict_to_tf_example(depth_map_path, normal_map_path, foreground_map_path, point_path, sdf_path)
                writer.write(tf_example.SerializeToString())
        except ValueError:
            found = False

        if not found:
            print('Could not find {} or it is invalid, ignoring example.\n'.format(example), file=sys.stderr)

    # A writer should be closed after writing
    writer.close()


def main():
    print("Processing the dataset...\n")

    if not os.path.isdir(IMAGE_DIR):
        raise ValueError("Depth map and normal map directory doesn't exist or there is an error in reading it.")

    if not os.path.isdir(SDF_DIR):
        raise ValueError("Points and SDF directory doesn't exist or there is an error in reading it.")

    # Read training and validation images list, usually .txt file.
    train_examples = dataset_util.read_examples_list(TRAIN_DATA_LIST_NAME)
    val_examples = dataset_util.read_examples_list(VAL_DATA_LIST_NAME)

    # Run the create tf record method for the training data.
    print("Processing the training data...")
    create_tf_record(TRAIN_TF_RECORD_NAME, IMAGE_DIR, FOREGROUND_DIR, SDF_DIR, train_examples)
    print("DONE!\n")

    if VALIDATION_EXISTS:
        # Run the create tf record method for the validation data.
        print("Processing the validation data...\n")
        create_tf_record(VAL_TF_RECORD_NAME, IMAGE_DIR, FOREGROUND_DIR, SDF_DIR, val_examples)
        print("DONE!\n")


if __name__ == '__main__':
    # Run the main program
    main()
