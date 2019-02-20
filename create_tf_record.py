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
DEPTH_MAP_DIR = args.DEPTH_MAP_DIR
# Path to the directory containing the coordinates and the corresponding sdf values
SDF_DIR = args.SDF_DIR
# Path to the directory containing the scales and the quaternions
POSE_DIR = args.POSE_DIR
# Output height and width of depth maps
OUTPUT_DM_HEIGHT = args.OUTPUT_DM_HEIGHT
OUTPUT_DM_WIDTH = args.OUTPUT_DM_WIDTH
# The number of different views for a single object
NUM_VIEWS = args.NUM_VIEWS


def load_image(filename, output_size):
    """
    Read a depth map(.png) from a filename, change it to gray-scale image and resize it to the required output size

    :param filename: a string representing the file name

    :param output_size: a tuple [height, width]

    :return: Depth map after the above operations performed
    """
    img = Image.open(filename)
    img = np.array(img)
    img_resize = cv2.resize(img, (output_size[1], output_size[0]), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)

    return img_gray


def load_npy(filename):
    """
    Read a numpy array from .npy

    :param filename: a string representing the file name

    :return: corresponding numpy array
    """
    return np.load(filename)


def dict_to_tf_example(image_path, scale_path, quaternion_path, point_path, sdf_path, angle_idx):
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
    img_gray = load_image(image_path, (OUTPUT_DM_HEIGHT, OUTPUT_DM_WIDTH)).astype(np.uint8)
    scale = load_npy(scale_path)[angle_idx]
    quaternion = load_npy(quaternion_path)[angle_idx]
    # flatten here is necessary, otherwise the multi-dimensional ndarray cannot be stored in tf records
    points = load_npy(point_path).flatten()
    sdf = load_npy(sdf_path).flatten()

    # Create the TFRecord example
    example = tf.train.Example(features=tf.train.Features(feature={
        'depth_map/height': dataset_util.int64_feature(img_gray.shape[0]),
        'depth_map/width': dataset_util.int64_feature(img_gray.shape[1]),
        'depth_map/gray-scale': dataset_util.bytes_feature(tf.compat.as_bytes(img_gray.tostring())),
        'scale': dataset_util.float_feature(scale),
        'quaternion': dataset_util.float_list_feature(quaternion),
        'points': dataset_util.float_list_feature(points),
        'sdf': dataset_util.float_list_feature(sdf)}))

    return example


def create_tf_record(output_filename,
                     image_dir,
                     sdf_dir,
                     pose_dir,
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
        for i in range(NUM_VIEWS):
            image_path, scale_path, quaternion_path, point_path, sdf_path = "", "", "", "", ""
            angle = str(int(360/NUM_VIEWS) * i)

            if not os.path.exists(image_path):
                image_path = os.path.join(image_dir, example + '_r_' + angle + '0001.png')
            if not os.path.exists(scale_path):
                scale_path = os.path.join(pose_dir, example + '_scale.npy')
            if not os.path.exists(quaternion_path):
                quaternion_path = os.path.join(pose_dir, example + '_quaternion.npy')
            if not os.path.exists(point_path):
                point_path = os.path.join(sdf_dir, example + '_points.npy')
            if not os.path.exists(sdf_path):
                sdf_path = os.path.join(sdf_dir, example + '_sdf.npy')

            # Break when everything is correct!
            if os.path.exists(image_path) and os.path.exists(scale_path) and os.path.exists(quaternion_path) \
                    and os.path.exists(point_path) and os.path.exists(sdf_path):
                found = True

            # Try to create the TFRecord example. If it can't be done, ignore the example.
            try:
                if found:
                    tf_example = dict_to_tf_example(image_path, scale_path, quaternion_path,
                                                    point_path, sdf_path, i)
                    writer.write(tf_example.SerializeToString())
            except ValueError:
                found = False

            if not found:
                print('Could not find {} or it is invalid, ignoring example.\n'.format(example), file=sys.stderr)

    # A writer should be closed after writing
    writer.close()


def main():
    print("Processing the dataset...\n")

    if not os.path.isdir(DEPTH_MAP_DIR):
        raise ValueError("Depth map directory doesn't exist or there is an error in reading it.")

    if not os.path.isdir(SDF_DIR):
        raise ValueError("Points and SDF directory doesn't exist or there is an error in reading it.")

    if not os.path.isdir(POSE_DIR):
        raise ValueError("Scale and quaternion directory doesn't exist or there is an error in reading it.")

    # Read training and validation images list, usually .txt file.
    train_examples = dataset_util.read_examples_list(TRAIN_DATA_LIST_NAME)
    val_examples = dataset_util.read_examples_list(VAL_DATA_LIST_NAME)

    # Run the create tf record method for the training data.
    print("Processing the training data...")
    create_tf_record(TRAIN_TF_RECORD_NAME, DEPTH_MAP_DIR, SDF_DIR, POSE_DIR, train_examples)
    print("DONE!\n")

    if VALIDATION_EXISTS:
        # Run the create tf record method for the validation data.
        print("Processing the validation data...\n")
        create_tf_record(VAL_TF_RECORD_NAME, DEPTH_MAP_DIR, SDF_DIR, POSE_DIR, val_examples)
        print("DONE!\n")


if __name__ == '__main__':
    # Run the main program
    main()
