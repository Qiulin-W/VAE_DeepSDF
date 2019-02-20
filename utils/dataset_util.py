"""
Utility functions for creating data sets.
"""

import tensorflow as tf
import numpy as np


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def parse_record(raw_record, image_size, num_sample_points):
    """
    Parse from a TFRecord
    ----
    Args:
    ----
    :param raw_record: a TFRecord

    :param image_size: [Height, Width, Channels]

    :param num_sample_points: points sampled from the object mesh
    ----
    Return:
    ----
    A tuple consisting of an feature and its ground truth tensor
    """

    # Keys to features stored in the TFRecord
    keys_to_features = {
        'depth_map/height':
            tf.FixedLenFeature((), tf.int64),
        'depth_map/width':
            tf.FixedLenFeature((), tf.int64),
        'depth_map/gray-scale':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'scale':
            tf.VarLenFeature(tf.float32),
        'quaternion':
            tf.VarLenFeature(tf.float32),
        'points':
            tf.VarLenFeature(tf.float32),
        'sdf':
            tf.VarLenFeature(tf.float32)
    }

    # The following lines are used to extract an image and a label from a TFRecord
    parsed = tf.parse_single_example(raw_record, keys_to_features)

    depth_map = tf.to_float(tf.decode_raw(parsed['depth_map/gray-scale'], tf.uint8))
    depth_map = tf.reshape(depth_map, image_size)

    scale = tf.sparse_tensor_to_dense(parsed['scale'], default_value=0)
    scale = tf.reshape(scale, [1])

    quaternion = tf.sparse_tensor_to_dense(parsed['quaternion'], default_value=0)
    quaternion = tf.reshape(quaternion, [4])

    points = tf.sparse_tensor_to_dense(parsed['points'], default_value=0)
    points = tf.reshape(points, [num_sample_points, 3])

    sdf = tf.sparse_tensor_to_dense(parsed['sdf'], default_value=0)
    sdf = tf.reshape(sdf, [num_sample_points, 1])

    return depth_map, points, scale, quaternion, sdf


def input_fn(data_file, image_size, batch_size=16, num_epochs_to_repeat=1, buffer_size=128,
             num_sample_points=3000):
    """
    input_fn in the tf.data input pipeline
    ----
    Args:
    ----
    :param data_file: The file containing the data either a "train" TFRecord file or a "validation" TFRecord file

    :param image_size: [Height, Width, Channels]

    :param batch_size: The number of samples per batch.

    :param num_epochs_to_repeat: The number of epochs to repeat the dataset. Set it to 1, and OutOfRangeError exception
    will be thrown at the end of each epoch which is used by TFEstimator for example.

    :param buffer_size: an integer to indicate the size of the buffer. If it equals to the whole dataset size, all of
    the records will be loaded in memory

    :param is_training: A boolean to indicate whether training is being done or not

    ----
    Return:
    ----
    A tuple consisting of an image tensor and its label tensor
    """
    # Create a dataset from the datafile
    dataset = tf.data.Dataset.from_tensor_slices([data_file])
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    # Parse the record into an image and its label
    dataset = dataset.map(lambda record: parse_record(record, image_size, num_sample_points))

    # Load "buffer_size" records from the disk.
    dataset = dataset.prefetch(buffer_size)

    # Repeat the dataset if train function works for multiple epochs or throw OutOfRangeError exception
    dataset = dataset.repeat(num_epochs_to_repeat)

    # Batch the dataset into portions. The size of each one is equal to batch_size
    dataset = dataset.batch(batch_size)

    # Create an iterator from this dataset to yield (features, labels) tuple
    iterator = dataset.make_one_shot_iterator()
    next_item = iterator.get_next()

    features = {'depth_map': next_item[0],
                'points': next_item[1]}

    labels = {'scale': next_item[2],
              'quaternion': next_item[3],
              'sdf': next_item[4]}

    return features, labels


def get_num_records(tf_record_filename):
    """
    Get the number of records stored in a TFRecord file
    ----
    Args:
    ----
    :param tf_record_filename: path to the tfrecord file
    ----
    Return:
    ----
    Number of records (int)
    """
    return np.sum([1 for _ in tf.python_io.tf_record_iterator(tf_record_filename)])


def read_examples_list(path):
    """Read list of training or validation examples.

    The file is assumed to contain a single example per line where the first
    token in the line is an identifier that allows us to find the image and
    annotation xml for that example.

    For example, the line:
    xyz 3
    would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).

    Args:
      path: absolute path to examples list file.

    Returns:
      list of example identifiers (strings).
    """
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]
