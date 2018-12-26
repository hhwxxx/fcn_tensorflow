from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import glob
import collections
import six

import tensorflow as tf
import numpy as np
from PIL import Image

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('image_folder',
                    './VOC2012/JPEGImages', 
                    'Folder containing images.')
flags.DEFINE_string('label_folder',
                    './VOC2012/SegmentationClassRaw', 
                    'Folder containing semantic segmentation annotations.')
flags.DEFINE_string('list_folder',
                    './VOC2012/ImageSets/Segmentation', 
                    'Folder containing lists for training and validation.')
flags.DEFINE_string('output_dir',
                    './tfrecords', 'Path to save tfrecord.')
flags.DEFINE_string('image_format', 'jpg', 'Image format')
flags.DEFINE_string('label_format', 'png', 'Label format.')


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self, image_format='jpeg', channels=3):
        """Class constructor.
        Args:
            image_format: Image format. Only 'jpeg', 'jpg', or 'png' are supported.
            channels: Image channels.
        """
        with tf.Graph().as_default():
            self._decode_data = tf.placeholder(dtype=tf.string)
            self._image_format = image_format
            self._session = tf.Session()
            if self._image_format in ('jpeg', 'jpg'):
                self._decode = tf.image.decode_jpeg(self._decode_data,
                                                    channels=channels)
            elif self._image_format == 'png':
                self._decode = tf.image.decode_png(self._decode_data,
                                                   channels=channels)

    def read_image_dims(self, image_data):
        """Reads the image dimensions.
        Args:
            image_data: string of image data.
        Returns:
            image_height and image_width.
        """
        image = self.decode_image(image_data)

        return image.shape[:2]

    def decode_image(self, image_data):
        """Decodes the image data string.
        Args:
            image_data: string of image data.
        Returns:
            Decoded image data.
        Raises:
            ValueError: Value of image channels not supported.
        """
        image = self._session.run(self._decode,
                                  feed_dict={self._decode_data: image_data})
        if len(image.shape) != 3 or image.shape[2] not in (1, 3):
            raise ValueError('The image channels not supported.')

        return image


def _int64_list_feature(values):
    """Returns a TF-Feature of int64_list.

    Args:
        values: A scalar or list of values.

    Returns:
      A TF-Feature.
    """
    if not isinstance(values, collections.Iterable):
        values = [values]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_list_feature(values):
    """Returns a TF-Feature of float_list.
    
    Args:
        values: A float or list of floats.
    
    Returns:
        A TF-Feature.
    """
    if not isinstance(values, collections.Iterable):
        values = [values]

    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _bytes_list_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
        values: A string.

    Returns:
        A TF-Feature.
    """
    def norm2bytes(value):
        return value.encode() if isinstance(value, str) and six.PY3 else value

    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


def convert_to_tfrecord(dataset_split):
    """Convert dataset split into tfrecord file.
    """
    dataset = os.path.basename(dataset_split)[:-4]
    print('Processing {} data.'.format(dataset))
    filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
    num_images = len(filenames)

    image_reader = ImageReader(image_format='jpg', channels=3)
    label_reader = ImageReader(image_format='png', channels=1)

    output_filename = os.path.join(FLAGS.output_dir, dataset + '.tfrecord')
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for i in range(num_images):
            image_filename = os.path.join(
                FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
            image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
            image_height, image_width = image_reader.read_image_dims(image_data)
            label_filename = os.path.join(
                FLAGS.label_folder, filenames[i] + '.' + FLAGS.label_format)
            label_data = tf.gfile.FastGFile(label_filename, 'rb').read()
            label_height, label_width = label_reader.read_image_dims(label_data)

            if image_height != label_height or image_width != label_width:
                raise RuntimeError('Shape mismatched between image and label.')
            
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image/encoded': _bytes_list_feature(image_data),
                        'image/filename': _bytes_list_feature(filenames[i]),
                        'image/format': _bytes_list_feature(FLAGS.image_format),
                        'image/height': _int64_list_feature(image_height),
                        'image/width': _int64_list_feature(image_width),
                        'image/channel': _int64_list_feature(3),
                        'label/encoded': _bytes_list_feature(label_data),
                        'label/format': _bytes_list_feature(FLAGS.label_format),
                    }
                )
            )
            tfrecord_writer.write(example.SerializeToString())


def main(unused_argv):
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    dataset_splits = glob.glob(os.path.join(FLAGS.list_folder, '*.txt'))
    for dataset_split in dataset_splits:
        convert_to_tfrecord(dataset_split)
    
    print('Finished.')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
