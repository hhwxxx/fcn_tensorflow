from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

import core
import input_pipeline


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_dir',
                    './exp/fcn_32s/train',
                    'Directory containing trained checkpoints.')
flags.DEFINE_string('model_variant',
                    'fcn_32s',
                    'Model variant.')
flags.DEFINE_string('image_file',
                    './VOC2012/JPEGImages/2007_000346.jpg',
                    'Image file to predict and visualize.')
flags.DEFINE_boolean('is_training', False, 'Is training?')


def bit_get(val, idx):
    """Gets the bit value.

    Args:
        val: Input value, int or numpy int array.
        idx: Which bit of the input val.

    Returns:
        The "idx"-th bit of input val.
    """
    return (val >> idx) & 1


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


def predict():
    with tf.Graph().as_default() as g:
        image = tf.gfile.FastGFile(FLAGS.image_file, 'rb').read()
        image = tf.image.decode_jpeg(image, channels=3)
        original_image = image
        image = tf.image.resize_images(
            image, input_pipeline.INPUT_SIZE,
            method=tf.image.ResizeMethod.BILINEAR, align_corners=True)

        logits = core.inference(
            FLAGS.model_variant, tf.expand_dims(image, axis=0),
            FLAGS.is_training)
        prediction = tf.argmax(logits, axis=-1)
        prediction = tf.image.resize_nearest_neighbor(
            tf.expand_dims(prediction, axis=-1), 
            tf.shape(original_image)[:2], align_corners=True)
        
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.train.MonitoredSession(
                session_creator=tf.train.ChiefSessionCreator(
                    config=config,
                    checkpoint_dir=FLAGS.checkpoint_dir)) as mon_sess:
            colormap = create_pascal_label_colormap()

            original_image, prediction = mon_sess.run(
                [original_image, prediction])
            prediction = np.squeeze(prediction)
            prediction = colormap[prediction]

            plt.figure(1)
            plt.subplot(121)
            plt.imshow(original_image)

            plt.subplot(122)
            plt.imshow(prediction)

            plt.show()
            

def main(unused_argv):
    predict()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
