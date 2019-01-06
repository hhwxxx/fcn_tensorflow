from __future__ import absolute_import
from __future__ import division
from __future__ import division

import os
import shutil

import tensorflow as tf
import numpy as np
from PIL import Image

import input_pipeline
import core


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('tfrecord_folder',
                    './tfrecords',
                    'Directory containing tfrecords.')
flags.DEFINE_string('dataset_split', 'val',
                    'Dataset split used to visualize.')

flags.DEFINE_string('model_variant', 'fcn_32s', 'Model variant.')
flags.DEFINE_string('checkpoint_dir',
                    './exp/fcn_32s/train',
                    'Directory containing trained checkpoints.')

flags.DEFINE_string('vis_dir',
                    './exp/fcn_32s/vis',
                    'Directory containing visualization results.')

flags.DEFINE_boolean('is_training', False, 'Is training?')
flags.DEFINE_integer('batch_size', 1, 'Batch size used for visualization.')


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


def vis(tfrecord_folder, dataset_split, is_training):
    with tf.Graph().as_default() as g:
        with tf.device('/cpu:0'):
            input_dict = input_pipeline.inputs(
                tfrecord_folder, dataset_split, is_training,
                is_vis=True, batch_size=FLAGS.batch_size, num_epochs=1)
            original_images = input_dict['original_image']
            images = input_dict['image']
            filename = input_dict['filename']

        logits = core.inference(FLAGS.model_variant, images, FLAGS.is_training)
        predictions = tf.argmax(logits, axis=-1)
        predictions = tf.expand_dims(predictions, axis=-1)
        predictions = tf.image.resize_nearest_neighbor(
            predictions, tf.shape(original_images)[1:3], align_corners=True)

        if not dataset_split in ['train', 'val', 'trainval', 'test']:
            raise ValueError('Invalid argument.')
        elif dataset_split == 'train':
            num_iters = input_pipeline.NUMBER_TRAIN_DATA
        elif dataset_split == 'val':
            num_iters = input_pipeline.NUMBER_VAL_DATA
        elif dataset_split == 'trainval':
            num_iters = input_pipeline.NUMBER_TRAINVAL_DATA

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = int(
                ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print('Get global_step from checkpoint name')
        else:
            global_step = tf.train.get_or_create_global_step()
            print('Create global_step.')

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.train.MonitoredSession(
                session_creator=tf.train.ChiefSessionCreator(
                    config=config,
                    checkpoint_dir=FLAGS.checkpoint_dir)) as mon_sess:
            colormap = create_pascal_label_colormap()
            cur_iter = 0
            while cur_iter < num_iters:
                (original_image, prediction, image_name) = mon_sess.run(
                    [original_images, predictions, filename])
                original_image = np.squeeze(original_image)
                prediction = np.squeeze(prediction)
                image_name = image_name[0]
                print('Visualing {}'.format(image_name))

                pil_image = Image.fromarray(original_image)
                pil_image.save(
                    '{}/{}.png'.format(FLAGS.vis_dir, image_name),
                    format='PNG')

                prediction = colormap[prediction]
                pil_prediction = Image.fromarray(prediction.astype(dtype=np.uint8))
                pil_prediction.save(
                    '{}/{}_prediction.png'.format(FLAGS.vis_dir, image_name),
                    format='PNG')

                cur_iter += 1

            print('Finished!')


def main(unused_argv):
    if os.path.exists(FLAGS.vis_dir):
        shutil.rmtree(FLAGS.vis_dir)
    if not os.path.exists(FLAGS.vis_dir):
        os.makedirs(FLAGS.vis_dir)

    vis(FLAGS.tfrecord_folder, FLAGS.dataset_split, FLAGS.is_training)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
