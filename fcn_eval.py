from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math

import tensorflow as tf

import input_pipeline
import core


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('tfrecord_folder',
                    './tfrecords',
                    'Directory containing tfrecords.')
flags.DEFINE_string('dataset_split', 'val', 'Dataset split used to evaluate.')

flags.DEFINE_string('model_variant', 'fcn_32s', 'Model variant.')
flags.DEFINE_string('checkpoint_dir',
                    './exp/fcn_32s/train',
                    'Directory containing trained checkpoints.')

flags.DEFINE_string('eval_dir',
                    './exp/fcn_32s/eval',
                    'Evaluation directory.')

flags.DEFINE_boolean('is_training', False, 'Is training?')
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('eval_interval_secs', 60 * 3,
                     'Evaluation interval seconds.')


def eval(tfrecord_folder, dataset_split, is_training):
    with tf.Graph().as_default() as g:
        with tf.device('/cpu:0'):
            input_dict = input_pipeline.inputs(
                tfrecord_folder, dataset_split, is_training, is_vis=False, 
                batch_size=FLAGS.batch_size, num_epochs=1)
            images = input_dict['image']
            labels = input_dict['label']

        labels = tf.squeeze(labels, axis=[-1])
        logits = core.inference(FLAGS.model_variant, images, FLAGS.is_training)
        predictions = tf.argmax(
            logits, axis=-1, name='prediction', output_type=tf.int64)

        weights = tf.to_float(tf.not_equal(labels, core.IGNORE_LABEL))
        labels = tf.where(tf.equal(labels, core.IGNORE_LABEL),
                          tf.zeros_like(labels),
                          labels)
        mean_iou, update_op = tf.metrics.mean_iou(
            labels=labels, predictions=predictions,
            num_classes=core.NUMBER_CLASSES, weights=weights, name='mean_iou')

        summary_op = tf.summary.scalar('mean_iou', mean_iou)
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        num_batches = int(
            math.ceil(input_pipeline.NUMBER_VAL_DATA / float(FLAGS.batch_size)))

        # get global_step used in summary_writer.
        ckpt = tf.train.get_checkpoint_state(
            checkpoint_dir=FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = int(
                ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print('Get global_step from checkpoint name.')
        else:
            global_step = tf.train.get_or_create_global_step()
            print('Create gloabl_step')

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.train.MonitoredSession(
                session_creator=tf.train.ChiefSessionCreator(
                    config=config,
                    checkpoint_dir=FLAGS.checkpoint_dir
                )) as mon_sess:
            for _ in range(num_batches):
                mon_sess.run(update_op)

            summary = mon_sess.run(summary_op)
            summary_writer.add_summary(summary, global_step=global_step)
            summary_writer.flush()
            print('*' * 60)
            print('mean_iou:', mon_sess.run(mean_iou))
            print('*' * 60)
            summary_writer.close()


def main(unused_argv):
    while True:
        eval(FLAGS.tfrecord_folder, FLAGS.dataset_split, FLAGS.is_training)
        time.sleep(FLAGS.eval_interval_secs)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
