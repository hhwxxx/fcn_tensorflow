from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import time
from datetime import datetime

import tensorflow as tf

import core
import models
import input_pipeline

slim = tf.contrib.slim
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('tfrecord_folder',
                    './tfrecords',
                    'Directory containing tfrecords.')
flags.DEFINE_string('dataset_split', 'train',
                    'Using which dataset split to train the network.')

flags.DEFINE_string('model_variant', 'fcn_8s', 'Model variant.')
flags.DEFINE_string('restore_ckpt_path',
                    './init_models/vgg_16.ckpt',
                    'Path to checkpoint.')

flags.DEFINE_string('train_dir',
                    './exp/fcn_8s/train',
                    'Training directory.')

flags.DEFINE_boolean('is_training', True, 'Is training?')
flags.DEFINE_integer('batch_size', 8, 'Batch size used for train.')
flags.DEFINE_integer('max_steps', 10000, 'Max training step.')

flags.DEFINE_float('initial_learning_rate', 0.00001,
                   'Initial learning rate.')
flags.DEFINE_integer('decay_steps', 1000,
                     'Decay steps in exponential learning rate decay policy.')
flags.DEFINE_float('decay_rate', 0.9,
                   'Decay rate in exponential learning rate decay policy.')

flags.DEFINE_integer('save_checkpoint_steps', 500, 'Save checkpoint steps.')
flags.DEFINE_integer('log_frequency', 10, 'Log frequency.')


def train(tfrecord_folder, dataset_split, is_training):
    with tf.Graph().as_default() as g:
        global_step = tf.train.get_or_create_global_step()

        with tf.device('/cpu:0'):
            input_data = input_pipeline.inputs(
                tfrecord_folder, dataset_split, is_training, is_vis=False,
                batch_size=FLAGS.batch_size, num_epochs=None)
            images = input_data['image']
            labels = input_data['label']

        tf.summary.image('images', images)
        tf.summary.image('labels', tf.cast(labels, tf.uint8))

        logits = core.inference(FLAGS.model_variant, images, FLAGS.is_training)
        total_loss = core.loss(logits, labels)

        tf.summary.histogram('logits', logits)
        tf.summary.scalar('total_loss', total_loss)

        learning_rate = tf.train.exponential_decay(
            FLAGS.initial_learning_rate, global_step, FLAGS.decay_steps,
            FLAGS.decay_rate, staircase=False)

        tf.summary.scalar('learning_rate', learning_rate)
        for var in tf.model_variables():
            tf.summary.histogram(var.op.name, var)

        with tf.variable_scope('adam'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(total_loss, global_step)

        variables_to_restore = slim.get_variables_to_restore(
            exclude=(models.EXCLUDE_LIST_MAP[FLAGS.model_variant]
                     + ['adam', 'global_step']))
        def name_in_checkpoint(var):
            return var.op.name.replace(FLAGS.model_variant + '/', '')
        variables_to_restore = {
            name_in_checkpoint(var):var for var in variables_to_restore
            if 'vgg_16' in var.op.name}

        restorer = tf.train.Saver(variables_to_restore)
        def init_fn(scaffold, sess):
            restorer.restore(sess, FLAGS.restore_ckpt_path)


        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    self._step = int(
                        ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]) - 1
                else:
                    self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                # Asks for loss value.
                return tf.train.SessionRunArgs(total_loss)

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = (FLAGS.log_frequency
                                        * FLAGS.batch_size / duration)
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; '
                                  '%.3f sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                                         examples_per_sec, sec_per_batch))

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                scaffold=tf.train.Scaffold(init_fn=init_fn),
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(total_loss),
                       _LoggerHook()],
                config=config,
                save_checkpoint_steps=FLAGS.save_checkpoint_steps) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(unused_argv):
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    train(FLAGS.tfrecord_folder, FLAGS.dataset_split, FLAGS.is_training)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
