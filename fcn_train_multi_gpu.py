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

flags.DEFINE_integer('num_gpus', 2, 'How many GPUs to use.')

flags.DEFINE_string('tfrecord_folder',
                    './tfrecords',
                    'Directory containing tfrecords.')
flags.DEFINE_string('dataset_split', 'train',
                    'Using which dataset split to train the network.')

flags.DEFINE_string('model_variant', 'fcn_8s', 'Model variant.')
flags.DEFINE_boolean('use_init_model', True,
                     'Whether to initialize variables from pretrained model.')
flags.DEFINE_string('restore_ckpt_path',
                    './init_models/vgg_16.ckpt',
                    'Path to checkpoint.')

flags.DEFINE_string('train_dir',
                    './exp/fcn_8s/train',
                    'Training directory.')

flags.DEFINE_boolean('is_training', True, 'Is training?')
flags.DEFINE_integer('batch_size', 8, 'Batch size used for train.')
flags.DEFINE_integer('num_epochs', 500, 'Number of epochs to train.')

flags.DEFINE_float('initial_learning_rate', 0.00001,
                   'Initial learning rate.')
flags.DEFINE_integer('num_epochs_per_decay', 200,
                     'Decay steps in exponential learning rate decay policy.')
flags.DEFINE_float('learning_rate_decay_factor', 0.9,
                   'Decay rate in exponential learning rate decay policy.')

flags.DEFINE_integer('save_checkpoint_steps', 500, 'Save checkpoint steps.')
flags.DEFINE_integer('log_frequency', 10, 'Log frequency.')


# TODO(hhw): Move to core.py
def tower_loss(scope, images, labels):
    logits = core.inference(FLAGS.model_variant, images,
                            is_training=FLAGS.is_training)

    labels = tf.one_hot(labels, depth=core.NUMBER_CLASSES, 
                        on_value=1, off_value=0)
    cross_entropy_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels, logits=logits, scope=scope)
    regularization_loss = tf.losses.get_regularization_loss()
    total_loss = cross_entropy_loss + regularization_loss
    total_loss = tf.identity(total_loss, 'total_loss')

    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('total_loss', total_loss)

    return total_loss


# TODO(hhw): Move to core.py
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.

    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, axis=0)
            grads.append(expanded_g)

        grad = tf.concat(values=grads, axis=0)
        grad = tf.reduce_mean(grad, axis=0)

        var = grad_and_vars[0][1]
        grad_and_var = (grad, var)

        average_grads.append(grad_and_var)

    return average_grads


def train(tfrecord_folder, dataset_split, is_training):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.train.get_or_create_global_step()

        with tf.device('/cpu:0'):
            input_data = input_pipeline.inputs(
                tfrecord_folder, dataset_split, is_training, is_vis=False,
                batch_size=FLAGS.batch_size, num_epochs=None)
            images = input_data['image']
            labels = input_data['label']

            tf.summary.image('images', images)
            tf.summary.image('labels', tf.cast(labels, tf.uint8))

        num_batches_per_epoch = (input_pipeline.NUMBER_TRAIN_DATA
                                 / FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(
            FLAGS.initial_learning_rate, global_step, decay_steps,
            FLAGS.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # TODO(hhw): Change to adam optimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:{}'.format(i)):
                    with tf.name_scope('{}_{}'.format('tower', i)) as scope:
                        loss = tower_loss(scope, images, labels)

                        # tf.get_variable_scope().reuse_variables()

                        grads = optimizer.compute_gradients(
                            loss, tf.trainable_variables())

                        tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(grads, global_step=global_step)


        if FLAGS.use_init_model:
            variables_to_restore = slim.get_variables_to_restore(
                exclude=(models.EXCLUDE_LIST_MAP[FLAGS.model_variant]
                         + ['global_step', 'adam']))
            restorer = tf.train.Saver(variables_to_restore)
            def init_fn(scaffold, sess):
                restorer.restore(sess, FLAGS.restore_ckpt_path)
        else:
            init_fn = None

        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    self._step = int(
                        ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]) - 1
                else:
                    self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(
                    [tf.get_default_graph().get_tensor_by_name(
                        'tower_{}/total_loss:0'.format(i)) 
                     for i in range(FLAGS.num_gpus)])

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = (FLAGS.log_frequency
                                        * FLAGS.batch_size / duration)
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('{}: step {}, tower_loss = {} ({:5.3f} '
                                  'examples/sec; {:02.3} sec/batch)')
                    print(format_str.format(datetime.now(), self._step, loss_value,
                                            examples_per_sec, sec_per_batch))

        num_train_steps = int(num_batches_per_epoch * FLAGS.num_epochs)
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                scaffold=tf.train.Scaffold(init_fn=init_fn),
                hooks=[tf.train.StopAtStepHook(last_step=num_train_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=config,
                save_summaries_steps=100,
                save_checkpoint_steps=FLAGS.save_checkpoint_steps) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(unused_argv):
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    train(FLAGS.tfrecord_folder, FLAGS.dataset_split, FLAGS.is_training)


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()