from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf 

import models

NUMBER_CLASSES = 21
IGNORE_LABEL = 255

MODEL_MAP = {
    'fcn_32s': models.FCN32s,
    'fcn_16s': models.FCN16s,
    'fcn_8s': models.FCN8s,
}


def inference(model, images, is_training):
    """Performs forward pass of the model."""
    model = MODEL_MAP[model](NUMBER_CLASSES, is_training)
    logits = model.extract_features(images)

    return logits


def loss(logits, labels):
    """Computes and returns loss."""
    labels = tf.squeeze(labels, axis=[3])
    labels = tf.reshape(labels, shape=[-1])
    not_ignore_mask = tf.to_float(tf.not_equal(labels, IGNORE_LABEL))
    onehot_labels = tf.one_hot(
        indices=tf.cast(labels, tf.int32), depth=NUMBER_CLASSES,
        on_value=1, off_value=0)

    # automatically add loss to collection tf.GraphKeys.LOSSES.
    cross_entropy_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels,
        logits=tf.reshape(logits, shape=[-1, NUMBER_CLASSES]),
        weights=not_ignore_mask)
    cross_entropy_loss = tf.identity(cross_entropy_loss, 'cross_entropy_loss')
    tf.summary.scalar('losses/corss_entropy_loss', cross_entropy_loss)

    regularization_loss = tf.losses.get_regularization_loss()
    regularization_loss = tf.identity(regularization_loss, 'regularization_loss')
    tf.summary.scalar('losses/regularization_loss', regularization_loss)

    total_loss = tf.losses.get_total_loss()
    total_loss = tf.identity(total_loss, 'total_loss')
    tf.summary.scalar('losses/total_loss', total_loss)
    
    return total_loss


def tower_loss(model, images, labels, is_training, scope):
    logits = inference(model, images, is_training)

    labels = tf.squeeze(labels, axis=[3])
    labels = tf.reshape(labels, shape=[-1])
    not_ignore_mask = tf.to_float(tf.not_equal(labels, IGNORE_LABEL))
    onehot_labels = tf.one_hot(
        indices=tf.cast(labels, tf.int32), depth=NUMBER_CLASSES,
        on_value=1, off_value=0)

    # automatically add loss to collection tf.GraphKeys.LOSSES.
    cross_entropy_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels,
        logits=tf.reshape(logits, shape=[-1, NUMBER_CLASSES]),
        weights=not_ignore_mask, scope=scope)
    cross_entropy_loss = tf.identity(cross_entropy_loss, 'cross_entropy_loss')

    regularization_loss = tf.losses.get_regularization_loss()
    regularization_loss = tf.identity(regularization_loss, 'regularization_loss')

    total_loss = tf.losses.get_total_loss(name='total_loss')

    tf.summary.scalar('losses/cross_entropy_loss', cross_entropy_loss)
    tf.summary.scalar('losses/regularization_loss', regularization_loss)
    tf.summary.scalar('losses/total_loss', total_loss)

    return total_loss


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

