from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf 

import models

NUMBER_CLASSES = 21
IGNORE_LABEL = 255

MODEL_MAP = {
    'fcn_32s': models.fcn_32s,
    'fcn_16s': models.fcn_16s,
    'fcn_8s' : models.fcn_8s,
}


def inference(model, images, is_training):
    model = MODEL_MAP[model]
    logits = model(images, is_training)

    return logits


def loss(logits, labels):
    labels = tf.squeeze(labels, axis=[3])
    labels = tf.reshape(labels, shape=[-1])
    not_ignore_mask = tf.to_float(tf.not_equal(labels, IGNORE_LABEL))
    onehot_labels = tf.one_hot(
        indices=tf.cast(labels, tf.int32), depth=NUMBER_CLASSES,
        on_value=1, off_value=0)

    cross_entropy_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels,
        logits=tf.reshape(logits, shape=[-1, NUMBER_CLASSES]),
        weights=not_ignore_mask)
    tf.summary.scalar('corss_entropy_loss', cross_entropy_loss)
    tf.losses.add_loss(cross_entropy_loss)
    
    total_loss = tf.losses.get_total_loss()
    
    return total_loss
