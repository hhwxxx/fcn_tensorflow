from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from slim.nets import vgg

slim = tf.contrib.slim

NUMBER_CLASSES = 21
IGNORE_LABEL = 255

WEIGHT_DECAY = 0.0001
BATCH_NORM_DECAY = 0.9
DROPOUT_KEEP_PROB = 0.5

exclude_list_fcn_32s = ['fcn_32s/vgg_16/fc8']
exclude_list_fcn_16s = [
    'fcn_16s/vgg_16/fc8',
    'fcn_16s/deconv1',
    'fcn_16s/pool4_conv',
]
exclude_list_fcn_8s = [
    'fcn_8s/vgg_16/fc8',
    'fcn_8s/deconv1',
    'fcn_8s/pool4_conv',
    'fcn_8s/deconv2',
    'fcn_8s/pool3_conv',
]

EXCLUDE_LIST_MAP = {
    'fcn_32s' : exclude_list_fcn_32s,
    'fcn_16s' : exclude_list_fcn_16s,
    'fcn_8s'  : exclude_list_fcn_8s,
}


def fcn_32s(images, is_training):
    """fcn_32s.
    """
    with slim.arg_scope(vgg.vgg_arg_scope()):
        with tf.variable_scope('fcn_32s'):
            fc8_logits, end_points = vgg.vgg_16(
                inputs=images,
                num_classes=NUMBER_CLASSES,
                is_training=is_training,
                spatial_squeeze=False,
                fc_conv_padding='SAME',
                global_pool=False
            )
            logits = tf.image.resize_bilinear(
                fc8_logits, tf.shape(images)[1:3], align_corners=True)
    
    return logits


def fcn_16s(images, is_training):
    """fcn_16s.
    """
    with tf.variable_scope('fcn_16s'):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            fc8_logits, end_points = vgg.vgg_16(
                inputs=images,
                num_classes=NUMBER_CLASSES,
                is_training=is_training,
                spatial_squeeze=False,
                fc_conv_padding='SAME',
                global_pool=False
            )

            upsampled_fc8_logits = slim.conv2d_transpose(
                fc8_logits, NUMBER_CLASSES, kernel_size=[4, 4], stride=[2, 2],
                padding='SAME', activation_fn=None, normalizer_fn=None,
                scope='deconv1')
            pool4 = end_points['fcn_16s/vgg_16/pool4']
            pool4_logits = slim.conv2d(
                pool4, NUMBER_CLASSES, [1, 1], activation_fn=None,
                normalizer_fn=None, scope='pool4_conv')
            fused_logits = tf.add(pool4_logits, upsampled_fc8_logits,
                                  name='fused_logits')
            logits = tf.image.resize_bilinear(
                fused_logits, tf.shape(images)[1:3], align_corners=True)

    return logits


def fcn_8s(images, is_training):
    """fcn_8s.
    """
    with tf.variable_scope('fcn_8s'):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            fc8_logits, end_points = vgg.vgg_16(
                inputs=images,
                num_classes=NUMBER_CLASSES,
                is_training=is_training,
                spatial_squeeze=False,
                fc_conv_padding='SAME',
                global_pool=False
            )

            upsampled_fc8_logits = slim.conv2d_transpose(
                fc8_logits, NUMBER_CLASSES, kernel_size=[4, 4], stride=[2, 2],
                padding='SAME', activation_fn=None, normalizer_fn=None,
                scope='deconv1')
            pool4 = end_points['fcn_8s/vgg_16/pool4'] 
            pool4_logits = slim.conv2d(
                pool4, NUMBER_CLASSES, [1, 1],
                activation_fn=None, normalizer_fn=None, scope='pool4_conv')
            fused_logits_1 = tf.add(pool4_logits,
                                  upsampled_fc8_logits,
                                  name='fused_logits_1')

            pool3 = end_points['fcn_8s/vgg_16/pool3']
            pool3_logits = slim.conv2d(
                pool3, NUMBER_CLASSES, [1, 1],
                activation_fn=None, normalizer_fn=None, scope='pool3_conv')
            upsampled_fused_logits = slim.conv2d_transpose(
                fused_logits_1, NUMBER_CLASSES, kernel_size=[4, 4],
                stride=[2, 2], padding='SAME', activation_fn=None,
                normalizer_fn=None, scope='deconv2')
            fused_logits_2 = tf.add(upsampled_fused_logits,
                                    pool3_logits,
                                    name='fused_logits_2')

            logits = tf.image.resize_bilinear(
                fused_logits_2, tf.shape(images)[1:3], align_corners=True)
    
    return logits
