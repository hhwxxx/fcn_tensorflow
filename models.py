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


def fcn_32s_backup(images, is_training):
    batch_norm_params = {
        'is_training': is_training,
        'decay': BATCH_NORM_DECAY,
        'epsilon': 1e-5,
        'scale': True,
    }

    with tf.variable_scope('fcn_32s'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY)):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params) as scope:
                net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool5')

                # fully -> conv
                net = slim.conv2d(net, 4096, [7, 7], scope='fc6')
                net = slim.dropout(net, DROPOUT_KEEP_PROB, is_training=is_training, scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(net, DROPOUT_KEEP_PROB, is_training=is_training, scope='dropout7')

                prediction = slim.conv2d(net, NUMBER_CLASSES, [1, 1], activation_fn=None, 
                                         normalizer_fn=None, scope='prediction')
                logits = tf.image.resize_bilinear(prediction, tf.shape(images)[1:3], align_corners=True)

    return logits


def fcn_16s_backup(images, is_training):
    batch_norm_params = {
        'is_training': is_training,
        'decay': BATCH_NORM_DECAY,
        'epsilon': 1e-5,
        'scale': True,
    }

    end_points = {}
    with tf.variable_scope('fcn_16s'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY)):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params) as scope:
                net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool4')
                end_points['pool4'] = net
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool5')

                # fully -> conv
                net = slim.conv2d(net, 4096, [7, 7], scope='fc6')
                net = slim.dropout(net, DROPOUT_KEEP_PROB, is_training=is_training, scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(net, DROPOUT_KEEP_PROB, is_training=is_training, scope='dropout7')

                prediction_fc7 = slim.conv2d(net, NUMBER_CLASSES, [1, 1], activation_fn=None, 
                                             normalizer_fn=None, scope='prediction_fc7')
                upsampled_prediction_fc8 = slim.conv2d_transpose(prediction_fc7, NUMBER_CLASSES, kernel_size=[4, 4], 
                                                                 stride=[2, 2], padding='SAME', activation_fn=None, 
                                                                 normalizer_fn=None, scope='transposed_conv')
                prediction_pool4 = slim.conv2d(end_points['pool4'], NUMBER_CLASSES, [1, 1], activation_fn=None, 
                                               normalizer_fn=None, scope='prediction_pool4')
                prediction_fusion = tf.add(prediction_pool4, upsampled_prediction_fc8, name='prediction_fusion')
                logits = tf.image.resize_bilinear(prediction_fusion, tf.shape(images)[1:3], align_corners=True)

    return logits


def fcn_8s_backup(images, is_training):
    batch_norm_params = {
        'is_training': is_training,
        'decay': BATCH_NORM_DECAY,
        'epsilon': 1e-5,
        'scale': True,
    }

    end_points = {}
    with tf.variable_scope('fcn_8s'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY)):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params) as scope:
                net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool3')
                end_points['pool3'] = net
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool4')
                end_points['pool4'] = net
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool5')

                # fully -> conv
                net = slim.conv2d(net, 4096, [7, 7], scope='fc6')
                net = slim.dropout(net, DROPOUT_KEEP_PROB, is_training=is_training, scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(net, DROPOUT_KEEP_PROB, is_training=is_training, scope='dropout7')

                prediction_fc7 = slim.conv2d(net, NUMBER_CLASSES, [1, 1], activation_fn=None, 
                                             normalizer_fn=None, scope='prediction_fc7')
                upsampled_prediction_fc7 = slim.conv2d_transpose(prediction_fc7, NUMBER_CLASSES, kernel_size=[4, 4], 
                                                                 stride=[2, 2], padding='SAME', activation_fn=None, 
                                                                 normalizer_fn=None, scope='transposed_conv_1')
                prediction_pool4 = slim.conv2d(end_points['pool4'], NUMBER_CLASSES, [1, 1], activation_fn=None, 
                                               normalizer_fn=None, scope='prediction_pool4')
                prediction_fusion_1 = tf.add(prediction_pool4, upsampled_prediction_fc7, name='prediction_fusion_1')

                prediction_pool3 = slim.conv2d(end_points['pool3'], NUMBER_CLASSES, [1, 1], activation_fn=None, 
                                               normalizer_fn=None, scope='prediction_pool3')
                upsampled_prediction_fusion_1 = slim.conv2d_transpose(prediction_fusion_1, NUMBER_CLASSES, 
                                                                      kernel_size=[4, 4], stride=[2, 2], padding='SAME', 
                                                                      activation_fn=None, normalizer_fn=None, 
                                                                      scope='transposed_conv_2')
                prediction_fusion_2 = tf.add(upsampled_prediction_fusion_1, prediction_pool3, name='prediction_fusion_2')

                logits = tf.image.resize_bilinear(prediction_fusion_2, tf.shape(images)[1:3], align_corners=True)

    return logits
