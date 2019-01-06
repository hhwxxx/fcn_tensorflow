from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod

import tensorflow as tf 
from slim.nets import vgg

slim = tf.contrib.slim


class Model(object):
    """Slim model definition."""

    def __init__(self,
                 num_classes,
                 is_training,
                 spatial_squeeze=False,
                 global_pool=False,
                 weight_decay=0.0005):
        """Constructor.

        Args:
            num_classes: Number of classes to predict.
            is_training: Whether the network is in training mode.
            spatial_squeeze: Whether to perform spatial squeeze.
            global_pool: Whether to perform global pooling.
            weight_decay: Weight decay of parameters.
        """
        self._num_classes = num_classes
        self._is_training = is_training
        self._spatial_squeeze = spatial_squeeze
        self._global_pool = global_pool
        self._weight_decay = weight_decay

    @abstractmethod
    def extract_features(self, inputs):
        """Extracts features from inputs.

        This function is responsible for extracting feature maps from `input`.

        Args:
            inputs: a [batch, height, width, channels] float tensor representing
                a batch of images.

        Returns:
            feature: a float tensor.
        """
        pass

    @classmethod
    @abstractmethod
    def exclude_list(self):
        """Returns exclude list used in restoring from checkpoint."""
        pass


class FCN32s(Model):
    """FCN32s."""

    def __init__(self,
                 num_classes,
                 is_training,
                 spatial_squeeze=False,
                 global_pool=False,
                 weight_decay=0.0005):
        super(FCN32s, self).__init__(
            num_classes=num_classes,
            is_training=is_training,
            spatial_squeeze=spatial_squeeze,
            global_pool=global_pool,
            weight_decay=weight_decay)

    def extract_features(self, inputs):
        with slim.arg_scope(
                vgg.vgg_arg_scope(weight_decay=self._weight_decay)):
            with tf.variable_scope('fcn_32s'):
                fc8_logits, end_points = vgg.vgg_16(
                    inputs=inputs,
                    num_classes=self._num_classes,
                    is_training=self._is_training,
                    spatial_squeeze=self._spatial_squeeze,
                    fc_conv_padding='SAME',
                    global_pool=self._global_pool)
                logits = tf.image.resize_bilinear(
                    fc8_logits, tf.shape(inputs)[1:3], align_corners=True)

        return logits

    @classmethod
    def exclude_list(self):
        return ['fcn_32s/vgg_16/fc8']


class FCN16s(Model):
    """FCN16s."""

    def __init__(self,
                 num_classes,
                 is_training,
                 spatial_squeeze=False,
                 global_pool=False,
                 weight_decay=0.0005):
        super(FCN16s, self).__init__(
            num_classes=num_classes,
            is_training=is_training,
            spatial_squeeze=spatial_squeeze,
            global_pool=global_pool,
            weight_decay=weight_decay)

    def extract_features(self, inputs):
        with tf.variable_scope('fcn_16s'):
            with slim.arg_scope(
                    vgg.vgg_arg_scope(weight_decay=self._weight_decay)):
                fc8_logits, end_points = vgg.vgg_16(
                    inputs=inputs,
                    num_classes=self._num_classes,
                    is_training=self._is_training,
                    spatial_squeeze=self._spatial_squeeze,
                    fc_conv_padding='SAME',
                    global_pool=self._global_pool)

                upsampled_fc8_logits = slim.conv2d_transpose(
                    fc8_logits, self._num_classes, kernel_size=[4, 4],
                    stride=[2, 2], padding='SAME', activation_fn=None,
                    normalizer_fn=None, scope='deconv1')
                pool4 = end_points['fcn_16s/vgg_16/pool4']
                pool4_logits = slim.conv2d(
                    pool4, self._num_classes, [1, 1], activation_fn=None,
                    normalizer_fn=None, scope='pool4_conv')
                fused_logits = tf.add(pool4_logits, upsampled_fc8_logits,
                                      name='fused_logits')
                logits = tf.image.resize_bilinear(
                    fused_logits, tf.shape(inputs)[1:3], align_corners=True)

        return logits

    @classmethod
    def exclude_list(self):
        return ['fcn_16s/vgg_16/fc8', 'fcn_16s/deconv1', 'fcn_16s/pool4_conv']


class FCN8s(Model):
    """FCN8s."""

    def __init__(self,
                 num_classes,
                 is_training,
                 spatial_squeeze=False,
                 global_pool=False,
                 weight_decay=0.0005):
        super(FCN8s, self).__init__(
            num_classes=num_classes,
            is_training=is_training,
            spatial_squeeze=spatial_squeeze,
            global_pool=global_pool,
            weight_decay=weight_decay)

    def extract_features(self, inputs):
        with tf.variable_scope('fcn_8s'):
            with slim.arg_scope(
                    vgg.vgg_arg_scope(weight_decay=self._weight_decay)):
                fc8_logits, end_points = vgg.vgg_16(
                    inputs=inputs,
                    num_classes=self._num_classes,
                    is_training=self._is_training,
                    spatial_squeeze=self._spatial_squeeze,
                    fc_conv_padding='SAME',
                    global_pool=self._global_pool)

                upsampled_fc8_logits = slim.conv2d_transpose(
                    fc8_logits, self._num_classes, kernel_size=[4, 4],
                    stride=[2, 2], padding='SAME', activation_fn=None,
                    normalizer_fn=None, scope='deconv1')
                pool4 = end_points['fcn_8s/vgg_16/pool4'] 
                pool4_logits = slim.conv2d(
                    pool4, self._num_classes, [1, 1], activation_fn=None,
                    normalizer_fn=None, scope='pool4_conv')
                fused_logits_1 = tf.add(pool4_logits,
                                        upsampled_fc8_logits,
                                        name='fused_logits_1')

                pool3 = end_points['fcn_8s/vgg_16/pool3']
                pool3_logits = slim.conv2d(
                    pool3, self._num_classes, [1, 1], activation_fn=None,
                    normalizer_fn=None, scope='pool3_conv')
                upsampled_fused_logits = slim.conv2d_transpose(
                    fused_logits_1, self._num_classes, kernel_size=[4, 4],
                    stride=[2, 2], padding='SAME', activation_fn=None,
                    normalizer_fn=None, scope='deconv2')
                fused_logits_2 = tf.add(upsampled_fused_logits,
                                        pool3_logits,
                                        name='fused_logits_2')

                logits = tf.image.resize_bilinear(
                    fused_logits_2, tf.shape(inputs)[1:3], align_corners=True)

        return logits


    @classmethod
    def exclude_list(self):
        return ['fcn_8s/vgg_16/fc8', 'fcn_8s/deconv1', 'fcn_8s/pool4_conv',
                'fcn_8s/deconv2', 'fcn_8s/pool3_conv']