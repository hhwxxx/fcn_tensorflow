from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import functools

import tensorflow as tf 


NUMBER_TRAIN_DATA = 1464
NUMBER_VAL_DATA = 1449
NUMBER_TRAINVAL_DATA = 2913
INPUT_SIZE = [384, 384]


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/filename': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/format': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/height': tf.FixedLenFeature([], tf.int64, default_value=0),
            'image/width': tf.FixedLenFeature([], tf.int64, default_value=0),
            'image/channel': tf.FixedLenFeature([], tf.int64, default_value=3),
            'label/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
            'label/format': tf.FixedLenFeature([], tf.string, default_value=''),
        }
    )

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    label = tf.image.decode_png(features['label/encoded'], channels=1)
    filename = features['image/filename']
    height = features['image/height']
    width = features['image/width']

    input_dict = {
        'image': image,
        'label': label,
        'filename': filename,
        'height': height,
        'width': width,
    }

    return input_dict


def shift_image(image, label, width_shift_range, height_shift_range):
    """This fn will perform the horizontal or vertical shift"""
    if width_shift_range or height_shift_range:
        if width_shift_range:
            width_shift_range = tf.random_uniform(
                [],
                -width_shift_range * INPUT_SIZE[1],
                width_shift_range * INPUT_SIZE[1])
        if height_shift_range:
            height_shift_range = tf.random_uniform(
                [],
                -height_shift_range * INPUT_SIZE[0],
                height_shift_range * INPUT_SIZE[0])
        # Translate both 
        image = tf.contrib.image.translate(
            image, [width_shift_range, height_shift_range])
        label = tf.contrib.image.translate(
            label, [width_shift_range, height_shift_range])

    return image, label


def flip_image(horizontal_flip, image, label):
    if horizontal_flip:
        flip_prob = tf.random_uniform([], 0.0, 1.0)
        image, label = tf.cond(
            tf.less(flip_prob, 0.5),
            lambda: (tf.image.flip_left_right(image),
                     tf.image.flip_left_right(label)),
            lambda: (image, label))

    return image, label


def normalize(image):
    """Normalize image to [-1, 1]"""
    image = (2.0 / 255.0) * tf.to_float(image) - 1.0

    return image


def augment(input_dict,
            is_vis,
            resize=None,  
            hue_delta=0, 
            horizontal_flip=False,
            width_shift_range=0,
            height_shift_range=0):
    """Data augmentation and preprocessing.

    Args:
        input_dict: input_dict.
        is_vis: boolean indicating whether in visualization mode.
        resize: Resize the image to some size e.g. [512, 512]
        hue_delta: Adjust the hue of an RGB image by random factor
        horizontal_flip: Random left right flip,
        width_shift_range: Randomly translate the image horizontally
        height_shift_range: Randomly translate the image vertically 

    Returns:
        input_dict.
    """
    image = input_dict['image']
    label = input_dict['label']

    if is_vis:
        input_dict['original_image'] = image

    if resize is not None:
        image = tf.image.resize_images(
            image, resize, align_corners=True, 
            method=tf.image.ResizeMethod.BILINEAR)
        label = tf.image.resize_images(
            label, resize, align_corners=True, 
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    if hue_delta:
        image = tf.image.random_hue(image, hue_delta)
    
    image, label = flip_image(horizontal_flip, image, label)
    image, label = shift_image(
        image, label, width_shift_range, height_shift_range)

    image = normalize(image)

    input_dict['image'] = image
    input_dict['label'] = label
    
    return input_dict


train_config = {
    'is_vis': False,
    'resize': [INPUT_SIZE[0], INPUT_SIZE[1]],
    'hue_delta': 0.1,
    'horizontal_flip': True,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1
}
train_preprocessing_fn = functools.partial(augment, **train_config)

val_config = {
    'is_vis': False,
    'resize': [INPUT_SIZE[0], INPUT_SIZE[1]],
}
val_preprocessing_fn = functools.partial(augment, **val_config)

vis_config = { 
    'is_vis': True,
    'resize': [INPUT_SIZE[0], INPUT_SIZE[1]],
}
vis_preprocessing_fn = functools.partial(augment, **vis_config)


def inputs(tfrecord_folder, dataset_split, is_training, is_vis,
           batch_size, num_epochs=None):
    filename = os.path.join(tfrecord_folder, dataset_split + '.tfrecord')

    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)

        dataset = dataset.map(decode)

        if is_training:
            dataset = dataset.map(train_preprocessing_fn)
            min_queue_examples = int(NUMBER_TRAIN_DATA * 0.2)
            dataset = dataset.shuffle(
                buffer_size=min_queue_examples + 3 * batch_size)
        elif is_vis:
            dataset = dataset.map(vis_preprocessing_fn)
        else:
            dataset = dataset.map(val_preprocessing_fn)

        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=batch_size)

        iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()
