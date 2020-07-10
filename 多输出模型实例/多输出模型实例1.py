"""
多输出模型实例
"""
##
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from glob import glob
##
# 加载数据
data_dir = r'D:\dataset\multi-output-classification\dataset'
colors = ['black', 'blue', 'red']
coats = ['jeans', 'shoes', 'dress', 'shirt']
all_dirs = os.listdir(data_dir)

image_filenames = []
labels = []
for each_dir in all_dirs:
    color, coat = each_dir.split('_')
    color_id, coat_id = colors.index(color), coats.index(coat)
    image_docunts = len(os.listdir(os.path.join(data_dir, each_dir)))
    labels.extend([[color_id, coat_id]] * image_docunts)
    image_filenames.extend(glob(os.path.join(data_dir, each_dir, '*')))
##
height = 256
width = 256
channels = 3
batch_size = 16
train_size = 0.8
train_count = int(len(image_filenames) * train_size)

def preprocessing(x, y):
    image = tf.io.read_file(x)
    image = tf.image.decode_jpeg(image, channels=3)
    image = 2 * (tf.cast(image, tf.float32) / 255.0) - 1
    image = tf.image.resize(image, size=(height, width))
    return image, y

labels_color, labels_coat = list(zip(*labels))
dataset = tf.data.Dataset.from_tensor_slices((image_filenames, (list(labels_color), list(labels_coat))))
dataset = dataset.map(preprocessing).shuffle(len(image_filenames))
train_dataset = dataset.take(train_count)
test_dataset = dataset.skip(train_count)
##
print(len(list(train_dataset.as_numpy_iterator())), len(list(test_dataset.as_numpy_iterator())))
##
# 创建模型
# inception_base = tf.keras.applications.inception_v3.InceptionV3(
#     include_top=False, weights='image_net', input_shape=(256, 256, 3)
# )


