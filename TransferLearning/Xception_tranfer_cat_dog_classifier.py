"""
使用Xception进行迁移学习对猫狗数据集进行分类
"""
#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from datetime import datetime

# %%
# 数据准备
train_dir = r'D:\dataset\dc_2000\train'
test_dir = r'D:\dataset\dc_2000\test'

height = 256
width = 256
channel = 3
batch_size = 16


train_data_gen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.xception.preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True,
)
train_generator = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(height, width),
    class_mode='binary',
    batch_size=batch_size,
    shuffle=True,
    seed=1
)

test_data_gen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=keras.applications.xception.preprocess_input
)
test_generator = test_data_gen.flow_from_directory(
    test_dir,
    target_size=(height, width),
    class_mode='binary',
    batch_size=batch_size,
)

# %%
# 构建网络
xception_base = keras.applications.xception.Xception(
    include_top=False,
    weights='imagenet',
    input_shape=(height, width, channel),
    pooling='avg'
)
xception_base.trainable = False
# %%
model = keras.Sequential([
    xception_base,
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1, activation='sigmoid')
])

# %%
model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.Adam(lr=1e-4),
    metrics=['accuracy']
)

log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
callbacks = [
    keras.callbacks.ReduceLROnPlateau(patience=10),
    keras.callbacks.EarlyStopping(patience=15),
    keras.callbacks.ModelCheckpoint('Xception_transfer_cat_dog_classifier.tf', save_best_only=True),
    keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]
# %%
epochs = 100
train_count = train_generator.samples
test_count = test_generator.samples
model.fit(train_generator, epochs=epochs, steps_per_epoch=train_count // batch_size, 
validation_data=test_generator, validation_steps=test_count // batch_size,
callbacks=callbacks
)

# %%
