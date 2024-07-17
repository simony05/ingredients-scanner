import tensorflow as tf
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras import optimizers
from keras import models, layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# data
train_dir = 'train'
validation_dir = 'validation'
test_dir = 'test'

# classes
classes = [class_name for class_name in os.listdir(train_dir)]
count = []
for class_name in classes :
    count.append(len(os.listdir(os.path.join(train_dir, class_name))))

# data to dataframe
def create_df(folder_path) :
    all_images = []    
    for class_name in classes :
        class_path = os.path.join(folder_path, class_name)
        all_images.extend([(os.path.join(class_path, file_name), class_name) for file_name in os.listdir(class_path)])
    df = pd.DataFrame(all_images, columns=['file_path', 'label'])
    return df

train_df = create_df(train_dir)
validation_df = create_df(validation_dir)
test_df = create_df(test_dir)

# image data augmentation and generation to avoid overfitting, more efficient by batching and shuffling

# train generator
train_datagen = ImageDataGenerator(
    rescale = 1./255,                 # images in range 0-1
    rotation_range = 20,              # rotate images by 20 deg
    width_shift_range = 0.2,          # horizontal shift up to 20%
    height_shift_range = 0.2,         # vertical shift up to 20%
    zoom_range = 0.2,                 # zoom in/out by 20%
    horizontal_flip = True,           # flip horizontally
    shear_range = 0.2,                # shear 20%
    fill_mode = 'nearest',            # fill empty pixels with nearest
    )

train_gen = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    x_col = 'file_path',
    y_col = 'label',
    target_size = (224, 224),
    color_mode = 'rgb',
    class_mode = 'categorical',
    batch_size = 32,                  
    shuffle = True,
    seed = 1,
)

# validation generator

validation_datagen = ImageDataGenerator(rescale=1./255,)

validation_gen = validation_datagen.flow_from_dataframe(
    dataframe = validation_df,
    x_col = 'file_path',
    y_col = 'label',
    target_size = (224, 224),
    class_mode = 'categorical',
    batch_size = 32,
    seed = 1,
    shuffle = False
)

# test generator

test_datagen = ImageDataGenerator(rescale = 1./255,)

test_gen = test_datagen.flow_from_dataframe(
    dataframe = test_df,
    x_col = 'file_path',
    y_col = 'label',
    target_size = (224, 224),
    class_mode = 'categorical',
    batch_size = 32,
    seed = 1,
    shuffle = False
)

# MobileNetV2 model

base_model = MobileNetV2(
    input_shape = (224, 224, 3),
    include_top = False,
    weights = 'imagenet',
    pooling = 'avg',
)

# train only last layer, freeze all other layers

base_model.trainable = True
set_trainable = False

for layer in base_model.layers:
    if layer.name == 'block_16_expand':
        set_trainable = True
        layer.trainable = True
    else:
        layer.trainable = False

# add custom layers to base model
x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(256, activation = 'relu')(x)
x = layers.Dense(128, activation = 'relu')(x)
x = layers.Dense(36, activation = 'softmax')(x)

model = tf.keras.Model(inputs = base_model.input, outputs = x)

model.summary()

# compile 
model.compile(optimizer = optimizers.Adam(learning_rate = 0.001),
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

# callbacks: lower learning rate if no improvement, stop if still no improvement
cp = ModelCheckpoint('model.keras', save_best_only = True) 
es = EarlyStopping(patience = 10, restore_best_weights = True)
rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 2, min_lr = 1e-6)


history = model.fit(
    train_gen,
    epochs = 100,
    validation_data = validation_gen,
    callbacks = [cp, es, rlr]
)

# evaluate model

model = models.load_model('model.keras')

test_loss, test_acc = model.evaluate(test_gen)

print(f'test loss : {round(test_loss, 4)}')
print(f'test accuracy : {round(test_acc, 4)}')