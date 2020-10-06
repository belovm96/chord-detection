import tensorflow as tf
from tensorflow.keras import layers, models
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
import warnings
warnings.filterwarnings('ignore')

model = models.Sequential()
model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(1, 105, 15), padding='same', data_format='channels_first'))
model.add(layers.Dropout(0))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(32, 105, 15),  padding='same', data_format='channels_first'))
model.add(layers.Dropout(0))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(32, 105, 15), padding='same', data_format='channels_first'))
model.add(layers.Dropout(0))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(32, 105, 15),  padding='same', data_format='channels_first'))
model.add(layers.Dropout(0))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.MaxPooling2D((2, 1), data_format='channels_first'))

model.add(layers.Conv2D(64, 3, activation='relu', input_shape=(32, 52, 15), padding='valid', data_format='channels_first'))
model.add(layers.Dropout(0))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.Conv2D(64, 3, activation='relu', input_shape=(64, 50, 13),  padding='valid', data_format='channels_first'))
model.add(layers.Dropout(0))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.MaxPooling2D((2, 1), data_format='channels_first'))

model.add(layers.Conv2D(128, (12, 9), activation='relu', input_shape=(64, 24, 11), padding='valid', data_format='channels_first'))
model.add(layers.Dropout(0))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.Conv2D(25, (1, 1), activation='linear', input_shape=(128, 13, 3), padding='valid', data_format='channels_first'))
model.add(layers.Dropout(0))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.AveragePooling2D((13, 3), data_format='channels_first'))
model.add(layers.Flatten())
model.add(layers.Softmax())


model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

load_from = '/home/ubuntu/chord-detection/ChordDetection/modeling/models/model_exp_6/model_01_drop_0001_lr'
model.load_weights(load_from)

new_model = models.Sequential()
for layer in model.layers[:-6]:
    new_model.add(layer)

new_model.add(layers.AveragePooling2D((13, 3), data_format='channels_first', trainable=False))

for i in range(len(new_model.layers[:-1])):
    weights = model.layers[i].get_weights()
    new_model.layers[i].set_weights(weights)
    new_model.layers[i].trainable = False

weights = model.layers[-3].get_weights()
new_model.layers[-1].set_weights(weights)
new_model.layers[-1].trainable = False
new_model.add(layers.Flatten(trainable=False))

new_model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

new_model.save('cnn_extractor.h5')




