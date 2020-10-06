import tensorflow as tf
from tensorflow.keras import layers, models
from iterators import Batch
import warnings
warnings.filterwarnings('ignore')

model = models.Sequential()
model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(1, 105, 15), padding='same', data_format='channels_first'))
model.add(layers.Dropout(0.3))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(32, 105, 15),  padding='same', data_format='channels_first'))
model.add(layers.Dropout(0.3))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(32, 105, 15), padding='same', data_format='channels_first'))
model.add(layers.Dropout(0.3))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(32, 105, 15),  padding='same', data_format='channels_first'))
model.add(layers.Dropout(0.3))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.MaxPooling2D((2, 1), data_format='channels_first'))

model.add(layers.Conv2D(64, 3, activation='relu', input_shape=(32, 52, 15), padding='valid', data_format='channels_first'))
model.add(layers.Dropout(0.3))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.Conv2D(64, 3, activation='relu', input_shape=(64, 50, 13),  padding='valid', data_format='channels_first'))
model.add(layers.Dropout(0.3))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.MaxPooling2D((2, 1), data_format='channels_first'))

model.add(layers.Conv2D(128, (12, 9), activation='relu', input_shape=(64, 24, 11), padding='valid', data_format='channels_first'))
model.add(layers.Dropout(0.3))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.Conv2D(25, (1, 1), activation='linear', input_shape=(128, 13, 3), padding='valid', data_format='channels_first'))
model.add(layers.Dropout(0.3))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.AveragePooling2D((13, 3), data_format='channels_first'))
model.add(layers.Flatten())
model.add(layers.Softmax())


model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

load_from = '/home/ubuntu/chord-detection/ChordDetection/modeling/models/model_exp_6/model_01_drop_0001_lr'

model.load_weights(load_from)

path_to_train = '/home/ubuntu/mcgill-billboard-train/'
path_to_val = '/home/ubuntu/mcgill-billboard-val/'
path_to_test = '/home/ubuntu/mcgill-billboard-test/'

BATCH_SIZE = 1
CONTEXT_W = 7
batch_obj = Batch(BATCH_SIZE, CONTEXT_W, path_to_train, path_to_test, path_to_val, False, True)

test_generator = batch_obj.test_generator()

history = model.evaluate(test_generator, verbose=1, steps=160000)

print(history)
