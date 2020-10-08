"""
CNN model for feature extraction
@belom96
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from iterators import Batch
import warnings
warnings.filterwarnings('ignore')

model = models.Sequential()

reg = tf.keras.regularizers.L2(1e-5)
model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(1, 105, 15), padding='same', data_format='channels_first', kernel_regularizer=reg))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(32, 105, 15),  padding='same', data_format='channels_first', kernel_regularizer=reg))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(32, 105, 15), padding='same', data_format='channels_first', kernel_regularizer=reg))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(32, 105, 15),  padding='same', data_format='channels_first', kernel_regularizer=reg))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.MaxPooling2D((2, 1), data_format='channels_first'))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(64, 3, activation='relu', input_shape=(32, 52, 15), padding='valid', data_format='channels_first', kernel_regularizer=reg))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.Conv2D(64, 3, activation='relu', input_shape=(64, 50, 13),  padding='valid', data_format='channels_first', kernel_regularizer=reg))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.MaxPooling2D((2, 1), data_format='channels_first'))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(128, (12, 9), activation='relu', input_shape=(64, 24, 11), padding='valid', data_format='channels_first', kernel_regularizer=reg))
model.add(layers.BatchNormalization(axis=1))
model.add(layers.Dropout(0.5))


model.add(layers.Conv2D(25, (1, 1), activation='linear', input_shape=(128, 13, 3), padding='valid', data_format='channels_first', kernel_regularizer=reg))
model.add(layers.BatchNormalization(axis=1))

model.add(layers.AveragePooling2D((13, 3), data_format='channels_first'))
model.add(layers.Flatten())
model.add(layers.Softmax())

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


checkpoint_dir = '/home/ubuntu/rechord/chord-detection/ChordDetection/modeling/models/cnn/exp_2/model_02'
EPOCHS = 50
BATCH_SIZE = 500
CONTEXT_W = 7
TOTAL_FRAMES = 1400000

callbacks = [
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(
            filepath=checkpoint_dir,
            verbose=1,
            save_best_only=True,
            save_weights_only=True
            ),
        CSVLogger('/home/ubuntu/rechord/chord-detection/ChordDetection/modeling/models/cnn/exp_2/training.log')
        ]


path_to_train = '/home/ubuntu/data_train/'
path_to_val = '/home/ubuntu/data_val/'
path_to_test = '/home/ubuntu/data_test/'

batch_obj_1 = Batch(BATCH_SIZE, CONTEXT_W, path_to_train, path_to_test, path_to_val, True, True)
batch_obj_2 = Batch(BATCH_SIZE, CONTEXT_W, path_to_train, path_to_test, path_to_val, True, True)
training_generator = batch_obj_1.train_generator()
validation_generator = batch_obj_2.val_generator()

steps_per_epoch = TOTAL_FRAMES // BATCH_SIZE

model.fit(
        training_generator,
        validation_data=validation_generator,
        validation_steps=140000,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS, verbose=True, 
        )
