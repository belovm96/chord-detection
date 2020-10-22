"""
CNN model for feature extraction
@belom96
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from iterators import Batch
import yaml
import argparse
import warnings
warnings.filterwarnings('ignore')

class AudioCNN:
        def __init__(self, batch_size, context_w, L2, epochs, train_num, val_num, test_num,
                train_path, val_path, test_path, save_weights, save_log, lr, lr_step, lr_final, patience, drop, weights=None, augment=True, randomise=True):
                self.batch_s = batch_size
                self.context_w = context_w
                self.L2 = L2
                self.epochs = epochs
                self.train_num = train_num
                self.val_num = val_num
                self.test_num = test_num
                self.train_path = train_path
                self.val_path = val_path
                self.test_path = test_path
                self.save_weights = save_weights
                self.save_log = save_log
                self.lr = lr
                self.lr_step = lr_step
                self.lr_final = lr_final
                self.patience = patience
                self.drop = drop
                self.augment = augment
                self.randomise = randomise
                self.steps_per_epoch = train_num // batch_size
                self.weights = weights
        
        def __call__(self):
                model = models.Sequential()

                reg = tf.keras.regularizers.L2(self.L2)
                model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(1, 105, 15), padding='same', data_format='channels_first', kernel_regularizer=reg))
                model.add(layers.BatchNormalization(axis=1))

                model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(32, 105, 15),  padding='same', data_format='channels_first', kernel_regularizer=reg))
                model.add(layers.BatchNormalization(axis=1))

                model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(32, 105, 15), padding='same', data_format='channels_first', kernel_regularizer=reg))
                model.add(layers.BatchNormalization(axis=1))

                model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(32, 105, 15),  padding='same', data_format='channels_first', kernel_regularizer=reg))
                model.add(layers.BatchNormalization(axis=1))

                model.add(layers.MaxPooling2D((2, 1), data_format='channels_first'))
                model.add(layers.Dropout(self.drop))

                model.add(layers.Conv2D(64, 3, activation='relu', input_shape=(32, 52, 15), padding='valid', data_format='channels_first', kernel_regularizer=reg))
                model.add(layers.BatchNormalization(axis=1))

                model.add(layers.Conv2D(64, 3, activation='relu', input_shape=(64, 50, 13),  padding='valid', data_format='channels_first', kernel_regularizer=reg))
                model.add(layers.BatchNormalization(axis=1))

                model.add(layers.MaxPooling2D((2, 1), data_format='channels_first'))
                model.add(layers.Dropout(self.drop))

                model.add(layers.Conv2D(128, (12, 9), activation='relu', input_shape=(64, 24, 11), padding='valid', data_format='channels_first', kernel_regularizer=reg))
                model.add(layers.BatchNormalization(axis=1))
                model.add(layers.Dropout(self.drop))

                model.add(layers.Conv2D(25, (1, 1), activation='linear', input_shape=(128, 13, 3), padding='valid', data_format='channels_first', kernel_regularizer=reg))
                model.add(layers.BatchNormalization(axis=1))

                model.add(layers.AveragePooling2D((13, 3), data_format='channels_first'))
                model.add(layers.Flatten())
                model.add(layers.Softmax())

                self.model = model

                return model

        def compile(self):
                self.model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
        
        def summary(self):
                print(self.model.summary())

        def callbacks(self):
                self.callbacks = [
                        ReduceLROnPlateau(factor=self.lr_step, patience=self.patience, min_lr=self.lr_final, verbose=1),
                        ModelCheckpoint(
                        filepath=self.save_weights,
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=True,
                        monitor='val_accuracy'),
                        CSVLogger(self.save_log)
                        ]
                
        def build_generators(self):
                batch_obj_1 = Batch(self.batch_s, self.context_w, self.train_path, self.test_path, self.val_path, self.augment, self.randomise)
                batch_obj_2 = Batch(self.batch_s, self.context_w, self.train_path, self.test_path, self.val_path, self.augment, self.randomise)
                batch_obj_3 = Batch(self.batch_s, self.context_w, self.train_path, self.test_path, self.val_path, self.augment, self.randomise)

                self.training_generator = batch_obj_1.train_generator()
                self.validation_generator = batch_obj_2.val_generator()
                self.test_generator = batch_obj_3.test_generator()

        def fit_model(self):
                self.model.fit(
                        self.training_generator,
                        validation_data=self.validation_generator,
                        validation_steps=self.val_num,
                        callbacks=self.callbacks,
                        steps_per_epoch=self.steps_per_epoch,
                        epochs=self.epochs, verbose=True, 
                        )

        def test_model(self):
                self.model.load_weights(self.weights)
                history = self.model.evaluate(self.test_generator, verbose=1, steps=self.test_num)
                print(history)

        def make_extractor(self):
                """
                Creating and saving weights of FCNN extractor network whose outputs will be used as inputs to CRF
                """
                self.model.load_weights(self.weights)
                
                model_ext = models.Sequential()
                for layer in self.model.layers[:-5]:
                        model_ext.add(layer)

                model_ext.add(layers.AveragePooling2D((13, 3), data_format='channels_first', trainable=False))

                for i in range(len(model_ext.layers[:-1])):
                        weights = self.model.layers[i].get_weights()
                        model_ext.layers[i].set_weights(weights)
                        model_ext.layers[i].trainable = False

                weights = self.model.layers[-3].get_weights()
                model_ext.layers[-1].set_weights(weights)
                model_ext.layers[-1].trainable = False
                model_ext.add(layers.Flatten(trainable=False))

                model_ext.compile(optimizer='adam',
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                        metrics=['accuracy'])

                model_ext.save('./cnn_extractor.h5')




parser = argparse.ArgumentParser(description="Script for training and testing of the CNN")
parser.add_argument("--path_config", type=str, help="path to config file")
args = parser.parse_args()

with open(args.path_config) as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    CNN = AudioCNN(data['batch_size'], data['context_w'], data['reg']['L2'], data['epochs'], data['train_size'], data['val_size'], data['test_size'],
                data['train_path'], data['val_path'], data['test_path'], data['save_best_weights'], data['save_log'], data['lr_scheduler']['init'],
        data['lr_scheduler']['step'], data['lr_scheduler']['final'], data['lr_scheduler']['patience'], data['drop_out'])

model = CNN()
CNN.compile()
CNN.summary()

CNN.callbacks()
CNN.build_generators()

CNN.make_extractor()
