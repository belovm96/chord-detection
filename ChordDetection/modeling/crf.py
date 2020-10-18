"""
CRF model for chord sequence decoding
@belom96
"""
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from tf2crf import CRF
from iterators import Batch
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import argparse
import yaml
import warnings
warnings.filterwarnings('ignore')

class SeqCRF:
        def __init__(self, batch_size, context_w, L2, epochs, train_num, val_num, test_num,
                train_path, val_path, test_path, save_weights, save_log, lr, lr_step, lr_final, patience, 
        seq_dim, feat_dim, num_classes, weights=None, augment=True, randomise=True):
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
                self.augment = augment
                self.randomise = randomise
                self.steps_per_epoch = train_num // batch_size
                self.weights = weights
                self.num_classes = num_classes
                self.seq_dim = seq_dim
                self.feat_dim = feat_dim
                self.num_classes = num_classes
        
        def __call__(self):
                self.reg = tf.keras.regularizers.L2(self.L2)
                input = Input(shape=(self.seq_dim, self.feat_dim), dtype='float32')
                mid = Dense(self.num_classes, input_shape=(self.seq_dim, self.feat_dim), activation='softmax', kernel_regularizer=self.reg)(input)
                self.crf = CRF(dtype='float32', sparse_target=True)
                self.crf.sequence_lengths = self.seq_dim
                self.crf.output_dim = self.num_classes
                output = self.crf(mid)
                model = Model(input, output)
                
                self.model = model

                return model

        def compile(self):
                self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
                self.model.compile(loss=self.crf.loss, optimizer=self.opt, metrics=[self.crf.accuracy])
        
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

                self.training_generator = batch_obj_1.train_generator_seq()
                self.validation_generator = batch_obj_2.val_generator_seq()
                self.test_generator = batch_obj_3.test_generator_seq()

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

parser = argparse.ArgumentParser(description = "Script for training and testing of the CNN")
parser.add_argument("--path_config", type=str, help="path to config file")
args = parser.parse_args()

with open(args.path_config) as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    CRF_seq = SeqCRF(data['batch_size'], data['context_w'], data['reg']['L2'], data['epochs'], data['train_size'], data['val_size'], data['test_size'],
                data['train_path'], data['val_path'], data['test_path'], data['save_best_weights'], data['save_log'], data['lr_scheduler']['init'],
        data['lr_scheduler']['step'], data['lr_scheduler']['final'], data['lr_scheduler']['patience'], data['seq_frames'], data['feature_dim'], data['num_classes'] )

model = CRF_seq()

CRF_seq.compile()
CRF_seq.summary()

CRF_seq.callbacks()
CRF_seq.build_generators()

CRF_seq.fit_model()
