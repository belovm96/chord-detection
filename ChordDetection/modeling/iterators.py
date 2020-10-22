"""
Batch iterators for training, validation and testing
@belovm96
"""
import os
import numpy as np
import random
from sklearn import preprocessing
from augment import SemitoneShift, Detuning # TODO: change import path
from keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

class Batch:
    def __init__(self, batch_size, context_size, path_to_train, path_to_test, path_to_val, augment=False, randomise=True):
        self.model = load_model('/home/ubuntu/chord-detection/ChordDetection/modeling/cnn_extractor.h5') # TODO: update model path
        self.batch_size = batch_size
        self.context_size = context_size
        self.augment = augment
        if self.augment:
            # Initializing data augmentation functions
            self.shift_sem = SemitoneShift(0.3, 2, 2)
            self.detune = Detuning(0.3, 0.4, 2)
        self.randomise = randomise
        self.path_to_train = path_to_train
        self.path_to_val = path_to_val
        self.path_to_test = path_to_test

    def _generator(self, split):
        if split == 'train':
            b_size = self.batch_size
            self.path_to_data = self.path_to_train
        elif split == 'val':
            self.path_to_data = self.path_to_val
            b_size = 1
        else:
            self.path_to_data = self.path_to_test
            b_size = 1
        
        # Total number of frames per data point
        num_frames = 2 * self.context_size + 1

        batch = []
        
        while True:
            folders = os.listdir(self.path_to_data)
            if self.randomise and split == 'train':
                random.shuffle(folders)
            for folder in folders:
                files = os.listdir(self.path_to_data+folder)
                spec = np.load(self.path_to_data+folder+'/'+'spec.npy')
                target = np.load(self.path_to_data+folder+'/'+'target.npy')
                if self.context_size:
                    for i in range(spec.shape[0] - num_frames):
                        spect = spec[i:num_frames+i, :]
                        spect_scaled = preprocessing.scale(spect)
                        targ = target[i+self.context_size, :]
                        batch.append((spect_scaled, targ))
                        b_size -= 1
                        if b_size == 0:
                            yield batch
                            if split == 'train':
                                b_size = self.batch_size
                            else:
                                b_size = 1
                            batch = []


    def _generator_seq(self, split):
        if split == 'train':
            self.path_to_data = self.path_to_train
            b_size = self.batch_size
        elif split == 'val':
            self.path_to_data = self.path_to_val 
            b_size = 1
        else:
            self.path_to_data = self.path_to_test
            b_size = 1
            
        # TODO: put these params as arguments to Batch class
        num_frames = 2 * self.context_size + 1
        seq_frames = 1024
        feature_dim = 128
        num_classes = 25
        batch = []
        
        while True:
            spec_batch = np.zeros((b_size, seq_frames, feature_dim))
            target_batch = np.zeros((b_size, seq_frames))
            folders = os.listdir(self.path_to_data)
            if self.randomise and split == 'train':
                random.shuffle(folders)
            batch_num = 0
            for folder in folders:
                spec = np.load(self.path_to_data+folder+'/'+'features.npy')
                target = np.load(self.path_to_data+folder+'/'+'targets_seq.npy')
                spec_batch[batch_num, :, :] = spec
                target_batch[batch_num, :] = target
                batch_num += 1
                if batch_num == b_size:
                    yield spec_batch, target_batch
                    batch_num = 0
                    spec_batch = np.zeros((b_size, seq_frames, feature_dim))
                    target_batch = np.zeros((b_size, seq_frames))
    
    def train_generator_seq(self):
        for data, targets in self._generator_seq('train'):
            yield data, targets

    def val_generator_seq(self):
        for data, targets in self._generator_seq('val'):
            yield data, targets

    def test_generator_seq(self):
        for data, targets in self._generator_seq('val'):
            yield data, targets

    # TODO: Simplify this function, i.e. do not use hard coded values for feature size               
    def train_generator(self):
        for batch in self._generator('train'):
            if self.augment:
                rand_int = np.random.randint(2, size=1)
                if rand_int == 0:
                    batch = self.detune(batch, self.batch_size)
                    data_batch = np.array(batch[0])
                    targets = np.array(batch[1])
                    data_batch = np.reshape(data_batch, (self.batch_size, 1, data_batch.shape[1], data_batch.shape[2]))
                else:
                    data_batch = np.zeros((self.batch_size, 1, 105, 15))
                    targets = np.zeros((self.batch_size, 25))
                    i = 0
                    for data, target in batch:
                        data = data.transpose()
                        data_batch[i, 0, :, :] = data
                        targets[i, :] = target
                        i += 1
            yield data_batch, targets
            
    # TODO: Simplify this function, i.e. do not use shape method
    def val_generator(self):
        for batch in self._generator('val'):
            batch_scaled = np.zeros((1, 1, batch[0][0].shape[1], batch[0][0].shape[0]))
            targets = np.zeros((1, batch[0][1].shape[0]))
            ex_prep = batch[0][0].transpose()
            batch_scaled[0, 0, :, :] = ex_prep
            targets[0, :] = batch[0][1]
            yield batch_scaled, targets
    
    def test_generator(self):
       for batch in self._generator('test'):
            batch_scaled = np.zeros((1, 1, batch[0][0].shape[1], batch[0][0].shape[0]))
            targets = np.zeros((1, batch[0][1].shape[0]))
            ex_prep = batch[0][0].transpose()
            batch_scaled[0, 0, :, :] = ex_prep
            targets[0, :] = batch[0][1]
            yield batch_scaled, targets

                    
    def generate_features(self, split):
        if split == 'train':
            self.path_to_data = self.path_to_train
        elif split == 'val':
            self.path_to_data = self.path_to_val 
        else:
            self.path_to_data = self.path_to_test
            
        # TODO: Do not hard code these - put them as args to the function or class
        num_frames = 2 * self.context_size + 1
        seq_frames = 1024
        feature_dim = 128
        num_classes = 1
    
        folders = os.listdir(self.path_to_data)
        for folder in folders:
            files = os.listdir(self.path_to_data+folder)
            spec = np.load(self.path_to_data+folder+'/'+'spec.npy')
            target = np.load(self.path_to_data+folder+'/'+'target.npy')
            spec_cur = np.zeros((seq_frames, feature_dim))
            target_cur = np.zeros((seq_frames))
            if self.context_size:
                for i in range(seq_frames):
                    spec_frame = spec[i:num_frames+i, :]
                    spec_frame = preprocessing.scale(spec_frame).transpose().reshape((1, 1, spec_frame.shape[1], spec_frame.shape[0]))
                    feature = self.model.predict(spec_frame)
                    spec_cur[i, :] = feature
                    ind = target[i+self.context_size, :].argmax()
                    target_cur[i] = ind
                np.save(self.path_to_data+folder+'/'+'features.npy', spec_cur)
                np.save(self.path_to_data+folder+'/'+'targets_seq.npy', target_cur)

