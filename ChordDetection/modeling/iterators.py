"""
Batch iterators for training, validation and testing
@belovm96
"""
import os
import numpy as np
from sklearn import preprocessing
from augment import SemitoneShift, Detuning


class PreprocessFeatures:
    def __init__(self, cur_dir, dir_aligned):
        self.cur_dir = cur_dir
        self.dir_aligned = dir_aligned
        
    def align(self):
        for folder in os.listdir(self.cur_dir):
            files = os.listdir(self.cur_dir+folder)
            features = np.load(self.cur_dir+folder+'/'+files[0], allow_pickle=True)
            targets = np.load(self.cur_dir+folder+'/'+files[1], allow_pickle=True)
        
            feat_num_frames = features.shape[0]
            targ_num_frames = targets.shape[0]
            
            if feat_num_frames != targ_num_frames:
                diff = feat_num_frames - targ_num_frames
                if diff < 0:
                    targets =  targets[:feat_num_frames, :]
                else:
                    features = features[:targ_num_frames:, :]
            
            assert features.shape[0] == targets.shape[0]
            os.mkdir(self.dir_aligned+folder)
            
            np.save(self.dir_aligned+folder+'/'+files[0], features)
            np.save(self.dir_aligned+folder+'/'+files[1], targets)
            
        
cur_dir = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/mcgill-billboard-prep/'
aligned_dir = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/mcgill-billboard-aligned/'


class Batch:
    def __init__(self, batch_size, context_size, path_to_data, augment=True, randomise=False):
        self.batch_size = batch_size
        self.context_size = context_size
        self.augment = augment
        if self.augment:
            self.shift_sem = SemitoneShift(0.3, 4, 2)
            self.detune = Detuning(0.3, 0.4, 2)

        self.randomise = randomise
        self.path_to_data = path_to_data
        
    def _generator(self, batch_bool):
        if batch_bool:
            b_size = self.batch_size
        else:
            b_size = 1

        num_frames = 2 * self.context_size + 1

        batch = []

        while True:
            for folder in os.listdir(self.path_to_data):
                files = os.listdir(self.path_to_data+folder)
                spec = np.load(self.path_to_data+folder+'/'+files[0])
                target = np.load(self.path_to_data+folder+'/'+files[1])
                if self.context_size:
                    for i in range(spec.shape[0] - num_frames):
                        spect = spec[i:num_frames+i, :]
                        spect_scaled = preprocessing.scale(spect)
                        targ = target[i+self.context_size, :]
                        batch.append((spect_scaled, targ))
                        b_size -= 1
                        if b_size == 0:
                            yield batch
                            if batch_bool:
                                b_size = self.batch_size
                            else:
                                b_size = 1
                            batch = []


    def train_generator(self):
        for batch in self._generator(True):
            if self.augment:
                rand_int = np.random.randint(2, size=1)
                if rand_int == 0:
                    batch = self.shift_sem(batch, self.batch_size)
                else:
                    batch = self.detune(batch, self.batch_size)
                    
            data_batch = np.array(batch[0])
            targets = np.array(batch[1])
            
            data_batch = np.reshape(data_batch, (self.batch_size, 1, data_batch.shape[1], data_batch.shape[2]))
            yield data_batch, targets
        
    
    def val_generator(self):
        for batch in self._generator(False):
            batch_data = batch[0][0]
            targets = batch[0][1]
            
            batch_scaled = preprocessing.scale(batch_data)
            batch_scaled = batch_scaled.reshape(1, batch_scaled.shape[1], batch_scaled.shape[0]).shape
            targets = targets.reshape(1, targets.shape[0])
            
            batch_scaled = np.reshape(batch_scaled, (self.batch_size, 1, batch_scaled.shape[1], batch_scaled.shape[2]))
            
            yield batch_scaled, targets
    
    def test_generator(self):
       for batch in self._generator(False):
            batch_data = batch[0][0]
            targets = batch[0][1]
            
            batch_scaled = preprocessing.scale(batch_data)
            batch_scaled = batch_scaled.reshape(1, batch_scaled.shape[1], batch_scaled.shape[0]).shape
            targets = targets.reshape(1, targets.shape[0])
            
            batch_scaled = np.reshape(batch_scaled, (self.batch_size, 1, batch_scaled.shape[1], batch_scaled.shape[2]))
            
            yield batch_scaled, targets
                 

                    


        
                
        