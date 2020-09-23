"""
Batch iterators for training, validation and testing
@belovm96
"""
import os
import numpy as np

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
            
            
    def standardize(self):
        raise NotImplementedError('Implement this.')
        
cur_dir = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/mcgill-billboard-prep/'
aligned_dir = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/mcgill-billboard-aligned/'

Prep_obj = PreprocessFeatures(cur_dir, aligned_dir)

Prep_obj.align()


class BatchIterator:
    
    def __init__(self, batch_size, context_size, augment=True, iscontext=True, randomise=True):
        self.batch_size = batch_size
        self.context_size = context_size
        self.iscontext = iscontext
        self.audment = augment
        self.randomise = randomise
        
    def training_generator(self, data_path, target_path):
        i = 0
        targets = os.listdir(target_path)
        while True:
            for ex in os.listdir(data_path):
                spec = np.load(ex)
                target = np.load(target_path+'/'+targets[i])
                target_size = target.shape[0]
                spec_size = spec.shape[0]
                if self.pad:
                    padding = np.zeros(4)
    
                
        