"""
Batch iterators for training, validation and testing
@belovm96
"""
import os
import numpy as np

class PreprocessFeatures:
    
    def __init__(self, feat_dir, targets_dir, feat_aligned_dir, targ_aligned_dir):
        self.feat_dir = feat_dir
        self.targets_dir = targets_dir
        self.feat_dir_aligned = feat_aligned_dir
        self.targ_aligned_dir = targ_aligned_dir
        
    def align(self):
        targets = os.listdir(self.targets_dir)
        features = os.listdir(self.feat_dir)
        i = 0
        
        assert len(targets) == len(features)
        
        for i in range(len(targets)):
            feature = np.load(self.feat_dir+'/'+features[i])
            target = np.load(self.targets_dir+'/'+targets[i])
            feat_num_frames = feature.shape[0]
            targ_num_frames = target.shape[0]
            
            if feat_num_frames != targ_num_frames:
                diff = feat_num_frames - targ_num_frames
                if diff < 0:
                    target[feat_num_frames:, :] = target[feat_num_frames]
                    feature = np.vstack((feature, np.tile(feature[-1], (abs(diff),1))))
                else:
                    feature[targ_num_frames:, :] = feature[targ_num_frames]
                    target = np.vstack((target, np.tile(target[-1], (abs(diff),1))))
                    
            assert feature.shape == target.shape
            
            np.save(self.feat_dir_aligned+'/'+features[i], feature)
            np.save(self.targets_dir_aligned+'/'+targets[i], target)
            
            i += 1
            
    def standardize(self):
        
        
        
    
    
    
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
                    padding = np.
                
                
        