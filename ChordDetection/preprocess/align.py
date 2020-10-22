"""
ALign spectrogram frames and target chord annotations
@belovm96
"""
import numpy as np
import os
import argparse

class PreprocessFeatures:
    def __init__(self, cur_dir, dir_aligned):
        self.cur_dir = cur_dir
        self.dir_aligned = dir_aligned
        
    def align(self):
        """
        Since features and targets do not align most of the time - aligning features and targets such that they are of equal shapes
        """
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
            
parser = argparse.ArgumentParser(description = "Script for aligning spectrogram frames with chord annotations")
parser.add_argument("--path", type=str, help="path to processed dataset (spectrogram and vectorized annotations)")
args = parser.parse_args()

PREP = PreprocessFeatures(args.path, args.path)
PREP.align()
