"""
Generate Logarithmically Filtered Spectrograms
Credit to @fdlm whose script was used as skeleton code
@belovm96
"""
import madmom as mm
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


class LogFiltSpec:
    def __init__(self, frame_size, num_bands, fmin, fmax, fps, unique_filters, sample_rate=44100, fold=None):
        self.frame_size = frame_size
        self.num_bands = num_bands
        self.fmax = fmax
        self.fmin = fmin
        self.fps = fps
        self.unique_filters = unique_filters
        self.sample_rate = sample_rate

    @property
    def name(self):
        return 'lfs_fps={}_num-bands={}_fmin={}_fmax={}_frame_sizes=[{}]'.format(
                self.fps, self.num_bands, self.fmin, self.fmax,
                '-'.join(map(str, self.frame_sizes))
        ) + ('_uf' if self.unique_filters else '')

    def __call__(self, audio_file):
        spec = mm.audio.spectrogram.LogarithmicFilteredSpectrogram(
                audio_file, num_channels=1, sample_rate=self.sample_rate,
                fps=self.fps, frame_size=self.frame_size,
                num_bands=self.num_bands, fmin=self.fmin, fmax=self.fmax,
                unique_filters=self.unique_filters)
        
        return spec

log_obj = LogFiltSpec(8192, 24, 65, 2100, 10, True)
ds_path = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/McGill-Billboard'
for folder in os.listdir(ds_path):
    files = os.listdir(ds_path+'/'+folder)
    for file in files:
        if file[-3:] == 'wav':
            spec = log_obj(ds_path+'/'+folder+'/'+file)
            spec_np = np.array(spec)
            np.save(ds_path+'/'+folder+'/'+'spec.npy', spec_np, allow_pickle=True)
    
