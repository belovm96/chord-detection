"""
Generate Logarithmically Filtered Spectrograms for McGill Billboard's audio files
@belovm96
Credit to @fdlm whose script was used as skeleton code
"""
import madmom as mm
import os
import numpy as np


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
    if len(files) == 2:
        for file in files:
            if file[-4:] == '.wav':
                spec = log_obj(ds_path+'/'+folder+'/'+file)
                np.save(ds_path+'/'+folder+'/'+'spec.npy', spec)
    