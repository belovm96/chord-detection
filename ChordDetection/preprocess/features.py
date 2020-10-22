"""
Generate Logarithmically Filtered Spectrograms
Credit to @fdlm whose script was used as skeleton code
@belovm96
"""
import madmom as mm
import os
import numpy as np
import argparse

def generate_specs(SPEC, songs_path, save_to):
    for folder in os.listdir(songs_path):
        files = os.listdir(songs_path+'/'+folder)
        for file in files:
            if file[-3:] == 'wav':
                spec = SPEC(songs_path+'/'+folder+'/'+file)
                spec_np = np.array(spec)
                np.save(save_to+'/'+folder+'/'+'spec.npy', spec_np, allow_pickle=True)
    

class LogFiltSpec:
    """
    Generated logarithmically filetered spectrograms - note distance equalization step
    """
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

LOG_SPEC = LogFiltSpec(8192, 24, 65, 2100, 10, True)

parser = argparse.ArgumentParser(description = "Script for generating spectrograms from audio")
parser.add_argument("--songs_path", type=str, help="path to audio tracks")
parser.add_argument("--save_to", type=str, help='path to store spectrograms') 
args = parser.parse_args()

generate_specs(LOG_SPEC, args.songs_path, args.save_to)
