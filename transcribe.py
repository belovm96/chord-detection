import tensorflow as tf
import madmom as mm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import subprocess
from sklearn import preprocessing
from pydub import AudioSegment
from keras.models import load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tf2crf import CRF

import sys
sys.path.append('./ChordDetection')
import ChordDetection

import warnings
warnings.filterwarnings('ignore')

# Recommended figure parameters for visualization
plt.rcParams.update({'font.size': 20, 'xtick.major.pad': 15, 
                     'ytick.major.pad': 40, 'axes.titlepad': 15,
                     'xtick.bottom': False, 'ytick.left': False, 'figure.figsize': (70, 25)})


class DetectChords(ChordDetection.DetectChords):
    def __init__(self, seq_len, num_classes, feat_dim, context_s, 
                 crf_weights, cnn_net):
        super().__init__(seq_len, num_classes, feat_dim, context_s, 
                 crf_weights, cnn_net)
    
    def visualize(self, start, end):
        start = start * 10
        end = end * 10

        cmap = sns.dark_palette('purple', as_cmap=True)

        chart = sns.heatmap(self.one_hot_preds[:, start:end], cmap=cmap, xticklabels=np.round(np.arange(start//10, end//10, 0.1), 1), yticklabels=self.chords.keys(), linewidths=.03, cbar=False)

        chart.set_xticklabels(
            chart.get_xticklabels(), 
            rotation=-45, 
            horizontalalignment='right',
            fontweight='light',
            fontsize='large',
            )

        chart.set_yticklabels(
            chart.get_yticklabels(), 
            rotation=0,
            horizontalalignment='center',
            fontweight='light',
            fontsize='xx-large',
            )

        chart.set_title(f'Chords for {self.song_name}', fontsize=70)
        chart.set_ylabel('Chords', fontsize=50)
        chart.set_xlabel('Seconds', fontsize=50)

        plt.savefig(f'./annotations/{self.song_name} - Interval - {start//10} - {end//10}.png')

parser = argparse.ArgumentParser(description="Example script for chord sequence detection. \nNote: the script will do mp3 to wav conversion and store wav file in data!")
parser.add_argument("--song", type=str, help="path to the song to transcribe")
args = parser.parse_args()

crf_weights = './ChordDetection/app/model_01'
cnn_net = './ChordDetection/app/cnn_extractor.h5'

DC = DetectChords(1024, 25, 128, 7, crf_weights, cnn_net)
DC.build_cnn_extractor()
DC.build_crf()
DC.initialize_chord_axis()
DC.mp3_to_wav(args.song)
pred_shape = DC.predict_seq()
exit = False
print('\nSuggested time interval - less or equal to 10 seconds for a more readable visual representation!\n')
print(f'Song ends at {pred_shape[1]}th second.\n')
while not exit:
    format = True
    while format:
        interval = input('Please input time interval of the song that you are interested in (format - 12:20, where interval is in seconds): ')
        interval = interval.split(':')
        if len(interval) == 2:
            start, end = interval[0].strip(), interval[1].strip()
            start, end = int(start), int(end)
            if start < end and start >= 0 and start < pred_shape[1] and end >= 0 and start < pred_shape[1]:
                format = False
            else:
                print('Start value is greater than end value! Please try again.')
        else:
            print('Sorry, wrong interval format. Please try again.')

    DC.visualize(start, end)

    more = input('If you would like to save another time interval, press enter. Otherwise, enter EXIT: ')
    if more != '':
        exit = True
