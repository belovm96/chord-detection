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
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 20, 'xtick.major.pad': 15, 
                     'ytick.major.pad': 40, 'axes.titlepad': 15,
                     'xtick.bottom': False, 'ytick.left': False, 'figure.figsize': (70, 25)})


class DetectChords:
    def __init__(self, seq_len, num_classes, feat_dim, context_s, crf_weights, cnn_net):
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.context_s = context_s
        self.num_frames = 2 * self.context_s + 1
        self.crf_weights = crf_weights
        self.cnn_net = cnn_net

    def one_hot(self, class_ids, num_classes):
        class_ids = class_ids.astype('int32')
        oh = np.zeros((len(class_ids), num_classes), dtype=np.int32)
        oh[np.arange(len(class_ids)), class_ids] = 1

        assert (oh.argmax(axis=1) == class_ids).all()
        assert (oh.sum(axis=1) == 1).all()

        return oh

    def build_crf(self):
        reg = tf.keras.regularizers.L2(1e-3)
        input = Input(shape=(self.seq_len, self.feat_dim), dtype='float32')
        mid = Dense(self.num_classes, input_shape=(self.seq_len, self.feat_dim), activation='linear', kernel_regularizer=reg)(input)
        crf = CRF(dtype='float32', sparse_target=True)
        crf.sequence_lengths = self.seq_len
        crf.output_dim = self.num_classes
        output = crf(mid)
        model = Model(input, output)

        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(loss=crf.loss, optimizer=opt, metrics=[crf.accuracy])

        model.load_weights(self.crf_weights)
        self.crf = model

    def build_cnn_extractor(self):
        self.cnn = load_model(self.cnn_net)

    def mp3_to_wav(self, mp3_song):
        print('\n\nMp3 to wav conversion. Saving wav file to data...\n')
        self.wav_song = mp3_song[:-3]+'wav'.replace(' ', '_')
        self.song_name = mp3_song.split('/')[-1].replace('+', ' ')[:-4]
        subprocess.call(['ffmpeg', '-i', mp3_song, self.wav_song])

    def predict_seq(self):
        print('\n\nMaking predictions - this might take 1-3 minutes. Please wait...\n')
        spec = mm.audio.spectrogram.LogarithmicFilteredSpectrogram(self.wav_song, 
            num_channels=1, sample_rate=44100, fps=10, frame_size=8192, 
                num_bands=24, fmin=65, fmax=2100, unique_filters=True)

        spec = np.array(spec)
        
        pad_start = np.tile(spec[0, :], (self.context_s, 1))
        pad_end = np.tile(spec[-1, :], (self.context_s, 1))
        spec = np.vstack((pad_start, spec, pad_end))
        
        c = 0
        batch_frame = np.zeros((1, self.seq_len, self.feat_dim))
        for i in range(spec.shape[0] - 2 * self.context_s):
            spec_frame = spec[i:self.num_frames+i, :]
            spec_frame = preprocessing.scale(spec_frame).transpose().reshape((1, 1, spec_frame.shape[1], spec_frame.shape[0]))
            batch_frame[0, c, :] = self.cnn.predict(spec_frame)
            c += 1
            if (i + 1) % self.seq_len == 0:
                batch_frame_pred = self.crf.predict(batch_frame)
                if  i + 1 == self.seq_len:
                    final_preds = batch_frame_pred.flatten()
                else:
                    final_preds = np.concatenate((final_preds, batch_frame_pred.flatten()))
                batch_frame = np.zeros((1, self.seq_len, self.feat_dim))
                c = 0

        if c != 0 and i + 1 > self.seq_len:
            zero_row_inds = np.where(np.sum(batch_frame, axis=1) == 0)
            batch_frame[0, zero_row_inds, :] = batch_frame[0, c - 1, :]
            batch_frame_pred = self.crf.predict(batch_frame)
            final_preds = np.concatenate((final_preds, batch_frame_pred.flatten()[:c]))
        else:
            zero_row_inds = np.where(np.sum(batch_frame, axis=1) == 0)
            batch_frame[0, zero_row_inds, :] = batch_frame[0, c - 1, :]
            batch_frame_pred = self.crf.predict(batch_frame)
            final_preds = batch_frame_pred.flatten()[:c]

        one_hot_preds = self.one_hot(final_preds, self.num_classes)

        one_hot_preds = one_hot_preds.transpose()

        self.one_hot_preds = one_hot_preds

        return one_hot_preds.shape

    def initialize_chord_axis(self):
        roots = ['A','B','C','D','E','F','G']
        natural = zip(roots, [0, 2, 3, 5, 7, 8, 10])
        note_map = {}
        for chord, num in natural:
            note_map[chord] = num
            note_map[chord+'m'] = num + 12
            if chord not in ['E', 'B']:
                note_map[chord + '#'] = (num + 1) % 12
                note_map[chord + '#'+'m'] = note_map[chord + '#'] + 12

        note_map['N'] = 24

        self.chords = {k: v for k, v in sorted(note_map.items(), key=lambda item: item[1])}

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

        #plt.tight_layout()
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
print('\n\nSuggested time interval - less or equal to 10 seconds for a more readable visual representation!\n')
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
