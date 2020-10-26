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

class ChordDetectionObject:
    def __init__(self):
        print("et print, ergo sum")

    def fizz(self):
        return "buzz"

class DetectChords:
    def __init__(self, seq_len=1024, num_classes=25, feat_dim=128, context_s=7, 
                 crf_weights='../ChordDetection/app/model_01', cnn_net='../ChordDetection/app/cnn_extractor.h5'):
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
        print('\nMp3 to wav conversion.')
        self.wav_song = mp3_song[:-3]+'wav'
        self.song_name = mp3_song.split('/')[-1].replace('+', ' ')[:-4]
        subprocess.call(['ffmpeg', '-i', mp3_song, self.wav_song])
        print('\nSaving wav file to data...\n')

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
        print('Predictions are created.')

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

        plt.tight_layout()
        plt.show()

        

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


def one_hot(class_ids, num_classes):
    """
    Create one-hot encoding of class ids
    """
    oh = np.zeros((len(class_ids), num_classes), dtype=np.float32)
    oh[np.arange(len(class_ids)), class_ids] = 1

    assert (oh.argmax(axis=1) == class_ids).all()
    assert (oh.sum(axis=1) == 1).all()

    return oh


class IntervalAnnotationTarget:
    def __init__(self, fps, num_classes):
        self.fps = fps
        self.num_classes = num_classes

    def _annotations_to_targets(self, annotations):
        """
        Class ID of 'no chord' should always be last!
        """
        raise NotImplementedError('Implement this')

    def _targets_to_annotations(self, targets):
        raise NotImplementedError('Implement this.')

    def _dummy_target(self):
        raise NotImplementedError('Implement this.')

    def __call__(self, target_file, num_frames=None):
        """
        Creates one-hot encodings from an annotation file
        """
        ann = np.loadtxt(target_file,
                         comments=None,
                         dtype=[('start', np.float),
                                ('end', np.float),
                                ('label', 'S50')])
        if num_frames is None:
            num_frames = np.ceil(ann['end'][-1] * self.fps)
        
        # add a dummy class at the end and at the beginning,
        # because some annotations miss it, are not exactly aligned at the end
        # or do not start at the beginning of an audio file
        targets = np.vstack((self._dummy_target(),
                             self._annotations_to_targets(ann['label']),
                             self._dummy_target()))
        
        # add the times for the dummy events
        start = np.hstack(([-np.inf], ann['start'], ann['end'][-1]))
        end = np.hstack((ann['start'][0], ann['end'], [np.inf]))
        
        frame_times = np.arange(num_frames, dtype=np.float) / self.fps
        
        start = np.round(start, decimals=3)
        end = np.round(end, decimals=3)
        frame_times = np.round(frame_times, decimals=3)
        
        target_per_frame = ((start <= frame_times[:, np.newaxis]) & (frame_times[:, np.newaxis] < end))
        
        assert (target_per_frame.sum(axis=1) == 1).all()
        
        return targets[np.nonzero(target_per_frame)[1]].astype(np.float32)
        

class ChordsMajMin(IntervalAnnotationTarget):
    def __init__(self, fps):
        # 25 classes - 12 minor, 12 major, one "No Chord"
        super(ChordsMajMin, self).__init__(fps, 25)

    @property
    def name(self):
        return 'chords_majmin_fps={}'.format(self.fps)

    def _dummy_target(self):
        dt = np.zeros(self.num_classes, dtype=np.float32)
        dt[-1] = 1
        return dt

    def _annotations_to_targets(self, labels):
        """
        Maps chord annotations to 25 classes (12 major, 12 minor, 1 no chord)
        :param labels: chord labels
        :return: one-hot encoding of class id per annotation
        """
        roots = ['A','B','C','D','E','F','G']
        natural = zip(roots, [0, 2, 3, 5, 7, 8, 10])
        root_note_map = {}
        for chord, num in natural:
            root_note_map[chord] = num
            root_note_map[chord + '#'] = (num + 1) % 12
            root_note_map[chord + 'b'] = (num - 1) % 12

        root_note_map['N'] = 24
        root_note_map['X'] = 24
       
        labels = [c.decode('UTF-8') for c in labels]
        chord_root_notes = [c.split(':')[0].split('/')[0] for c in labels]
        chord_root_note_ids = np.array([root_note_map[crn] for crn in chord_root_notes])
        
        chord_type = [c.split(':')[1] if ':' in c else '' for c in labels]
        chord_type_shift = np.array([12 if 'min' in chord_t or 'dim' in chord_t else 0 for chord_t in chord_type])
        return one_hot(chord_root_note_ids + chord_type_shift, self.num_classes)


class PreprocessFeatures:
    def __init__(self, cur_dir, dir_aligned):
        self.cur_dir = cur_dir
        self.dir_aligned = dir_aligned
        
    def align(self):
        """
        Since features and targets do not align most of the time - aligning features and targets such that they are of equal shapes
        """
        features = np.load(self.cur_dir+'/'+'spec.npy', allow_pickle=True)
        targets = np.load(self.cur_dir+'/'+'target.npy', allow_pickle=True)
    
        feat_num_frames = features.shape[0]
        targ_num_frames = targets.shape[0]
        
        if feat_num_frames != targ_num_frames:
            diff = feat_num_frames - targ_num_frames
            if diff < 0:
                targets =  targets[:feat_num_frames, :]
            else:
                features = features[:targ_num_frames:, :]
            
        return features.shape, targets.shape
