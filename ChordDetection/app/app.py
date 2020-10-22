"""
Streamlit App - ReChord
@belovm96
"""
import streamlit as st
import tensorflow as tf
import madmom as mm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil
# SessionState script allows to preserve app state when page has to be refreshed to continue
import SessionState
from sklearn import preprocessing
from pydub import AudioSegment
from keras.models import load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tf2crf import CRF
import warnings
warnings.filterwarnings('ignore')

# Adjusting parameters of matplotlib figure
plt.rcParams.update({'font.size': 3.5, 'xtick.major.pad': 0, 
                     'ytick.major.pad': 5, 'axes.titlepad': 5,
                     'xtick.bottom': False, 'ytick.left': False})

st.set_option('deprecation.showPyplotGlobalUse', False)

# Setting global network params
SEQ_LEN = 1024
NUM_CLASSES = 25
FEAT_DIM = 128

# Since tf2crf is a custom class, need to build the CRF model and load the weights (can not load the model directly)
reg = tf.keras.regularizers.L2(1e-3)
input = Input(shape=(SEQ_LEN, FEAT_DIM), dtype='float32')
mid = Dense(NUM_CLASSES, input_shape=(SEQ_LEN, FEAT_DIM), activation='linear', kernel_regularizer=reg)(input)
crf = CRF(dtype='float32', sparse_target=True)
crf.sequence_lengths = SEQ_LEN
crf.output_dim = NUM_CLASSES
output = crf(mid)
model = Model(input, output)

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss=crf.loss, optimizer=opt, metrics=[crf.accuracy])

load_from = './model_01'

model.load_weights(load_from)

crf = model
# Loading FCNN feature extractor
cnn = load_model('cnn_extractor.h5')

CONTEXT_SIZE = 7
NUM_FRAMES = 2 * CONTEXT_SIZE + 1

def one_hot(class_ids, num_classes):
    """
    Creating one-hot representation from class labels
    """
    class_ids = class_ids.astype('int32')
    oh = np.zeros((len(class_ids), num_classes), dtype=np.int32)
    oh[np.arange(len(class_ids)), class_ids] = 1

    assert (oh.argmax(axis=1) == class_ids).all()
    assert (oh.sum(axis=1) == 1).all()

    return oh

def make_preds(song):
    """
    Making predictions based on the user input - the entire pipeline - extract features --> CRF --> estimated predictions --> one-hot predictions
    """
    with st.spinner('Downloading the song. Please wait...'):
        os.system(f'spotdl --song {song} --overwrite skip')
        songs = [f for f in os.listdir('.') if os.path.isfile(f) and f[-3:] == 'mp3']
        if len(songs) == 1:
            song = songs[0]
            song_name = song[:-4]
            mp3_song = song.replace(' ', '_')
            shutil.copy(song, mp3_song)
            os.remove(song)
            
            wav_song = mp3_song[:-4]+'.wav'
            os.system(f'ffmpeg -n -i {mp3_song}  {wav_song}')
            os.remove(mp3_song)
            with st.spinner(f'Annotating {song_name}...'):
                spec = mm.audio.spectrogram.LogarithmicFilteredSpectrogram(wav_song, num_channels=1, sample_rate=44100,
                fps=10, frame_size=8192, num_bands=24, fmin=65, fmax=2100, unique_filters=True)
                spec = np.array(spec)
                pad_start = np.tile(spec[0, :], (CONTEXT_SIZE, 1))
                pad_end = np.tile(spec[-1, :], (CONTEXT_SIZE, 1))
                spec = np.vstack((pad_start, spec, pad_end))

                c = 0
                batch_frame = np.zeros((1, SEQ_LEN, FEAT_DIM))
                for i in range(spec.shape[0] - 2 * CONTEXT_SIZE):
                    spec_frame = spec[i:NUM_FRAMES+i, :]
                    spec_frame = preprocessing.scale(spec_frame).transpose().reshape((1, 1, spec_frame.shape[1], spec_frame.shape[0]))
                    cnn_features = cnn.predict(spec_frame)
                    batch_frame[0, c, :] = cnn.predict(spec_frame)
                    c += 1
                    if (i + 1) % SEQ_LEN == 0:
                        batch_frame_pred = crf.predict(batch_frame)
                        if  i + 1 == SEQ_LEN:
                            final_preds = batch_frame_pred.flatten()
                        else:
                            final_preds = np.concatenate((final_preds, batch_frame_pred.flatten()))
                        batch_frame = np.zeros((1, SEQ_LEN, FEAT_DIM))
                        c = 0

                if c != 0 and i + 1 > SEQ_LEN:
                    zero_row_inds = np.where(np.sum(batch_frame, axis=1) == 0)
                    batch_frame[0, zero_row_inds, :] = batch_frame[0, c - 1, :]
                    batch_frame_pred = crf.predict(batch_frame)
                    final_preds = np.concatenate((final_preds, batch_frame_pred.flatten()[:c]))
                else:
                    zero_row_inds = np.where(np.sum(batch_frame, axis=1) == 0)
                    batch_frame[0, zero_row_inds, :] = batch_frame[0, c - 1, :]
                    batch_frame_pred = crf.predict(batch_frame)
                    final_preds = batch_frame_pred.flatten()[:c]

                one_hot_preds = one_hot(final_preds, NUM_CLASSES)

                one_hot_preds = one_hot_preds.transpose()

                st.subheader('Your song is annotated!')

                np.save('preds.npy', one_hot_preds)

                return True, wav_song, song_name
        else: 
            st.subheader('Sorry, can not annotate the song provided by the link. Please try again.')
            return False, None, None
                
# Building a chord to id mapping for plotting
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

# Reversing the mapping - map chord ids to chords, while sorting by chord id values first
chords = {k: v for k, v in sorted(note_map.items(), key=lambda item: item[1])}

st.title('ReChord')
st.header('Create chord annotations of your favourite songs!')

session_state = SessionState.get(name="", annotate_clicked=False)

song = st.text_input('Enter Spotify link of the song to annotate', '')

if st.button('Annotate'):
    session_state.annotate_clicked = True
    session_state.predictions = False
    session_state.interval = False
    session_state.wav_song = None
    session_state.song_name = None

if session_state.annotate_clicked:
    if not session_state.predictions:
        session_state.predictions, session_state.wav_song, session_state.song_name = make_preds(song)

    time_int = st.text_input('Enter time interval of the song that you would like to transcribe in seconds (format - start:end, e.g. 5:12)', '')

    if st.button('Display time interval'):
        session_state.interval = True

    if session_state.interval:
        one_hot_preds = np.load('preds.npy')
        with st.spinner('Please wait...'):
            time_int = time_int.split(':')
            start, end = int(time_int[0]), int(time_int[1])
            if start < end:
                track = AudioSegment.from_wav(session_state.wav_song)
                start_sec = start * 1000
                end_sec = end * 1000
                start = start * 10
                end = end * 10
                song_snippet = track[start_sec:end_sec]
                song_snippet.export('snippet.wav', format='wav')

                audio_file = open('snippet.wav', 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes)

                cmap = sns.dark_palette('purple', as_cmap=True)

                chart = sns.heatmap(one_hot_preds[:, start:end], cmap=cmap, xticklabels=np.round(np.arange(start//10, end//10, 0.1), 1), yticklabels=chords.keys(), linewidths=.03, cbar=False)

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

                chart.set_title(f'Chords for {session_state.song_name}', fontsize=15)
                chart.set_ylabel('Chords', fontsize=10)
                chart.set_xlabel('Seconds', fontsize=10)

                plt.tight_layout()
                plt.show()
                st.pyplot()

            else:
                st.subheader('Sorry, wrong interval format. Please try again.')
