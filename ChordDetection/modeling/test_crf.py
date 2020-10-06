from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from tf2crf import CRF
from iterators import Batch
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SEQ_FRAMES = 1024
FEATURE_DIM = 128
BATCH_SIZE = 20
NUM_CLASSES = 25
CONTEXT_W = 7

input = Input(shape=(SEQ_FRAMES, FEATURE_DIM), dtype='float32')
mid = Dense(NUM_CLASSES, input_shape=(SEQ_FRAMES, FEATURE_DIM), activation='linear')(input)
crf = CRF(dtype='float32', sparse_target=True)
crf.sequence_lengths = SEQ_FRAMES
crf.output_dim = NUM_CLASSES
output = crf(mid)
model = Model(input, output)

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss=crf.loss, optimizer=opt, metrics=[crf.accuracy])

path_to_train = '/home/ubuntu/mcgill-billboard-train/'
path_to_val = '/home/ubuntu/mcgill-billboard-val/'
path_to_test = '/home/ubuntu/mcgill-billboard-test/'

batch_obj = Batch(BATCH_SIZE, CONTEXT_W, path_to_train, path_to_test, path_to_val, False, True)

load_from = '/home/ubuntu/chord-detection/ChordDetection/modeling/models/crf_1'

model.load_weights(load_from)

test_generator = batch_obj.test_generator_seq()

spec_batch = np.zeros((1, SEQ_FRAMES, FEATURE_DIM))
target_batch = np.zeros((1, SEQ_FRAMES))
spec = np.load('/home/ubuntu/mcgill-billboard-test/1012'+'/'+'features.npy')
target = np.load('/home/ubuntu/mcgill-billboard-test/1012'+'/'+'targets_seq.npy')
spec_batch[0, :, :] = spec
target_batch[0, :] = target
pred = model.predict(spec_batch, verbose=1)

print(pred.shape)
