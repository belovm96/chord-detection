from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from tf2crf import CRF
from iterators import Batch
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import warnings
warnings.filterwarnings('ignore')

SEQ_FRAMES = 1024
FEATURE_DIM = 128
EPOCHS = 20
BATCH_SIZE = 20
NUM_CLASSES = 25
CONTEXT_W = 7
TOTAL_EX = 400

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

batch_obj_1 = Batch(BATCH_SIZE, CONTEXT_W, path_to_train, path_to_test, path_to_val, False, True)
batch_obj_2 = Batch(BATCH_SIZE, CONTEXT_W, path_to_train, path_to_test, path_to_val, False, True)
training_generator = batch_obj_1.train_generator_seq()
validation_generator = batch_obj_2.val_generator_seq()

checkpoint_dir = '/home/ubuntu/chord-detection/ChordDetection/modeling/models/crf_1'

callbacks = [
        EarlyStopping(patience=5, verbose=1, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1, monitor='val_loss'),
        ModelCheckpoint(
            filepath=checkpoint_dir,
            verbose=1,
            save_best_only=True,
            save_weights_only=True, 
            monitor='val_accuracy'
            ),
        CSVLogger('/home/ubuntu/chord-detection/ChordDetection/modeling/training_crf.log')
        ]

steps_per_epoch = TOTAL_EX // BATCH_SIZE

model.fit(
        training_generator,
        validation_data=validation_generator,
        validation_steps=60,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS, verbose=True, 
        )

model.save_weights('crf_model')
