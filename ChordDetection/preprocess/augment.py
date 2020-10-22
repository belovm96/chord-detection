"""
Augmentation of features
@belovm96
Credit to @fdlm whose script was used as skeleton code
"""
import numpy as np
from scipy.ndimage import shift
import random


def one_hot(class_ids, num_classes):
    """
    Encoding a list of class labels into a one-hot representation
    """
    class_ids = class_ids.astype(int)
    oh = np.zeros((len(class_ids), num_classes), dtype=np.float32)
    oh[np.arange(len(class_ids)), class_ids] = 1

    assert (oh.argmax(axis=1) == class_ids).all()
    assert (oh.sum(axis=1) == 1).all()

    return oh


class SemitoneShift:
    """
    Simulating a semitone shift - adjusting targets accordingly
    """
    def __init__(self, p, max_shift, bins_per_semitone, target_type='chords_maj_min'):
        self.p = p
        self.max_shift = max_shift
        self.bins_per_semitone = bins_per_semitone

        if target_type == 'chords_maj_min':
            self.adapt_targets = self._adapt_targets_chords_maj_min
        elif target_type == 'chroma':
            self.adapt_targets = self._adapt_targets_chroma

    def _adapt_targets_chords_maj_min(self, targets, shifts):
        # TODO: target values roll over the boundary - need to fix this
        chord_classes = targets.argmax(-1)
        no_chord_class = targets.shape[-1] - 1
        no_chords = (chord_classes == no_chord_class)
        chord_roots = chord_classes % 12
        chord_majmin = chord_classes // 12

        new_chord_roots = (chord_roots + shifts) % 12
        new_chord_classes = new_chord_roots + chord_majmin * 12
        new_chord_classes[no_chords] = no_chord_class
        new_targets = one_hot(new_chord_classes, no_chord_class + 1)
        return new_targets

    def __call__(self, batch_iterator, batch_size):
        shifts = np.random.randint(-self.max_shift,
                                   self.max_shift + 1, batch_size)

        no_shift = random.sample(range(batch_size),
                                 int(batch_size * (1 - self.p)))
        
        shifts[no_shift] = 0
        new_data = []
        targ = np.zeros((batch_size, 25))
        i = 0
        for data, target in batch_iterator:
            targ[i, :] = target
            new_data.append(np.roll(
                data, shifts[i] * self.bins_per_semitone, axis=-1).transpose())
            i += 1
            
        new_targets = self.adapt_targets(targ, shifts)

        targets = []

        for i in range(len(new_data)):
            targets.append(new_targets[i, :])

        return new_data, targets


class Detuning:
    """
    Simulating detuning - at most half semitone shift - no need to adjust the targets
    """
    def __init__(self, p, max_shift, bins_per_semitone):
        if max_shift >= 0.5:
            raise ValueError('Detuning only works up to half a semitone!')
        self.p = p
        self.max_shift = max_shift
        self.bins_per_semitone = bins_per_semitone

    def __call__(self, batch_iterator, batch_size):
        shifts = np.random.rand(batch_size) * 2 * self.max_shift - \
            self.max_shift

        no_shift = random.sample(range(batch_size),
                                 int(batch_size * (1 - self.p)))
        shifts[no_shift] = 0

        new_data = []
        targets = []
        i = 0
        for data, target in batch_iterator:
            targets.append(target)
            new_data.append(shift(
                data, (shifts[i] * self.bins_per_semitone, 0)).transpose())
            i += 1
    
        return new_data, targets
