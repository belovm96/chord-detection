"""
Process Chord Annotations - Create target chord annotation per frame
@belovm96
Credit to @fdlm's targets.py which I used as a skeleton code for this script
"""

import numpy as np
import string
import os

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
        
        
    def write_chord_predictions(self, filename, predictions):
        with open(filename, 'w') as f:
            f.writelines(['{:.3f}\t{:.3f}\t{}\n'.format(*p)
                          for p in self._targets_to_annotations(predictions)])
     

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
    
    
    def _targets_to_annotations(self, targets):
        natural = zip([0, 2, 3, 5, 7, 8, 10], string.uppercase[:7])
        sharp = map(lambda v: ((v[0] + 1) % 12, v[1] + '#'), natural)

        semitone_to_label = dict(sharp + natural)

        def pred_to_label(pred):
            if pred == 24:
                return 'N'
            return '{}:{}'.format(semitone_to_label[pred % 12],
                                  'maj' if pred < 12 else 'min')

        spf = 1. / self.fps
        labels = [(i * spf, pred_to_label(p)) for i, p in enumerate(targets)]

        # join same consequtive predictions
        prev_label = (None, None)
        uniq_labels = []

        for label in labels:
            if label[1] != prev_label[1]:
                uniq_labels.append(label)
                prev_label = label

        # end time of last label is one frame duration after
        # the last prediction time
        start_times, chord_labels = zip(*uniq_labels)
        end_times = start_times[1:] + (labels[-1][0] + spf,)

        return zip(start_times, end_times, chord_labels)
    
annot_obj = ChordsMajMin(10)
path_to_ann = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/McGill-Billboard'

for folder in os.listdir(path_to_ann):
    files = os.listdir(path_to_ann+'/'+folder)
    if len(files) > 1:
        for file in files:
            if file == 'full.lab':
                anns = annot_obj(path_to_ann+'/'+folder+'/'+file)
                np.save(path_to_ann+'/'+folder+'/'+'target.npy', anns, allow_pickle=True)
