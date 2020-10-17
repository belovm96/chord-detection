"""
Process Chord Annotations - Create target chord annotation per frame
Credit to @fdlm's targets.py which I used as a skeleton code for this script
@belovm96
"""

import numpy as np
import os
import argparse

def one_hot(class_ids, num_classes):
    """
    Create one-hot encoding of class ids
    """
    oh = np.zeros((len(class_ids), num_classes), dtype=np.float32)
    oh[np.arange(len(class_ids)), class_ids] = 1

    assert (oh.argmax(axis=1) == class_ids).all()
    assert (oh.sum(axis=1) == 1).all()

    return oh

def vectorize_anns(ANN, ann_path, save_path):
    for folder in os.listdir(ann_path):
        files = os.listdir(ann_path+'/'+folder)
        for file in files:
            if file[-3:] == 'lab' or file[-6:] == 'chords':
                anns = ANN(ann_path+'/'+folder+'/'+file)
                np.save(save_path+'/'+folder+'/'+'target.npy', anns, allow_pickle=True)

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
    
parser = argparse.ArgumentParser(description = "Script for vectorization of chord annotations")
parser.add_argument("--ann", type=str, help="path to annotations")
parser.add_argument("--save_to", type=str, help='path to store vectorized annotations') 
args = parser.parse_args()    

ANN = ChordsMajMin(10)

vectorize_anns(ANN, args.ann, args.save_to)
