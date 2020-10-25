import numpy as np
import numpy.testing as npt

import ChordDetection

"""
Sanity Checks
"""
def test_smoke_Chord_Dec():
    obj = ChordDetection.ChordDetectionObject()

def test_Chord_Dec_fizz():
    obj = ChordDetection.ChordDetectionObject()
    output = obj.fizz()
    npt.assert_equal(output, "buzz")

"""
Preprocessing Tests
"""
def test_feature_shape():
    song_path = './data/U2 - With Or Without You - Remastered.wav'
    LG_SPEC = ChordDetection.LogFiltSpec(8192, 24, 65, 2100, 10, True)
    spec = LG_SPEC(song_path)
    expected_shape = (3067, 105)
    npt.assert_equal(np.array(spec).shape, expected_shape)

def test_annotation_shape():
    ann_path = './data/03-With_or_Without_You.lab'
    ANN = ChordDetection.ChordsMajMin(10)
    ann = ANN(ann_path)
    expected_shape = (2960, 25)
    npt.assert_equal(ann.shape, expected_shape)

def test_alignment():
    path = './data'
    PREP = ChordDetection.PreprocessFeatures(path, path)
    spec_shape, target_shape = PREP.align()
    target_expected = (2960, 25)
    spec_expected = (2960, 105)
    npt.assert_equal(spec_shape, spec_expected)
    npt.assert_equal(target_shape, target_expected)
