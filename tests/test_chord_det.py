import numpy as np
import numpy.testing as npt

import ChordDetection

def test_smoke_Chord_Dec():
    obj = ChordDetection.ChordDetectionObject()

def test_Chord_Dec_fizz():
    obj = ChordDetection.ChordDetectionObject()
    output = obj.fizz()
    npt.assert_equal(output, "buzz")
