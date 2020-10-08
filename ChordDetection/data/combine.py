"""
Data Management Script
@belovm96
"""
import os
import shutil


ds_path = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/McGill-Billboard/'
new_dir = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/mcGill-billboard-prep/'

for folder in os.listdir(ds_path):
    files = os.listdir(ds_path+folder)
    os.mkdir(new_dir+folder)
    for file in files:
        if file[-4:] == '.npy':
            shutil.copy(ds_path+folder+'/'+file, new_dir+folder+'/'+file)
