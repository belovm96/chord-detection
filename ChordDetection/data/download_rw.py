"""
@belovm96
"""
import os

path = 'C:/Users/Mikhail/OneDrive/Desktop/data/robbie_williams/annotations/chords/'
songs = []
for ann in os.listdir(path):
    ann = ann.split('-')
    artist = ann[0][:-5].replace('_', ' ').lower().strip()
    title = ann[2][:-7].replace('_', ' ').lower().strip()
    songs.append(title+' '+artist)

song_lst = open('C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/songs_rw.txt', 'w')
for song in songs: 
    song_lst.write(song+'\n')
song_lst.close()

write_to = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/rw-mp3'

os.system(f'spotdl --list=C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/songs_rw.txt --overwrite skip -f {write_to}')
