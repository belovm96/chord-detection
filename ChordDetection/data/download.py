"""
Extracts list of songs and download them
@belovm96
"""

import os

all_songs_dir = os.listdir('C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/billboard-2.0-salami_chords.tar/McGill-Billboard/')
artist_title = []
for dir in all_songs_dir:
    file_to_open = os.listdir(f'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/billboard-2.0-salami_chords.tar/McGill-Billboard/{dir}')
    chord_f = open(f'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/billboard-2.0-salami_chords.tar/McGill-Billboard/{dir}/{file_to_open[0]}', 'r')
    art_title = chord_f.readlines()[:2]
    title = art_title[0][9:-1].lower()
    artist = art_title[1][10:].lower()
    artist_title.append(title+' '+artist)
    chord_f.close()
    
song_lst = open('C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/download/song_list.txt', 'w')
song_lst.writelines(artist_title)
song_lst.close()
  
write_to = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/download/billboard_mp3'

os.system(f'spotdl --list=C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/download/song_list.txt --overwrite skip -f {write_to}')  
