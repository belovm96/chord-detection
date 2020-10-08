"""
@belovm96
"""
import os

us_pop = open('C:/Users/Mikhail/OneDrive/Desktop/Chord-Annotations-master/uspopLabels.txt', 'r')
songs = []
for path in us_pop.readlines():
    path = path.split('/')
    artist = path[2].replace('_', ' ').strip().lower()
    title = path[4][3:-5].replace('_', ' ').strip().lower()
    songs.append(artist+' '+title)

us_pop.close()
song_lst = open('C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/songs_uspop.txt', 'w')
for song in songs: 
    song_lst.write(song+'\n')
song_lst.close()
"""
write_to = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/uspop-mp3'

os.system(f'spotdl --list=C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/songs_uspop.txt --overwrite skip -f {write_to}')
"""