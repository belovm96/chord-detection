"""
@belovm96
"""

import os

path = 'C:/Users/Mikhail/OneDrive/Desktop/more_chord_data/'

artist_title = []
paths = []
for artist in os.listdir(path):
    if artist == 'Carole King':
        for song in os.listdir(path+artist):
            title = song[:-4].split(' ')
            title = ' '.join(title[1:])
            artist_title.append(artist.strip().lower()+' '+title.strip().lower())
            paths.append(path+artist+'/'+song)
    else:
        for album in os.listdir(path+artist):
            for song in os.listdir(path+artist+'/'+album):
                if song[-3:] == 'lab':
                    song = song.replace(' ', '_')
                    title = song[:-4].split('_')
                    if artist == 'Queen':
                        title = ' '.join(title[1:])
                        artist_title.append(artist.strip().lower()+' '+title.strip().lower())
                        paths.append(path+artist+'/'+album+'/'+song.replace('_', ' '))
                    elif title[0][:2] != 'CD':
                        title = ' '.join(title[2:])
                        artist_title.append(artist.strip().lower()+' '+title.strip().lower())
                        paths.append(path+artist+'/'+album+'/'+song)
                    else:
                        title = ' '.join(title[4:])
                        artist_title.append(artist.strip().lower()+' '+title.strip().lower())
                        paths.append(path+artist+'/'+album+'/'+song)
                    
song_lst = open('C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/songs.txt', 'w')
path_lst = open('C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/songs_path.txt', 'w')
for i, song in enumerate(artist_title): 
    song_lst.write(song+'\n')
    path_lst.write(paths[i]+'\n')
song_lst.close()
path_lst.close()

write_to = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/more-data-mp3'

os.system(f'spotdl --list=C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/songs.txt --overwrite skip -f {write_to}')  
