"""
Audio data management script

@belovm96
"""
import os
import shutil

class DataWrangling:
     def __init__(self, path_rename, path_renamed, path_convert_to, path_convert_from):
         self.path_renamed = path_renamed
         self.path_rename = path_rename
         self.path_convert_to = path_convert_to
         self.path_convert_from = path_convert_from
    
     def audio_rename(self, path_to_file, to_save_path):
        path = path_to_file.split('/')
        file_name = path[-1]
        art_song = file_name.split('-')
        
        artist, title = art_song[0], art_song[1:]
        
        str_art = [char if char != ' ' and char != '&' else '_' for char in artist]
        str_tit = [char if char != ' ' and char != '&' else '_' for word in title for char in word]
        art_tit = str_art+str_tit
        
        song_name = ''.join(art_tit)
        new_file_name_path = to_save_path+'/'+song_name
        shutil.copy(path_to_file, new_file_name_path)
        
     def mp3_to_wav(self, mp3_path, to_save_path):
        path = mp3_path.split('/')
        file_name = path[-1].strip()
        wav_name = file_name[:-4]+'.wav'
        os.system(f'ffmpeg -i {mp3_path} {to_save_path}/{wav_name}')
        
     def manage_anns(self, songs_dir, song_list, save_songs_path, anns):
        anns_path = open(anns, 'r')
        anns_path = anns_path.readlines()
        
        song_list = open(song_list, 'r')
        for c, song in enumerate(song_list.readlines()):
            art_title = song[:-1]
            art_title = art_title.strip()
            max_match = float('-inf')
            max_match_song =''
            for song in songs_dir:
                num_match = 0
                for i, char in enumerate(song):
                    if i < len(art_title) and art_title[i] == char:
                        num_match += 1   
                    if num_match > max_match:
                        max_match = num_match
                        max_match_song = song
                        
            if max_match >= len(art_title) - 5:
                print(art_title)
                os.mkdir(save_songs_path+art_title)
                shutil.copy(songs_dir[max_match_song], save_songs_path+art_title)
                shutil.copy(anns_path[c][:-1], save_songs_path+art_title)

path_rename = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/rw-mp3'
path_renamed = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/rw-mp3-renamed'
path_convert_from = path_renamed
path_convert_to = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/rw-wav-renamed'

dr_obj = DataWrangling(path_rename, path_renamed, path_convert_to, path_convert_from)

for song in os.listdir(path_rename):
    dr_obj.audio_rename(path_rename+'/'+song, path_renamed)

for song in os.listdir(path_convert_from):
    dr_obj.mp3_to_wav(path_convert_from+'/'+song, path_convert_to)

path_convert_to = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/dataset/audio_data/more-data-wav-renamed'
songs_prep = {}
for i, song in enumerate(os.listdir(path_convert_to)):
    song_sliced = song.split('_')
    song_sliced = [word for word in song_sliced if word != '']
    song_sliced = ' '.join(song_sliced).lower().replace('\'','')
    song_sliced = song_sliced[:-4]
    songs_prep[song_sliced] = path_convert_to+'/'+song

song_anns_path = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/songs.txt'
save_songs_path = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/dataset/preprocessed/'
anns = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/songs_path.txt'
dr_obj.manage_anns(songs_prep, song_anns_path, save_songs_path, anns)


