"""
Data Format Conversion and Annotation Matching Script
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
        
     def manage_anns(self, song_list, song_anns_dir, save_songs_path):
        for dir in os.listdir(song_anns_dir):
            file_to_open = os.listdir(song_anns_dir+'/'+dir)
            chord_f = open(song_anns_dir+'/'+dir+'/'+file_to_open[0], 'r')
            art_title = chord_f.readlines()[:2]
            title = art_title[0][9:-1].strip().lower().replace('\'','')
            artist = art_title[1][10:].strip().lower().replace('\'','')
            art_title = artist+' '+title
            max_match = float('-inf')
            max_match_song =''
            for song in song_list:
                num_match = 0
                for i, char in enumerate(song):
                    if i < len(art_title) and art_title[i] == char:
                        num_match += 1   
                    if num_match > max_match:
                        max_match = num_match
                        max_match_song = song
                        
            if max_match >= len(art_title) - 5:
                shutil.copy(song_list[max_match_song], save_songs_path+'/'+dir)
                        
       
path_rename = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/download/billboard_mp3'
path_renamed = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/download/billboard_mp3_renamed'
path_convert_from = path_renamed
path_convert_to = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/download/billboard_wav_renamed'

dr_obj = DataWrangling(path_rename, path_renamed, path_convert_to, path_convert_from)

for song in os.listdir(path_rename):
    dr_obj.audio_rename(path_rename+'/'+song, path_renamed)

for song in os.listdir(path_convert_from):
    dr_obj.mp3_to_wav(path_convert_from+'/'+song, path_convert_to)

songs_prep = {}
for i, song in enumerate(os.listdir(path_convert_to)):
    song_sliced = song.split('_')
    song_sliced = [word for word in song_sliced if word != '' ]
    song_sliced = ' '.join(song_sliced)[:-4].lower().replace('\'','')
    songs_prep[song_sliced] = path_convert_to+'/'+song

song_anns_path = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/billboard-2.0-salami_chords.tar/McGill-Billboard'
save_songs_path = 'C:/Users/Mikhail/OneDrive/Desktop/chord-recognition/McGill-Billboard'
dr_obj.manage_anns(songs_prep, song_anns_path, save_songs_path)

   

