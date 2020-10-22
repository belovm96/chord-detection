"""
Data Format Conversion and Annotation Matching Script
@belovm96
"""
import os
import shutil
import argparse

class DataWrangling:
     def __init__(self, path_rename, path_renamed, path_convert_to, path_convert_from):
         self.path_renamed = path_renamed
         self.path_rename = path_rename
         self.path_convert_to = path_convert_to
         self.path_convert_from = path_convert_from
         self.name_to_path = {}
    
     def audio_rename(self):
        """
        Renaming songs since ffmpeg does not like whitespaces in song names...
        """
        for song in os.listdir(self.path_rename):
            path = self.path_rename+'/'+song.split('/')
            file_name = path[-1]
            art_song = file_name.split('-')
            
            artist, title = art_song[0], art_song[1:]
            
            str_art = [char if char != ' ' and char != '&' else '_' for char in artist]
            str_tit = [char if char != ' ' and char != '&' else '_' for word in title for char in word]
            art_tit = str_art+str_tit
            
            song_name = ''.join(art_tit)
            shutil.copy(self.path_rename+'/'+song, self.renamed+'/'+song_name)
        
     def mp3_to_wav(self):
        """
        Converting mp3 files to wav format using ffmpeg
        """
        for song in os.listdir(self.path_convert_from):
            path = self.path_convert_from+'/'+song.split('/')
            file_name = path[-1].strip()
            wav_name = file_name[:-4]+'.wav'
            os.system(f'ffmpeg -i {self.path_convert_from}/song {self.path_convert_to}/{wav_name}')
        
     def store(self, ann_paths, anns, save_files_to):
        """
        Looking for a corresponding chord annotation for each song
        and saving them to a folder
        """
        anns = open(anns, 'r')
        ann_paths = open(ann_paths, 'r')
        ann_paths = ann_paths.readlines()
        for i, ann in enumerate(anns.readlines()):
            ann = ann[:-1].strip()
            max_match = float('-inf')
            max_match_song =''
            for song in self.name_to_path:
                num_match = 0
                for j, char in enumerate(song):
                    if j < len(ann) and ann[j] == char:
                        num_match += 1   
                    if num_match > max_match:
                        max_match = num_match
                        max_match_song = song
        
            if max_match >= len(ann) - 5:
                if not os.path.isfile(save_files_to+'/'+ann):
                    os.mkdir(save_files_to+'/'+ann)
                    shutil.copy(self.name_to_path[max_match_song], save_files_to+'/'+ann)
                    shutil.copy(ann_paths[i][:-1], save_files_to+'/'+ann)
                
                
     def song_to_path(self):
        """
        Creating a song name --> song's path relations for future use in store function
        """
        for i, song in enumerate(os.listdir(self.path_convert_to)):
            song_sliced = song.split('_')
            song_sliced = [word for word in song_sliced if word != '' ]
            song_sliced = ' '.join(song_sliced)[:-4].lower().replace('\'','')
            self.name_to_path[song_sliced] = self.path_convert_to+'/'+song


parser = argparse.ArgumentParser(description = "Script for managing audio files and chord annotations")
parser.add_argument("--mp3", type=str, help="path to mp3 audio files")
parser.add_argument("--mp3_renamed", type=str, help='path to store renamed mp3 files') 
parser.add_argument("--wav_renamed", type=str, help='path to store renamed wav files')
parser.add_argument("--ann_paths", type=str, help='file containing paths to annotations')
parser.add_argument("--anns", type=str, help='path to file containing song names (artist title)')
parser.add_argument("--save_to", type=str, help='path to folder to save annotations and songs')
args = parser.parse_args()

D_Wr = DataWrangling(args.mp3, args.mp3_renamed, args.wav_renamed, args.mp3_renamed)
