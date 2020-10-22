"""
Extracts lists of songs and downloads them
For the script to run, please install spotdl
@belovm96
"""
import os
import argparse

class DownloadAudio:
    """
    Class for downloading audio data
    """
    def __init__(self, song_l_path, path_to_save_aud, path_to_anns, ds):
        self.dataset = ds
        self.song_l_path = song_l_path
        self.path_to_save_aud = path_to_save_aud
        self.path_to_anns = path_to_anns

    def download(self):
        os.system(f'spotdl --list={self.song_l_path} --overwrite skip -f {self.path_to_save_aud}')  

    def make_song_list(self):
        """
        Iterating through files and saving song names and paths to songs in text files
        """
        if self.dataset == 'McGill-Billboard':
            artist_title = []
            paths = []
            for dir in os.listdir(self.path_to_anns):
                files = os.listdir(f'{self.path_to_anns}/{dir}')
                chord_f = open(f'{self.path_to_anns}/{dir}/{files[0]}', 'r')
                art_title = chord_f.readlines()[:2]
                title = art_title[0][9:-1].lower()
                artist = art_title[1][10:-1].lower()
                artist_title.append(artist+' '+title+'\n')
                paths.append(f'{self.path_to_anns}/{dir}/{files[0]}\n')
                chord_f.close()
                
            song_lst = open(f'{self.song_l_path}/songs_billboard.txt', 'w')
            path_lst = open(f'{self.song_l_path}/songs_billboard_paths.txt', 'w')
            song_lst.writelines(artist_title)
            path_lst.writelines(paths)
            song_lst.close()
            path_lst.close()

        elif self.dataset == 'RW':
            songs = []
            paths = []
            for ann in os.listdir(self.path_to_anns):
                paths.append(self.path_to_anns+'/'+ann+'\n')
                ann = ann.split('-')
                artist = ann[0][:-5].replace('_', ' ').lower().strip()
                title = ann[2][:-7].replace('_', ' ').lower().strip()
                songs.append(artist+' '+title+'\n')

            song_lst = open(f'{self.song_l_path}/songs_rw.txt', 'w')
            path_lst = open(f'{self.song_l_path}/songs_rw_paths.txt', 'w')
            path_lst.writelines(paths)
            song_lst.writelines(songs)
            song_lst.close()
            path_lst.close()

        elif self.dataset == 'USPop':
            us_pop = open(self.path_to_anns, 'r')
            path_to_anns = self.path_to_anns.split('\\')
            path_to_anns = '/'.join(path_to_anns[:-1])    
            songs = []
            paths = []
            for path in us_pop.readlines():
                save = path[1:]
                path = path.split('/')
                artist = path[2].replace('_', ' ').strip().lower()
                title = path[4][3:-5].replace('_', ' ').strip().lower()
                songs.append(artist+' '+title+'\n')
                paths.append(path_to_anns+save)

            us_pop.close()

            song_lst = open(f'{self.song_l_path}/songs_uspop.txt', 'w')
            path_lst = open(f'{self.song_l_path}/songs_uspop_paths.txt', 'w')
            path_lst.writelines(paths)
            song_lst.writelines(songs)
            song_lst.close()
            path_lst.close()

        elif self.dataset == 'Isophonics':
            artist_title = []
            paths = []
            for artist in os.listdir(self.path_to_anns):
                if artist == 'Carole King':
                    for song in os.listdir(self.path_to_anns+artist):
                        title = song[:-4].split(' ')
                        title = ' '.join(title[1:])
                        artist_title.append(artist.strip().lower()+' '+title.strip().lower())
                        paths.append(path+artist+'/'+song)
                else:
                    for album in os.listdir(self.path_to_anns+artist):
                        for song in os.listdir(self.path_to_anns+artist+'/'+album):
                            if song[-3:] == 'lab':
                                song = song.replace(' ', '_')
                                title = song[:-4].split('_')
                                if artist == 'Queen':
                                    title = ' '.join(title[1:])
                                    artist_title.append(artist.strip().lower()+' '+title.strip().lower())
                                    paths.append(self.path_to_anns+artist+'/'+album+'/'+song.replace('_', ' '))
                                elif title[0][:2] != 'CD':
                                    title = ' '.join(title[2:])
                                    artist_title.append(artist.strip().lower()+' '+title.strip().lower())
                                    paths.append(self.path_to_anns+artist+'/'+album+'/'+song)
                                else:
                                    title = ' '.join(title[4:])
                                    artist_title.append(artist.strip().lower()+' '+title.strip().lower())
                                    paths.append(self.path_to_anns+artist+'/'+album+'/'+song)
                                
            song_lst = open(f'{self.song_l_path}/songs_iso.txt', 'w')
            path_lst = open(f'{self.song_l_path}/songs_iso_path.txt', 'w')
            for i, song in enumerate(artist_title): 
                song_lst.write(song+'\n')
                path_lst.write(paths[i]+'\n')
            song_lst.close()
            path_lst.close()

        else:
            print('Check dataset name. You entered: ', self.dataset)


parser = argparse.ArgumentParser(description = "Script for downloading audio")
parser.add_argument("--ann", type=str, help="path to annotations")
parser.add_argument("--songs", type=str, help='path to store song list') 
parser.add_argument("--dw", type=str, help="path to save downloaded songs")
parser.add_argument("--ds", type=str, help="dataset name")
args = parser.parse_args()

DW = DownloadAudio(args.songs, args.dw, args.ann, args.ds)
DW.make_song_list()
