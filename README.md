[![Generic badge](https://img.shields.io/badge/Insight-Artificial%20Intelligence-lightgrey)](https://shields.io/)
[![GitHub license](https://img.shields.io/github/license/belovm96/chord-detection)](https://github.com/belovm96/chord-detection/blob/master/LICENSE)
# ReChord
A Tool for Chord Sequence Detection

## Table of Contents  
- [Motivation](#Motivation)  
- [Requirements](#Requirements)  
- [Usage](#Usage)
    - [Web App](#web-app)
    - [Command Line](#command-line)
- [Approach](#Approach)
    - [Preprocessing](#first-stage---preprocessing)
    - [Modeling](#second-stage---modeling)
- [Product Design](#product-design)
- [Tools](#Tools)
- [Acknowledgements](#Acknowledgements)


## Motivation
Chord Transcription is a skill of detecting chord progressions in a musical piece by ear. While the majority of musical instrument players use sheet music to learn songs, it is rare that a musician is trained enough to figure out a composition by ear alone. Moreover, chord transcription takes hours at times, especially for beginner musicians.

While it is easy to find sheet music or tabs for old time classics, that is not the case for newly released or more obscure music.

In this project, I aim to help musicians with chord transcription and introduce ReChord - web application that transcribes chords for you in just few minutes! 

I utilized my skills in Software Development, Deep Learning, Signal Processing, and Music Theory to create the application, and hope that it will prove to be useful for fellow musicians and music enthusiasts.

Learn more: [Slides](https://docs.google.com/presentation/d/14M2gyLT41rfnpafnfzjeqVfVyiiaQKW5gx3dN0QmwBE/edit#slide=id.p)
[Demo](https://drive.google.com/file/d/1uvhqbAUlB80Brls5BFPFogwyL811ilBw/view?usp=sharing)

## Requirements
Please install dependencies:
* Clone the repo with `git clone https://github.com/belovm96/chord-detection`
* From the repo's root directory run `pip3 install -r requirements.txt`
* Python v.3.7 or higher


If you would like to use ReChord App, you will need Docker, Streamlit, and FFmpeg:
* [Docker](https://docs.docker.com/get-docker/)
* [Streamlit](https://docs.streamlit.io/en/stable/installation.html)
* [FFmpeg](https://ffmpeg.org/download.html)


## Usage
Note: Dockerized ReChord Streamlit App and ReChord Command Line Tool require GPU on your machine!
### Web App
  * Clone this repository
  * From the repo's root directory `cd ChordDetection/app`
  * Pull [this](https://hub.docker.com/layers/tensorflow/tensorflow/latest-gpu/images/sha256-37c7db66cc96481ac1ec43af2856ef65d3e664fd7f5df6b5e54855149f7f8594?context=explore) docker image - `docker pull tensorflow/tensorflow:latest-gpu`
  * Create docker image - `docker image build -t streamlit:app . `
  * Run docker image - `docker container run --gpus all -p 8501:8501 streamlit:app`
  
### Command Line
  * Clone this repository
  * Put a song that you would like to transcribe in `data` folder of the repo's root directory
  * To get chord transcriptions run `python transcribe.py --song ./data/song name`
    * If your song name has white spaces, please enclose it with quotes, e.g. `python transcribe.py --song ./data/'U2 - With Or Without You - Remastered.mp3'`
    * The script will ask you to provide time interval of the song that you would like to annotate
    * Chord - Time representations will be saved to `annotations` folder in `png` format
    
#### Example Use
  1. Run `python transcribe.py --song ./data/'U2 - With Or Without You - Remastered.mp3'`
  2. Input time interval `20:25`
  3. Chord - Time representation will be saved to annotations folder, and it looks like this:
![example annotations](/annotations/U2 - With or Without You_Interval-20-25.png)
  
## Approach
My data &#8594; model &#8594; predictions pipeline can be summarized as follows:

### First stage - Preprocessing

![audio](/static/prep.png)
&nbsp;&nbsp;&nbsp;&nbsp; Short-time Fourier Transform algorithm is used to convert raw audio signal into the Spectrogram - Time - Frequency representation of the signal. Then, a filterbank with logarithmically spaced filters is applied to the spectrogram to scale the frequency axis values with the purpose of equalizing the frequency distance between each note in all areas of the spectrogram. Finally, logarithmic function is applied to the spectrogram values to compress the value range, and spectrogram is cut up into 0.1-second spectrogram frames in a sliding window fashion. These 0.1-second spectrogram frames are fed into the deep learning model for training/inference.

### Second stage - Modeling

![modeling](/static/modeling.png)
 &nbsp;&nbsp;&nbsp;&nbsp; A Fully Convolutional Neural Network is trained on the log filtered spectrogram frames for chord prediction. However, these predictions are not used directly, since doing so ensures that the final chord sequence predictions are fragmented, which might confuse the end user of the application. Moreover, FCNN model does not exploit the fact that chords are always parts of chord progressions, i.e. sequences, losing a part of the potential predictive power. Therefore, Conditional Random Fields are introduced into the deep learning architecture to smooth out chord sequence predictions and to capture frame-to-frame dependencies between the predictions at every time step. The features extracted by the CNN are the inputs to the CRF, and the final chord sequence predictions are obtained using Viterbi Decoding.

## Product Design
![product](/static/product.png)


## Tools
Packages & Tools used for development: 
* Docker
* Streamlit
* FFmpeg
* Spotdl
* Tensorflow
* Keras
* Tf2crf
* NumPy
* Seaborn
* Matplotlib
* Madmom
* Yaml

## Acknowledgements
[1] Filip Korzeniowski whose [research paper](https://arxiv.org/pdf/1612.05082.pdf) was implemented and integrated into this application. \
[2] Insight Artificial Intelligence Program for presenting me with an opportunity to work on this project, their guidance and support.


