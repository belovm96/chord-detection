[![Generic badge](https://img.shields.io/badge/Insight-Artificial%20Intelligence-lightgrey)](https://shields.io/)
[![GitHub license](https://img.shields.io/github/license/belovm96/chord-detection)](https://github.com/belovm96/chord-detection/blob/master/LICENSE)
# Music Chord Detection
A Tool for Chord Sequence Detection

## Motivation
Musical community has been in need of a musical chord trascription software for some time. In this project, I utilize my skills in Software Development, Deep Learning, Signal Processing, and Music Theory to automate chord transcription.

Learn more: [Slides](https://docs.google.com/presentation/d/14M2gyLT41rfnpafnfzjeqVfVyiiaQKW5gx3dN0QmwBE/edit#slide=id.p)
[Demo](https://drive.google.com/file/d/1uvhqbAUlB80Brls5BFPFogwyL811ilBw/view?usp=sharing)

## Requirements
Please install required packages:
* Clone the repo with **git clone https://github.com/belovm96/chord-detection**
* From the repo's root directory run **pip3 install -r requirements.txt**


If you would like to use ReChord App, you will need Docker, Streamlit, and FFmpeg:
* [Docker](https://docs.docker.com/get-docker/)
* [Streamlit](https://docs.streamlit.io/en/stable/installation.html)
* [FFmpeg](https://ffmpeg.org/download.html)


## Use Cases
### Web Application
Note: Dockerized ReChord Streamlit App requires GPU on your machine!
  * Clone this repository
  * Go to this repo's root directory
  * From this repo's root directory *cd ChordDetection/app*
  * Pull [this](https://hub.docker.com/layers/tensorflow/tensorflow/latest-gpu/images/sha256-37c7db66cc96481ac1ec43af2856ef65d3e664fd7f5df6b5e54855149f7f8594?context=explore) docker image - *docker pull tensorflow/tensorflow:latest-gpu*
  * Create docker image - *docker image build -t streamlit:app .*
  * Run docker image - *docker container run --gpus all -p 8501:8501 streamlit:app*
  
### Command Line Tool
  * Coming soon...

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

## Acknowledgements
I would like to give credit to Filip Korzeniowski whose [research paper](https://arxiv.org/pdf/1612.05082.pdf) was implemented and integrated into this application.


