# Music Chord Detection
A Tool for Chord Sequence Detection

## Requirements
Please install packages mentioned in requirements.txt file.


If you would like to use ReChord App, you will need Docker and Streamlit:
* Docker installation instructions are [here](https://docs.docker.com/get-docker/)
* Streamlit installation instructions are [here](https://docs.streamlit.io/en/stable/installation.html)


## Use Cases
* Docker App
  * Clone this repository
  * Go to this repo's root directory
  * From the root directory cd to ChordDetection-->app
  * To create docker image run *docker image build -t streamlit:app .*
  * To run docker image use *docker container run -p 8501:8501 streamlit:app*
* Command Line Tool

## Tools
Packages & Tools used for development: 
* ffmpeg
* spotdl
* tensorflow
* numpy
* seaborn
* streamlit
* matplotlib
* pydub
* tf2crf
* keras
* madmom
* docker



