# base image
FROM tensorflow/tensorflow:latest-gpu

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

RUN apt-get install ffmpeg -y

# exposing default port for streamlit
EXPOSE 8501

# making directory of app
WORKDIR /streamlit-docker

# copy over requirements
COPY requirements.txt ./requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# install pip then packages
RUN pip3 install --upgrade pip
RUN pip3 install cython
RUN pip3 install numpy
RUN pip3 install -r requirements.txt

# copying all files over
COPY . .

# cmd to launch app when container is run
CMD streamlit run app.py

# streamlit-specific commands for config
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'
