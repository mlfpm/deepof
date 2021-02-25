FROM continuumio/anaconda3
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN python -V
RUN pip install pipenv
RUN pipenv install --python=3.8 --dev --ignore-pipfile
RUN conda install tensorflow
