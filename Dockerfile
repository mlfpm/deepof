FROM continuumio/anaconda3
COPY . /deepof
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN pip install pipenv
RUN pipenv install --python=3.8 --dev --ignore-pipfile
RUN conda install tensorflow
RUN conda install coverage
RUN pipenv install -e deepof --ignore-pipfile