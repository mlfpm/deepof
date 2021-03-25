FROM continuumio/anaconda3:2019.10
COPY Pipfile .
COPY Pipfile.lock .
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1-mesa-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pipenv
RUN pipenv install --python=3.8 --dev --ignore-pipfile
RUN conda install tensorflow
CMD [ "/bin/bash" ]