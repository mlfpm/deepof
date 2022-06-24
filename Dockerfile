FROM continuumio/anaconda3:2021.11
WORKDIR /
COPY Pipfile* .
COPY Pipfile.lock* .
RUN apt-get clean \
 && apt-get --allow-releaseinfo-change update \
 && apt-get install -y gcc \
 && apt-get install -y --no-install-recommends libgl1-mesa-dev \
 && apt-get install -y --no-install-recommends libdatrie-dev \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir pipenv \
 && pipenv install --python=3.9 --dev --system --ignore-pipfile \
 && pip install --no-cache-dir deepof
CMD [ "/bin/bash" ]
