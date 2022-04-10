FROM continuumio/anaconda3:2021.05
WORKDIR /
COPY Pipfile .
COPY Pipfile.lock .
RUN apt-get --allow-releaseinfo-change update \
 && apt-get install -y --no-install-recommends libgl1-mesa-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir pipenv \
 && pipenv install --python=3.9.12 --dev --system --ignore-pipfile \
 && pip install --no-cache-dir deepof
CMD [ "/bin/bash" ]
