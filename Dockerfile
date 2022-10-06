FROM python:3.9.14 as python-base
WORKDIR /
COPY poetry.lock* .
COPY pyproject.toml* .
RUN apt-get clean \
 && apt-get update \
 && apt-get upgrade \
 && apt-get --allow-releaseinfo-change update \
 && apt-get install -y gcc \
 && apt-get install -y --no-install-recommends libgl1-mesa-dev \
 && apt-get install -y --no-install-recommends libdatrie-dev \
 && apt-get install -y ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir pipx \
 && pipx install poetry==1.2.1 \
 && pipx ensurepath \
 && export PATH="$PATH:$HOME/.local/bin" \
 && poetry config virtualenvs.create false \
 && poetry install
ENV PATH="./root/.local/pipx/venvs/poetry/bin:$PATH"
CMD [ "/bin/bash" ]
