FROM python:3.12-slim

ARG SLITHER_VERSION=0.10.4
ARG SOLC_VERSION=0.8.24

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "slither-analyzer==${SLITHER_VERSION}" "solc-select"
RUN solc-select install ${SOLC_VERSION} && solc-select use ${SOLC_VERSION}

RUN useradd -m -u 1000 aegis
USER aegis
WORKDIR /src

COPY run.py /usr/local/bin/run.py
ENTRYPOINT ["python", "/usr/local/bin/run.py"]
