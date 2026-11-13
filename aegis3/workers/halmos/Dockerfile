FROM python:3.12-slim

ARG HALMOS_VERSION=0.1.15
ARG SOLC_VERSION=0.8.24

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "halmos==${HALMOS_VERSION}" "solc-select"
RUN solc-select install ${SOLC_VERSION} && solc-select use ${SOLC_VERSION}

RUN useradd -m -u 1000 aegis
USER aegis
WORKDIR /src

COPY run.py /usr/local/bin/run.py
ENTRYPOINT ["python3", "/usr/local/bin/run.py"]
