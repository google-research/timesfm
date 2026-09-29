FROM trailofbits/echidna:latest

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 aegis || true
USER aegis
WORKDIR /src

COPY run.py /usr/local/bin/run.py
ENTRYPOINT ["python3", "/usr/local/bin/run.py"]
