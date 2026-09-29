FROM ghcr.io/foundry-rs/foundry:latest

USER root
RUN apk add --no-cache python3 py3-pip
RUN adduser -D -u 1000 aegis
USER aegis
WORKDIR /src

COPY run.py /usr/local/bin/run.py
ENTRYPOINT ["python3", "/usr/local/bin/run.py"]
