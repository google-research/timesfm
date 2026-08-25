FROM golang:1.22-bookworm AS build
ARG MEDUSA_VERSION=0.1.7
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && git clone --depth 1 --branch v${MEDUSA_VERSION} https://github.com/crytic/medusa /src/medusa \
 && cd /src/medusa && go build -o /usr/local/bin/medusa ./cmd/medusa

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/bin/medusa /usr/local/bin/medusa
RUN useradd -m -u 1000 aegis
USER aegis
WORKDIR /src
COPY run.py /usr/local/bin/run.py
ENTRYPOINT ["python3", "/usr/local/bin/run.py"]
