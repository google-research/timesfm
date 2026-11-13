from __future__ import annotations

import structlog
from redis import Redis
from rq import Queue, Worker

from aegis3.config import settings

log = structlog.get_logger(__name__)

QUEUES = ("q.static", "q.fuzz", "q.symbolic", "q.normalize")


def run() -> None:
    log.info("orchestrator.start", redis=settings.redis_url, queues=QUEUES)
    redis = Redis.from_url(settings.redis_url)
    queues = [Queue(name, connection=redis) for name in QUEUES]
    Worker(queues, connection=redis).work(with_scheduler=True)


if __name__ == "__main__":
    run()
