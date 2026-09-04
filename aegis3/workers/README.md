# Aegis3 worker images

One image per analyzer. Each image:

- pins the tool by version,
- has **no** secrets baked in,
- runs as a non-root user,
- expects:
  - `/src` mounted read-only (target source),
  - `/out` mounted writable (artifacts dir, also passed as `OUT_DIR`),
- writes results to `/out/result.json` (and `/out/result.sarif` where supported),
- exits 0 on success, non-zero on tool error.

Build all at once:

```bash
make workers-build
```

Or one at a time:

```bash
docker build -t aegis3/slither:dev workers/slither
```

Run from the orchestrator. Do **not** run these images by hand against
untrusted code without the sandbox flags from `aegis3.orchestrator.sandbox`.
