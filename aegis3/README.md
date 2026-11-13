# Aegis3

Local-first offensive security platform for smart contracts and Web3 protocols.
EVM-first, Solidity-first, air-gappable.

> Status: **pre-MVP code skeleton + design blueprints**. The architecture is
> fixed (`docs/ARCHITECTURE.md`), the multi-agent fleet is designed
> (`docs/MULTI_AGENT.md`), and the service stubs compile and run. Analyzer
> logic, the agent fleet, and the UI are still TODO. Use `docs/OPERATIONS.md`
> to bring up the host today.

## Docs

- `docs/ARCHITECTURE.md` — platform: services, DB schema, API, jobs, sandbox.
- `docs/OPERATIONS.md` — host OS choice + step-by-step bring-up.
- `docs/MULTI_AGENT.md` — the AI fleet: agents, topology, orchestration, safety.

## Features (target MVP)

- Inputs: repo URL, local codebase, deployed address+chain, ABI+bytecode pair.
- Analyzers: Slither, Foundry, Echidna, Medusa, Mythril, Halmos.
- Normalized findings schema, deduped across tools.
- Attack graph: contracts, roles, privileged functions, assets, external
  dependencies, upgrade paths.
- Outputs: technical findings, exploit hypotheses, OWASP SC Top 10 2026
  mapping, bounty-ready Markdown report.
- Local-first, sandboxed-by-default. No telemetry. No phone-home.

## Layout

```
docs/                  architecture, ops, multi-agent blueprints
src/aegis3/            Python source for cli, api, orchestrator, graph,
                       hypo, report, policy, schema
workers/               per-tool Docker images + entrypoints
db/migrations/         SQL migrations (Postgres 16)
deploy/                docker-compose + seccomp profile
ui/                    Next.js (placeholder)
tests/                 smoke tests
.github/workflows/     CI
```

## Quickstart

```bash
make install
make db-up && make db-migrate
make test
make api          # http://127.0.0.1:8787
aegis --help
aegis tools doctor
```

See `docs/OPERATIONS.md` for full host setup, including Foundry/Slither/
Echidna/Medusa/Mythril/Halmos installation.

## License

Proprietary. All rights reserved. See `LICENSE`.
