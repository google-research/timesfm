# Aegis3 — Local-First Offensive Security Platform for Smart Contracts & Web3 Protocols

> Personal analyst tool. EVM-first. Solidity-first. Air-gappable.
> Inspired in places by Claude Code's security model (sandboxed permissions,
> hook-based policy, MCP-style scoped tool servers, subagent context isolation,
> worktree-based ephemeral execution).

---

## 0. Design Tenets

1. **Local-first.** Every byte of source, bytecode, and finding stays on the
   analyst's machine unless the analyst opts into egress per-job.
2. **Deterministic, reproducible runs.** Pin compilers, tool versions, RPC block
   numbers, fuzz seeds. A run is a content-addressed bundle.
3. **Sandbox by default.** Each analyzer runs in a per-job container with no
   network, read-only source mount, and write-only artifact mount — modeled on
   Claude Code's permission-prompt-by-default posture.
4. **Hookable policy.** Pre/post-tool hooks gate dangerous operations (mainnet
   RPC calls, LLM egress, write to filesystem outside artifact dir). Same idea
   as Claude Code's `PreToolUse` / `PostToolUse` hooks.
5. **Findings are first-class.** Tool outputs are raw artifacts; the canonical
   object is the **normalized finding**, deduped across tools.
6. **Graph as the analyst's worldview.** The attack graph is the durable model
   the analyst reasons over; tools feed it.
7. **Bring-your-own-LLM.** Hypothesis generation is pluggable
   (Anthropic/OpenAI/local Llama/Ollama). Off by default.

---

## 1. System Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │              aegis-ui (Next.js)              │
                  │   project view · graph · findings · report   │
                  └───────────────────┬──────────────────────────┘
                                      │ HTTP (loopback only by default)
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        aegis-api  (FastAPI, Python 3.12)                  │
│    auth · projects · jobs · findings · graph · hypotheses · reports       │
└───────┬───────────────┬──────────────┬─────────────┬────────────┬─────────┘
        │               │              │             │            │
        ▼               ▼              ▼             ▼            ▼
   ┌─────────┐   ┌────────────┐  ┌───────────┐  ┌─────────┐  ┌─────────┐
   │ Ingestor │   │ Orchestr.  │  │  Graph    │  │  Hypo.  │  │ Report  │
   │  svc    │   │  (DAG)     │  │  service  │  │  engine │  │ composer│
   └────┬────┘   └──────┬─────┘  └─────┬─────┘  └────┬────┘  └────┬────┘
        │               │              │             │            │
        │     enqueue   │              │  read/write │   LLM      │  render
        ▼               ▼              ▼             ▼            ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │                           Redis (queues)                            │
   │                   q.static · q.fuzz · q.symbolic                    │
   └────────────────────────────┬────────────────────────────────────────┘
                                │
        ┌───────────────────────┼─────────────────────────────┐
        ▼                       ▼                             ▼
  ┌──────────────┐       ┌──────────────┐             ┌──────────────┐
  │ static-worker│       │ fuzz-worker  │             │ symbolic-w.  │
  │  Slither     │       │  Echidna     │             │  Halmos      │
  │  Mythril(SA) │       │  Medusa      │             │  Mythril(sym)│
  │              │       │  Foundry     │             │              │
  └──────┬───────┘       └──────┬───────┘             └──────┬───────┘
         │                      │                            │
         └──────────┬───────────┴─────────────┬──────────────┘
                    ▼                         ▼
            ┌────────────────┐        ┌────────────────────┐
            │  Postgres 16   │        │  Artifact store    │
            │  metadata,     │        │  (local FS or MinIO)│
            │  findings,     │        │  raw tool outputs, │
            │  graph         │        │  corpora, traces   │
            └────────────────┘        └────────────────────┘
```

### 1.1 Components

| Component        | Tech                  | Purpose |
|------------------|-----------------------|---------|
| `aegis-cli`      | Python (Typer)        | Headless driver: `aegis scan ./repo`, `aegis report`, `aegis graph` |
| `aegis-ui`       | Next.js + React Flow  | Local web UI (loopback). Project, graph, findings, report views |
| `aegis-api`      | FastAPI               | REST gateway. AuthN, validation, job creation, query layer |
| `aegis-ingestor` | Python                | Resolves inputs (repo URL → checkout, address → bytecode+ABI via local archive node or user-supplied RPC) |
| `aegis-orchestrator` | Python + RQ        | DAG scheduler. Plans steps, manages retries, fan-out/fan-in |
| `aegis-workers`  | Docker images         | One image per tool family. Stateless. Talk to Redis + artifact store |
| `aegis-graph`    | Python + Postgres+pgvector | Attack graph CRUD, queries, diff between runs |
| `aegis-hypo`     | Python                | Exploit hypothesis engine (LLM + rule-based seed) |
| `aegis-report`   | Python (Jinja2)       | Bounty-ready Markdown + PDF |
| `aegis-policy`   | Python + OPA-style    | Hook engine for pre/post-tool policy enforcement |

### 1.2 Inputs the platform accepts

- **Repo URL** — git clone into ephemeral worktree (Claude-Code-style isolation).
- **Local codebase path** — read-only mount; foundry/hardhat layout auto-detected.
- **Deployed contract address** (+ chain id) — fetch ABI from local Sourcify
  cache → Etherscan (with explicit user opt-in) → fallback to bytecode-only.
- **ABI + bytecode pair** — pure offline path; produces a "binary-only" project
  where Slither/Mythril/Halmos run on bytecode/IR.

---

## 2. Service Boundaries

Boundaries are drawn so that **trust** matches **blast radius**.

```
┌──────────────────────────── trust = analyst ─────────────────────────────┐
│  aegis-cli, aegis-ui (loopback), aegis-api                               │
│  - signs job specs                                                       │
│  - holds keychain handles (RPC keys, LLM keys, etherscan)                │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │ signed JobSpec
┌──────────────────── trust = orchestration core ──────────────────────────┐
│  aegis-orchestrator, aegis-graph, aegis-report                           │
│  - never executes target code                                            │
│  - reads artifacts and DB only                                           │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │ Step messages on Redis
┌──────────────────── trust = untrusted (sandboxed) ───────────────────────┐
│  aegis-workers (Slither/Foundry/Echidna/Medusa/Mythril/Halmos)           │
│  - executes target Solidity / EVM bytecode                               │
│  - no network, no host FS access, ephemeral, capability-dropped          │
│  - writes artifacts via unix socket to a single mediated channel         │
└──────────────────────────────────────────────────────────────────────────┘
```

**Why this matters:** target code (especially bytecode-only contracts) is
adversarial. We treat workers like Claude Code treats an untrusted MCP server
— scoped, allowlisted, and observable.

---

## 3. Database Schema (Postgres)

DDL-equivalent. UUID PKs, `created_at`/`updated_at` on every table omitted.

### 3.1 Project & sources

```sql
CREATE TABLE projects (
  id              UUID PRIMARY KEY,
  name            TEXT NOT NULL,
  slug            TEXT UNIQUE NOT NULL,
  scope           TEXT NOT NULL,            -- analyst-written authorization scope
  default_chain   INT  NOT NULL DEFAULT 1
);

CREATE TABLE sources (
  id              UUID PRIMARY KEY,
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind            TEXT NOT NULL,            -- 'git' | 'local' | 'address' | 'abi_bytecode'
  uri             TEXT NOT NULL,            -- repo url, path, eip-3770 addr, etc.
  ref             TEXT,                     -- commit sha or block number
  content_hash    BYTEA NOT NULL,           -- sha256 of resolved tree/bytecode
  metadata        JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE compilations (
  id              UUID PRIMARY KEY,
  source_id       UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  framework       TEXT NOT NULL,            -- 'foundry' | 'hardhat' | 'solc' | 'binary_only'
  solc_version    TEXT,
  evm_version     TEXT,
  optimizer       JSONB,
  artifacts_uri   TEXT NOT NULL,            -- pointer into artifact store
  status          TEXT NOT NULL,
  log_uri         TEXT
);

CREATE TABLE contracts (
  id              UUID PRIMARY KEY,
  compilation_id  UUID NOT NULL REFERENCES compilations(id) ON DELETE CASCADE,
  name            TEXT NOT NULL,
  path            TEXT,
  address         BYTEA,                    -- nullable for not-yet-deployed
  chain_id        INT,
  bytecode_hash   BYTEA NOT NULL,
  is_proxy        BOOLEAN NOT NULL DEFAULT FALSE,
  proxy_type      TEXT,                     -- 'eip1967' | 'uups' | 'beacon' | 'transparent' | 'diamond'
  implementation_id UUID REFERENCES contracts(id)
);
```

### 3.2 Jobs & artifacts

```sql
CREATE TABLE jobs (
  id              UUID PRIMARY KEY,
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  spec            JSONB NOT NULL,           -- canonical, signed JobSpec
  spec_hash       BYTEA NOT NULL,
  status          TEXT NOT NULL,            -- queued|running|succeeded|failed|cancelled
  started_at      TIMESTAMPTZ,
  finished_at     TIMESTAMPTZ,
  cost_seconds    INT
);

CREATE TABLE job_steps (
  id              UUID PRIMARY KEY,
  job_id          UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  tool            TEXT NOT NULL,            -- 'slither'|'foundry'|'echidna'|'medusa'|'mythril'|'halmos'
  tool_version    TEXT NOT NULL,
  inputs          JSONB NOT NULL,
  status          TEXT NOT NULL,
  exit_code       INT,
  stdout_uri      TEXT,
  stderr_uri      TEXT,
  artifacts_uri   TEXT,
  duration_ms     INT,
  resource_caps   JSONB                     -- cpu, mem, timeout, network policy
);

CREATE TABLE artifacts (
  id              UUID PRIMARY KEY,
  job_step_id     UUID REFERENCES job_steps(id) ON DELETE CASCADE,
  kind            TEXT NOT NULL,            -- 'sarif'|'trace'|'corpus'|'coverage'|'witness'|'log'
  uri             TEXT NOT NULL,
  sha256          BYTEA NOT NULL,
  size_bytes      BIGINT NOT NULL,
  signed_by       TEXT                      -- key fingerprint
);
```

### 3.3 Normalized findings

```sql
CREATE TABLE findings (
  id              UUID PRIMARY KEY,
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  job_id          UUID NOT NULL REFERENCES jobs(id),
  contract_id     UUID REFERENCES contracts(id),
  detector        TEXT NOT NULL,            -- canonical Aegis detector id, e.g. 'AEG-REENT-001'
  source_tool     TEXT NOT NULL,            -- which tool emitted it
  source_rule     TEXT NOT NULL,            -- tool-native rule id
  severity        TEXT NOT NULL,            -- info|low|medium|high|critical
  confidence      TEXT NOT NULL,            -- low|medium|high
  title           TEXT NOT NULL,
  description     TEXT NOT NULL,
  swc_id          TEXT,                     -- SWC registry id if any
  owasp_sc_2026   TEXT,                     -- e.g. 'SC01:2026'
  cwe             TEXT,
  locations       JSONB NOT NULL,           -- array of {file,line,col,bytecode_offset}
  evidence        JSONB,                    -- counter-example, witness tx, fuzz seed
  dedupe_key      BYTEA NOT NULL,           -- sha256 of (detector,contract,locations canonical)
  status          TEXT NOT NULL DEFAULT 'open',  -- open|triaged|fp|confirmed|fixed
  triage_notes    TEXT,
  UNIQUE (project_id, dedupe_key)
);

CREATE INDEX findings_project_severity_idx ON findings(project_id, severity);
CREATE INDEX findings_owasp_idx ON findings(project_id, owasp_sc_2026);

CREATE TABLE raw_tool_outputs (
  id              UUID PRIMARY KEY,
  job_step_id     UUID NOT NULL REFERENCES job_steps(id) ON DELETE CASCADE,
  format          TEXT NOT NULL,            -- 'sarif'|'json'|'echidna-yaml'|...
  payload_uri     TEXT NOT NULL
);
```

### 3.4 Attack graph

```sql
CREATE TABLE graph_nodes (
  id              UUID PRIMARY KEY,
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind            TEXT NOT NULL,
    -- 'contract'|'role'|'function'|'asset'|'external_dep'|'upgrade_slot'|'eoa'
  label           TEXT NOT NULL,
  ref_id          UUID,                     -- nullable FK back to contracts/etc.
  attrs           JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX graph_nodes_project_kind_idx ON graph_nodes(project_id, kind);

CREATE TABLE graph_edges (
  id              UUID PRIMARY KEY,
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  src_id          UUID NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
  dst_id          UUID NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
  relation        TEXT NOT NULL,
    -- 'has_role'|'can_call'|'controls'|'holds_asset'|'depends_on'
    -- |'can_upgrade'|'delegatecalls'|'reads'|'writes'|'mints'|'burns'
  weight          REAL,
  attrs           JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE roles (
  id              UUID PRIMARY KEY,
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  contract_id     UUID NOT NULL REFERENCES contracts(id),
  name            TEXT NOT NULL,            -- 'OWNER'|'ADMIN_ROLE'|...
  bytes32_id      BYTEA,                    -- AccessControl role id
  members         JSONB NOT NULL DEFAULT '[]'
);

CREATE TABLE assets (
  id              UUID PRIMARY KEY,
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind            TEXT NOT NULL,            -- 'erc20'|'erc721'|'erc1155'|'native'|'lp'|'vault_share'
  symbol          TEXT,
  address         BYTEA,
  custodian_id    UUID REFERENCES contracts(id)
);

CREATE TABLE dependencies (
  id              UUID PRIMARY KEY,
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  contract_id     UUID NOT NULL REFERENCES contracts(id),
  external_kind   TEXT NOT NULL,            -- 'oracle'|'router'|'token'|'lib'|'bridge'
  target_address  BYTEA,
  trust_level     TEXT NOT NULL             -- 'trusted'|'untrusted'|'unknown'
);

CREATE TABLE upgrade_paths (
  id              UUID PRIMARY KEY,
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  proxy_id        UUID NOT NULL REFERENCES contracts(id),
  pattern         TEXT NOT NULL,            -- eip1967|uups|beacon|transparent|diamond
  admin_node_id   UUID REFERENCES graph_nodes(id),
  timelock_seconds INT
);
```

### 3.5 Hypotheses & reports

```sql
CREATE TABLE hypotheses (
  id              UUID PRIMARY KEY,
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  title           TEXT NOT NULL,
  narrative       TEXT NOT NULL,            -- attacker story
  preconditions   JSONB NOT NULL,
  steps           JSONB NOT NULL,           -- ordered call list
  impact          TEXT NOT NULL,
  est_severity    TEXT NOT NULL,
  supporting_findings UUID[] NOT NULL,
  graph_path      JSONB,                    -- nodes/edges referenced
  status          TEXT NOT NULL DEFAULT 'proposed'  -- proposed|reproduced|refuted
);

CREATE TABLE reports (
  id              UUID PRIMARY KEY,
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  job_id          UUID REFERENCES jobs(id),
  format          TEXT NOT NULL,            -- 'md'|'pdf'|'sarif'
  uri             TEXT NOT NULL,
  sha256          BYTEA NOT NULL
);

CREATE TABLE audit_log (
  id              BIGSERIAL PRIMARY KEY,
  ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
  actor           TEXT NOT NULL,            -- 'cli:user'|'ui:user'|'worker:slither'
  action          TEXT NOT NULL,
  target          TEXT NOT NULL,
  detail          JSONB
);
```

---

## 4. API Routes

All routes are versioned (`/v1`), bound to `127.0.0.1` by default.
Auth is local-only: HMAC tokens stored in OS keychain (macOS Keychain /
Linux Secret Service / Windows Credential Manager).

### 4.1 Projects & sources

| Method | Path | Body | Notes |
|---|---|---|---|
| `POST`   | `/v1/projects`                           | `{name, scope, default_chain}` | Authorization scope is mandatory, free-text. |
| `GET`    | `/v1/projects`                           | — | |
| `GET`    | `/v1/projects/{id}`                      | — | |
| `POST`   | `/v1/projects/{id}/sources`              | `{kind, uri, ref?, abi?, bytecode?}` | Triggers ingest. |
| `GET`    | `/v1/projects/{id}/sources`              | — | |
| `GET`    | `/v1/sources/{id}`                       | — | |
| `POST`   | `/v1/sources/{id}/compile`               | `{framework?, solc?}` | Idempotent on `(source, framework, solc)`. |
| `GET`    | `/v1/projects/{id}/contracts`            | — | |

### 4.2 Jobs

| Method | Path | Body | Notes |
|---|---|---|---|
| `POST`  | `/v1/projects/{id}/jobs`                 | `{tools[], options, budget_seconds}` | Creates DAG. Returns `job_id`. |
| `GET`   | `/v1/jobs/{id}`                          | — | Includes step states. |
| `GET`   | `/v1/jobs/{id}/steps/{step_id}/logs`     | — | Streams from artifact store. |
| `POST`  | `/v1/jobs/{id}/cancel`                   | — | Cooperative cancel. |
| `POST`  | `/v1/jobs/{id}/rerun`                    | `{only_failed?: bool}` | |

### 4.3 Findings

| Method | Path | Body | Notes |
|---|---|---|---|
| `GET`   | `/v1/findings`                                    | filters: `project_id`,`severity`,`detector`,`owasp_sc_2026`,`status` |
| `GET`   | `/v1/findings/{id}`                               | — |
| `POST`  | `/v1/findings/{id}/triage`                        | `{status, notes}` |
| `POST`  | `/v1/findings/merge`                              | `{ids: []}` — analyst override of dedupe |

### 4.4 Graph

| Method | Path | Body | Notes |
|---|---|---|---|
| `GET`   | `/v1/projects/{id}/graph`                         | Cytoscape-shaped JSON. |
| `GET`   | `/v1/projects/{id}/graph/paths`                   | query: `from=<role>`, `to=<asset>`, `max_hops` |
| `POST`  | `/v1/projects/{id}/graph/annotate`                | `{node_id?, edge_id?, attrs}` |

### 4.5 Hypotheses & reports

| Method | Path | Body | Notes |
|---|---|---|---|
| `POST`  | `/v1/projects/{id}/hypotheses/generate`           | `{llm?: 'anthropic'|'ollama'|'none', max:int}` |
| `GET`   | `/v1/projects/{id}/hypotheses`                    | — |
| `POST`  | `/v1/hypotheses/{id}/reproduce`                   | Spawns a Foundry test stub. |
| `POST`  | `/v1/projects/{id}/reports`                       | `{format:'md'|'pdf', template:'bounty'|'internal'}` |
| `GET`   | `/v1/reports/{id}`                                | Returns signed download URL (loopback). |

### 4.6 Policy & system

| Method | Path | Body | Notes |
|---|---|---|---|
| `GET`   | `/v1/system/health`                               | — |
| `GET`   | `/v1/system/tools`                                | Versions of Slither/Foundry/Echidna/Medusa/Mythril/Halmos. |
| `GET`   | `/v1/policy`                                      | Current policy bundle. |
| `PUT`   | `/v1/policy`                                      | Replace policy bundle. |

---

## 5. Job Model

### 5.1 JobSpec (canonical)

```jsonc
{
  "version": 1,
  "project_id": "…",
  "source_id": "…",
  "compilation_id": "…",
  "tools": [
    { "tool": "slither",
      "options": { "detectors": "all", "exclude": ["naming-convention"] },
      "timeout_s": 600 },
    { "tool": "foundry",
      "options": { "command": "forge test --match-test invariant_", "fork_url": null },
      "timeout_s": 1200 },
    { "tool": "echidna",
      "options": { "config": "./echidna.yaml", "test_limit": 200000, "seed": 1337 },
      "timeout_s": 3600 },
    { "tool": "medusa",
      "options": { "config": "./medusa.json", "seed": 1337 },
      "timeout_s": 3600 },
    { "tool": "mythril",
      "options": { "modules": "all", "execution_timeout": 600 },
      "timeout_s": 1200 },
    { "tool": "halmos",
      "options": { "match_contract": "Invariant.*", "loop": 4 },
      "timeout_s": 1800 }
  ],
  "budget_seconds": 7200,
  "egress_policy": "deny",                       // 'deny' | 'allowlist'
  "egress_allowlist": [],
  "llm": null,
  "signed_by": "ed25519:…",
  "signature": "…"
}
```

### 5.2 DAG

```
ingest ─► compile ─┬─► slither      ┐
                   ├─► mythril      ├─► normalize ─► graph_build ─► hypo ─► report
                   ├─► foundry      │
                   ├─► echidna      │
                   ├─► medusa       │
                   └─► halmos       ┘
```

- Static tools fan out from `compile`.
- `normalize` is a fan-in barrier — all enabled analyzers must finish (or
  fail explicitly) before findings are deduped.
- `graph_build` consumes Slither IR + ABI + bytecode disasm.
- `hypo` and `report` are independent of analyzers; can be re-run cheaply.

### 5.3 Step state machine

```
QUEUED ─► RUNNING ─┬─► SUCCEEDED
                   ├─► FAILED       (exit != 0; recorded, dependents may still run)
                   ├─► TIMED_OUT
                   └─► CANCELLED
```

A step has: `tool`, `tool_version`, `image_digest`, `inputs`, `resource_caps`
(cpu, mem, wall clock, network policy), `outputs` (artifact ids).

### 5.4 Worker contract

Each worker image must:
1. Read JobStep from Redis.
2. Pull inputs from artifact store via mediated unix socket.
3. Run tool with no network (unless `egress_allowlist` says otherwise).
4. Emit results in **SARIF 2.1.0** (preferred) or tool-native JSON.
5. Sign artifacts with worker key, push back through mediated socket.
6. Publish status update.

Workers must never reach Postgres directly. Only `aegis-orchestrator` and
`aegis-api` write findings.

---

## 6. Sequence Diagrams

### 6.1 Ingest → Report (happy path)

```
analyst         api          orchestr.      ingestor    workers     graph    hypo    report
   │  POST /jobs   │              │             │           │         │        │       │
   ├──────────────►│              │             │           │         │        │       │
   │               │ create job   │             │           │         │        │       │
   │               ├─────────────►│             │           │         │        │       │
   │               │              │ resolve src │           │         │        │       │
   │               │              ├────────────►│           │         │        │       │
   │               │              │◄────────────┤ artifacts │         │        │       │
   │               │              │ compile     │           │         │        │       │
   │               │              ├─────────────────────────►(builder)│        │       │
   │               │              │ enqueue analyzers                 │        │       │
   │               │              ├──────────────────────────►        │        │       │
   │               │              │       SARIF + raw outputs          │        │       │
   │               │              │◄─────────────────────────         │        │       │
   │               │              │ normalize+dedupe (in orchestr.)   │        │       │
   │               │              │ build graph                                         │
   │               │              ├──────────────────────────────────►│        │       │
   │               │              │ generate hypotheses                        │       │
   │               │              ├───────────────────────────────────────────►│       │
   │               │              │ render report                                       │
   │               │              ├────────────────────────────────────────────────────►│
   │  GET /reports │              │                                                     │
   │◄──────────────┤              │                                                     │
```

### 6.2 Findings normalization (fan-in)

```
slither.sarif ─┐
mythril.json   ├─► normalize() ──► canonical Finding{ detector, locations, evidence }
echidna.yaml   │      │                       │
medusa.json    │      ▼                       ▼
halmos.json   ─┘   dedupe by sha256(          enrich:
foundry.json       detector|contract|locs)     - SWC, OWASP-SC-2026, CWE
                       │                       - severity reconcile (max)
                       ▼                       - confidence reconcile (vote)
                  findings table
```

### 6.3 Hypothesis generation (LLM optional)

```
analyst ── POST /hypotheses/generate ──► hypo engine
                                            │
                                            ├─► query graph: paths(role→asset)
                                            ├─► query findings: high+critical, with evidence
                                            ├─► seed templates (reentrancy, price-mani, accesss-ctrl,
                                            │    upgrade-hijack, signature-replay, oracle-stale,
                                            │    fee-on-transfer, donation, MEV-sandwich)
                                            ├─► [optional] LLM: refine narrative + preconditions
                                            │   (egress gate: pre-tool hook checks egress_policy)
                                            ▼
                                    hypotheses table (status=proposed)
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                  POST /hypotheses/{id}/reproduce   analyst triage
                              │
                              ▼
                  Foundry stub: forge test
                  → on success: status=reproduced, attach trace artifact
```

### 6.4 Sandbox execution per worker step

```
orchestr.  ──► spawn container (image@digest, --network=none, ro source mount,
                                rw artifacts mount via socket, cap_drop=ALL,
                                seccomp=aegis-default, pids=256, mem=8g, cpu=4)
                │
                ▼
           pre-tool hook (policy):
             - tool allowed?
             - options within bounds?
             - if egress requested → hard fail unless allowlisted
                │
                ▼
           run tool
                │
                ▼
           post-tool hook:
             - artifact size sane?
             - SARIF schema-valid?
             - sign artifacts
                │
                ▼
           publish step result
```

---

## 7. MVP Scope

Ship a single-binary CLI plus a local web UI that can take a Foundry repo
and produce a bounty-grade report. Defer everything else.

### 7.1 In scope

- `aegis-cli` with `init`, `scan`, `report`, `graph`.
- `aegis-api` + `aegis-ui` running on loopback.
- Inputs: **local codebase** (Foundry layout), **repo URL**, **address+ABI** via
  user-supplied RPC. (ABI+bytecode-only path stubbed.)
- Tools: **Slither**, **Foundry (forge test, invariants)**, **Echidna**.
  Mythril/Medusa/Halmos wired but feature-flagged.
- Normalized findings schema + SARIF import/export.
- Attack graph v1: contracts, roles (Ownable + AccessControl), privileged
  functions, ERC-20 assets, EIP-1967 upgrade detection.
- OWASP Smart Contract Top 10 2026 mapping table (static, hand-curated).
- Report: Markdown (bounty template) only.
- Hypotheses: rule-based templates only; LLM path stubbed and **off by default**.
- Sandbox: Docker, `--network=none`, read-only mounts.
- Auth: keychain-backed local HMAC.

### 7.2 Out of scope (post-MVP)

- Mythril/Medusa/Halmos as first-class.
- Bytecode-only project ingestion.
- Diamond proxy graph support.
- LLM-driven hypotheses with full prompt-cache + tool-use loop.
- PDF rendering.
- Multi-analyst collaboration / shared graph.
- Non-EVM (Solana/Move) — explicitly excluded.

### 7.3 Acceptance for MVP

1. `aegis scan ./examples/uniswap-v2-fork` finishes < 15 min on a laptop and
   produces ≥ 30 findings dedup'd across Slither + Foundry + Echidna.
2. Generated graph renders in UI with role→privileged-function→asset paths.
3. Markdown report contains: exec summary, OWASP-SC-2026 coverage table,
   per-finding section with location/evidence/recommendation, and at least
   one rule-based exploit hypothesis.
4. Re-running the same scan with the same seed produces byte-identical
   `findings.dedupe_key` set (reproducibility check).

---

## 8. Security Model

This is the part that matters. The platform analyzes adversarial code; we
defend the analyst's machine and the analyst's findings.

### 8.1 Threat model

| Asset | Threat actor | Threats |
|---|---|---|
| Analyst's host | Malicious target contract / corpus / ABI | Sandbox escape via tool RCE; resource exhaustion; data exfil via DNS |
| Analyst's secrets (RPC keys, Etherscan, LLM keys) | Same | Theft via process memory, env leakage |
| Findings & report | Same / opportunistic | Tampering, deletion, false-negative injection |
| Build toolchain | Supply chain attacker | Backdoored solc/Slither/etc. |
| Network | Local attacker | MITM on loopback (low), CSRF from browser tabs |

### 8.2 Controls

**Sandboxing (per worker step)** — modeled on Claude Code's permission posture
(deny-by-default, prompt-or-allowlist):
- Docker with `--network=none`, `--cap-drop=ALL`, `--read-only`,
  `--pids-limit=256`, `--memory=…`, `--cpus=…`.
- Default seccomp profile + custom denylist (`ptrace`, `process_vm_readv`,
  `unshare`, `mount`, `keyctl`).
- Source mounted read-only; artifacts written through a mediated unix socket
  (no direct host FS access).
- Container images pinned by digest. Renewed via signed manifest.

**Egress control** — like Claude Code's MCP allowlist:
- `egress_policy: deny` is the default for every job.
- `egress_allowlist` is per-job, per-host, per-port; enforced by an in-network
  proxy sidecar; logged in `audit_log`.
- LLM calls and Etherscan calls are explicit egress steps that require
  allowlist entries. They are **never** performed by sandboxed analyzer
  workers — they are performed by `aegis-ingestor` / `aegis-hypo`, which run
  outside the analyzer sandbox but inside a separate egress-enabled namespace.

**Hook-based policy** — same pattern as Claude Code `PreToolUse`/`PostToolUse`:
- `pre_step` hook validates JobStep against policy bundle (tool allowed,
  options within bounds, no shell metacharacters in user-supplied options).
- `post_step` hook validates artifacts (schema, size, signature) before they
  are committed to the DB.
- Policy bundles are signed; replacement via `PUT /v1/policy` requires the
  analyst to re-enter the keychain HMAC.

**Secrets**:
- All secrets (RPC keys, Etherscan, LLM provider keys, signing keys) live in
  the OS keychain. The API process reads them on demand and never persists
  them on disk.
- Sandboxed workers receive **no** secrets. Egress steps that need a key
  receive a single short-TTL handle, not the secret.

**Supply chain**:
- Tool binaries (slither, mythril, halmos, echidna, medusa, forge, anvil,
  solc) are pinned by SHA-256 in a signed `tools.lock`.
- A bootstrap step verifies digests at install and on every container launch.
- SBOM (CycloneDX) generated per release; published with the binary.
- Worker images built reproducibly; pinned by digest in `JobSpec`.

**Reproducibility & integrity**:
- Every run produces a content-addressed bundle: `runs/<job_id>/{spec.json,
  artifacts/…, findings.json, report.md, manifest.json}`.
- `manifest.json` lists every artifact with sha256 + signing key fingerprint.
- The bundle is the source of truth for any later dispute.

**Local API hardening**:
- Bound to `127.0.0.1` only. Optional unix-socket transport.
- HMAC tokens in `Authorization: Bearer aegis_…`; tokens scoped by route.
- CSRF: requires `X-Aegis-Token` header; same-origin enforced; no cookies.
- Origin allowlist for the UI (`http://127.0.0.1:<port>` only).
- Loopback DNS rebinding mitigation: reject requests where `Host` header is
  not `127.0.0.1` / `localhost`.

**Authorization scope (legal control, not technical)**:
- Every project requires a written `scope` field — analyst's attestation that
  they are authorized to test this target.
- Jobs targeting a deployed mainnet address must include an explicit
  `--mainnet-attest` flag and are recorded in `audit_log`.
- Aegis3 will refuse to run *active* exploits (state-changing transactions on
  mainnet) — only read-only fork simulation via local anvil.

**Privacy**:
- LLM egress is opt-in per job. Default model is local (Ollama).
- When a remote LLM is used, source code is redacted through a configurable
  filter (license headers, addresses, comments-only mode) before egress.
- Telemetry is **off**. There is no phone-home.

**Auditability**:
- Append-only `audit_log` table, mirrored to disk as JSONL signed every N
  entries with a Merkle root.
- Every API write records `actor`, `action`, `target`, `detail`.

---

## 9. Claude-Code-Inspired Features Worth Calling Out

The following Aegis3 designs are direct lifts from Claude Code's security and
ergonomics model:

| Claude Code feature | Aegis3 analogue |
|---|---|
| Sandbox / permission-prompt mode | Per-step Docker sandbox; analyst is prompted before any egress-enabled step |
| `PreToolUse` / `PostToolUse` hooks | `pre_step` / `post_step` policy hooks on every JobStep |
| MCP allowlist (scoped tool servers) | Per-job egress allowlist; per-tool capability caps |
| Subagent context isolation | Workers are stateless; receive only the inputs they need; never see secrets |
| Worktree isolation for risky edits | Repo URL ingest clones into an ephemeral worktree, deleted post-job |
| Settings precedence (project/user/policy) | Policy bundle precedence: enterprise > user > project, all signed |
| Output style configuration | Report templates are pluggable; bounty / internal / SARIF |
| Hooks executed by harness, not LLM | Policy enforced by orchestrator, never by hypothesis LLM |
| Secret scanning (GitHub MCP) | `aegis-ingestor` runs a secret-scanner over ingested repos before any worker sees them |

---

## 10. Open Questions (deliberately unresolved for v0)

1. **Diamond proxy modeling** — facets cross-cut graph relations. Probably
   needs hyper-edges or a `facet_of` relation; defer to v0.2.
2. **Cross-contract symbolic execution at scale** — Halmos + Mythril overlap;
   need a benchmark harness before deciding which is canonical.
3. **Differential analysis between runs** — graph diff is straightforward,
   finding diff requires a stable `dedupe_key` (we have one) and a
   semantic-equality fallback (we don't yet).
4. **Local LLM quality for hypothesis generation** — Ollama with a 70B class
   model is the floor. Need an eval set of historical exploits to benchmark
   precision/recall of `hypothesis → reproduced`.
