-- Aegis3 initial schema.
-- Postgres 16. UUID PKs. created_at/updated_at on every table.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE projects (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name            TEXT NOT NULL,
  slug            TEXT UNIQUE NOT NULL,
  scope           TEXT NOT NULL,
  default_chain   INT  NOT NULL DEFAULT 1,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE sources (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind            TEXT NOT NULL CHECK (kind IN ('git','local','address','abi_bytecode')),
  uri             TEXT NOT NULL,
  ref             TEXT,
  content_hash    BYTEA NOT NULL,
  metadata        JSONB NOT NULL DEFAULT '{}',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX sources_project_idx ON sources(project_id);

CREATE TABLE compilations (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source_id       UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  framework       TEXT NOT NULL CHECK (framework IN ('foundry','hardhat','solc','binary_only')),
  solc_version    TEXT,
  evm_version     TEXT,
  optimizer       JSONB,
  artifacts_uri   TEXT NOT NULL,
  status          TEXT NOT NULL,
  log_uri         TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE contracts (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  compilation_id    UUID NOT NULL REFERENCES compilations(id) ON DELETE CASCADE,
  name              TEXT NOT NULL,
  path              TEXT,
  address           BYTEA,
  chain_id          INT,
  bytecode_hash     BYTEA NOT NULL,
  is_proxy          BOOLEAN NOT NULL DEFAULT FALSE,
  proxy_type        TEXT,
  implementation_id UUID REFERENCES contracts(id),
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE jobs (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  spec            JSONB NOT NULL,
  spec_hash       BYTEA NOT NULL,
  status          TEXT NOT NULL CHECK (status IN ('queued','running','succeeded','failed','cancelled')),
  started_at      TIMESTAMPTZ,
  finished_at     TIMESTAMPTZ,
  cost_seconds    INT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX jobs_project_status_idx ON jobs(project_id, status);

CREATE TABLE job_steps (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id          UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  tool            TEXT NOT NULL,
  tool_version    TEXT NOT NULL,
  inputs          JSONB NOT NULL,
  status          TEXT NOT NULL,
  exit_code       INT,
  stdout_uri      TEXT,
  stderr_uri      TEXT,
  artifacts_uri   TEXT,
  duration_ms     INT,
  resource_caps   JSONB,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX job_steps_job_idx ON job_steps(job_id);

CREATE TABLE artifacts (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_step_id     UUID REFERENCES job_steps(id) ON DELETE CASCADE,
  kind            TEXT NOT NULL,
  uri             TEXT NOT NULL,
  sha256          BYTEA NOT NULL,
  size_bytes      BIGINT NOT NULL,
  signed_by       TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE findings (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  job_id          UUID NOT NULL REFERENCES jobs(id),
  contract_id     UUID REFERENCES contracts(id),
  detector        TEXT NOT NULL,
  source_tool     TEXT NOT NULL,
  source_rule     TEXT NOT NULL,
  severity        TEXT NOT NULL CHECK (severity IN ('info','low','medium','high','critical')),
  confidence      TEXT NOT NULL CHECK (confidence IN ('low','medium','high')),
  title           TEXT NOT NULL,
  description     TEXT NOT NULL,
  swc_id          TEXT,
  owasp_sc_2026   TEXT,
  cwe             TEXT,
  locations       JSONB NOT NULL,
  evidence        JSONB,
  dedupe_key      BYTEA NOT NULL,
  status          TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','triaged','fp','confirmed','fixed')),
  triage_notes    TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (project_id, dedupe_key)
);
CREATE INDEX findings_project_severity_idx ON findings(project_id, severity);
CREATE INDEX findings_owasp_idx ON findings(project_id, owasp_sc_2026);

CREATE TABLE raw_tool_outputs (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_step_id     UUID NOT NULL REFERENCES job_steps(id) ON DELETE CASCADE,
  format          TEXT NOT NULL,
  payload_uri     TEXT NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE graph_nodes (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind            TEXT NOT NULL CHECK (kind IN ('contract','role','function','asset','external_dep','upgrade_slot','eoa')),
  label           TEXT NOT NULL,
  ref_id          UUID,
  attrs           JSONB NOT NULL DEFAULT '{}',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX graph_nodes_project_kind_idx ON graph_nodes(project_id, kind);

CREATE TABLE graph_edges (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  src_id          UUID NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
  dst_id          UUID NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
  relation        TEXT NOT NULL,
  weight          REAL,
  attrs           JSONB NOT NULL DEFAULT '{}',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX graph_edges_project_relation_idx ON graph_edges(project_id, relation);

CREATE TABLE roles (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  contract_id     UUID NOT NULL REFERENCES contracts(id),
  name            TEXT NOT NULL,
  bytes32_id      BYTEA,
  members         JSONB NOT NULL DEFAULT '[]'
);

CREATE TABLE assets (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind            TEXT NOT NULL CHECK (kind IN ('erc20','erc721','erc1155','native','lp','vault_share')),
  symbol          TEXT,
  address         BYTEA,
  custodian_id    UUID REFERENCES contracts(id)
);

CREATE TABLE dependencies (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  contract_id     UUID NOT NULL REFERENCES contracts(id),
  external_kind   TEXT NOT NULL CHECK (external_kind IN ('oracle','router','token','lib','bridge')),
  target_address  BYTEA,
  trust_level     TEXT NOT NULL CHECK (trust_level IN ('trusted','untrusted','unknown'))
);

CREATE TABLE upgrade_paths (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id        UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  proxy_id          UUID NOT NULL REFERENCES contracts(id),
  pattern           TEXT NOT NULL CHECK (pattern IN ('eip1967','uups','beacon','transparent','diamond')),
  admin_node_id     UUID REFERENCES graph_nodes(id),
  timelock_seconds  INT
);

CREATE TABLE hypotheses (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id          UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  title               TEXT NOT NULL,
  narrative           TEXT NOT NULL,
  preconditions       JSONB NOT NULL,
  steps               JSONB NOT NULL,
  impact              TEXT NOT NULL,
  est_severity        TEXT NOT NULL,
  supporting_findings UUID[] NOT NULL,
  graph_path          JSONB,
  status              TEXT NOT NULL DEFAULT 'proposed' CHECK (status IN ('proposed','reproduced','refuted')),
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE reports (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id      UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  job_id          UUID REFERENCES jobs(id),
  format          TEXT NOT NULL CHECK (format IN ('md','pdf','sarif')),
  uri             TEXT NOT NULL,
  sha256          BYTEA NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE audit_log (
  id        BIGSERIAL PRIMARY KEY,
  ts        TIMESTAMPTZ NOT NULL DEFAULT now(),
  actor     TEXT NOT NULL,
  action    TEXT NOT NULL,
  target    TEXT NOT NULL,
  detail    JSONB
);
CREATE INDEX audit_log_actor_ts_idx ON audit_log(actor, ts);
