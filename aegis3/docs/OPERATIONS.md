# Aegis3 — Operations & Bootstrap Guide

> Aegis3 is currently an **architecture design** (see `ARCHITECTURE.md`).
> This guide tells you (1) which OS to host it on and why, (2) how to bring up
> the MVP stack as it gets implemented, and (3) how to run the underlying
> tools today while the platform itself is being built.

---

## 1. Recommended Host OS

**Use Linux. Specifically Ubuntu 24.04 LTS (or Debian 12) on x86_64.**

Why Linux is the right host:

| Capability Aegis3 depends on | Linux | macOS | Windows |
|---|---|---|---|
| Native Docker (no VM hop) | ✅ | ❌ (Docker Desktop VM) | ❌ (WSL2 VM) |
| Seccomp-bpf syscall filtering | ✅ | ❌ | ❌ |
| Linux capabilities (`cap_drop=ALL`) | ✅ | partial via VM | partial |
| User namespaces / rootless containers | ✅ | ❌ | ❌ |
| cgroups v2 resource caps | ✅ | ❌ | ❌ |
| First-class Echidna / Medusa / Halmos binaries | ✅ | ✅ | ⚠️ WSL only |
| Foundry (anvil, forge, cast) | ✅ | ✅ | ⚠️ WSL only |
| Reproducible solc binaries | ✅ | ✅ | ⚠️ |

The whole sandbox model in the security doc (per-step `--network=none`,
`--cap-drop=ALL`, custom seccomp profile) only delivers its real guarantees on
a Linux host. On macOS or Windows, Docker runs inside a VM that you don't
control, so a sandbox escape from a worker still lands inside a Linux VM —
it's an extra layer, but you also lose visibility and tunability.

### Acceptable alternatives

- **macOS (Apple Silicon or Intel)** — fine for development of Aegis3 itself.
  Run analyzer workers in `linux/amd64` containers via Rosetta. Treat the
  sandbox as "best effort" rather than hardened.
- **Windows 11 with WSL2 (Ubuntu 24.04)** — workable. Install Docker inside
  the WSL2 distro, **not** Docker Desktop. The Aegis3 host process must run
  inside WSL2.
- **Headless server (your own LAN)** — fine, but keep the API bound to
  loopback and tunnel via SSH local-forward. Do **not** expose port 8787 on
  the LAN.

### Hardware floor

| | Minimum | Comfortable |
|---|---|---|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB (Echidna/Halmos are hungry) |
| Disk | 40 GB free | 200 GB SSD (corpora, fork caches) |
| GPU | none | only if running local LLM for hypotheses |

---

## 2. Step-by-Step: Bring up the host

These steps prepare a clean Ubuntu 24.04 box to run Aegis3. They are valid
today; they will be the same steps the MVP installer expects.

### 2.1 Update and install base packages

```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install \
  build-essential git curl wget jq unzip ca-certificates gnupg \
  python3 python3-venv python3-pip pipx \
  postgresql-16 redis-server \
  uidmap   # for rootless docker
```

### 2.2 Install Docker Engine (rootless preferred)

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
newgrp docker
docker run --rm hello-world   # verify
```

Optional but recommended for personal-analyst use:

```bash
dockerd-rootless-setuptool.sh install
```

### 2.3 Install the EVM toolchain

**Foundry:**

```bash
curl -L https://foundry.paradigm.xyz | bash
~/.foundry/bin/foundryup
forge --version && anvil --version && cast --version
```

**Slither + Mythril (Python):**

```bash
pipx install slither-analyzer
pipx install mythril
slither --version && myth version
```

**Echidna:**

```bash
ECHIDNA_VER=2.2.5
curl -L -o /tmp/echidna.tar.gz \
  "https://github.com/crytic/echidna/releases/download/v${ECHIDNA_VER}/echidna-${ECHIDNA_VER}-x86_64-linux.tar.gz"
sudo tar -xzf /tmp/echidna.tar.gz -C /usr/local/bin echidna
echidna --version
```

**Medusa:**

```bash
MEDUSA_VER=0.1.7
curl -L -o /tmp/medusa.tar.gz \
  "https://github.com/crytic/medusa/releases/download/v${MEDUSA_VER}/medusa-linux-x64.tar.gz"
sudo tar -xzf /tmp/medusa.tar.gz -C /usr/local/bin medusa
medusa --version
```

**Halmos:**

```bash
pipx install halmos
halmos --version
```

**solc-select** (so different repos can pin different solc versions):

```bash
pipx install solc-select
solc-select install 0.8.24 0.8.20 0.8.13
solc-select use 0.8.24
```

### 2.4 Verify each tool independently (before any Aegis3 wiring)

Spin up a throwaway repo and confirm each runs end-to-end.

```bash
mkdir -p ~/aegis-smoke && cd ~/aegis-smoke
forge init demo && cd demo
forge test -vv
slither .
myth analyze src/Counter.sol --solv 0.8.24 || true
echidna test/Counter.t.sol --contract CounterTest --test-mode assertion --test-limit 5000 || true
halmos --match-contract Counter || true
```

If all five run without environment errors, the host is ready for Aegis3.

### 2.5 Postgres + Redis bring-up

```bash
sudo systemctl enable --now postgresql redis-server
sudo -u postgres createuser -s "$USER"
createdb aegis3
psql aegis3 -c "select version();"
redis-cli ping   # → PONG
```

---

## 3. Step-by-Step: Run the Aegis3 MVP (once implemented)

This is what `aegis up` will do. Until the implementation lands, treat this
section as the contract the MVP must satisfy.

### 3.1 First-time setup

```bash
git clone https://github.com/<you>/aegis3.git ~/aegis3
cd ~/aegis3
make install            # creates venv, installs aegis-cli, aegis-api, workers
aegis init              # writes ~/.config/aegis3/config.toml + keychain entries
aegis tools doctor      # verifies slither/forge/echidna/medusa/myth/halmos
```

`aegis init` will:
- generate an Ed25519 signing key, store in OS keychain;
- generate an HMAC API token, store in keychain;
- write a default policy bundle (`egress_policy: deny`);
- pull pinned worker images by digest.

### 3.2 Daily flow

```bash
# 1. Create a project (with a written authorization scope)
aegis project create --name uniswap-fork \
  --scope "personal review of fork at $HOME/code/uniswap-v2-fork"

# 2. Add a source. Three forms:
aegis source add --project uniswap-fork --kind local --uri ~/code/uniswap-v2-fork
aegis source add --project uniswap-fork --kind git   --uri https://github.com/foo/bar --ref main
aegis source add --project uniswap-fork --kind address --uri eip3770:eth:0xabc... --rpc $RPC_URL

# 3. Compile (idempotent; cached by content hash)
aegis compile --project uniswap-fork

# 4. Run a job (DAG of analyzers, sandboxed)
aegis scan --project uniswap-fork \
  --tools slither,foundry,echidna \
  --budget 30m \
  --egress deny

# 5. Inspect findings
aegis findings list --project uniswap-fork --severity high,critical
aegis findings show <finding_id>

# 6. Browse the attack graph
aegis ui                                # opens http://127.0.0.1:8787
# or:
aegis graph paths --from ROLE:OWNER --to ASSET:USDC --max-hops 4

# 7. Generate exploit hypotheses (rule-based by default; LLM is opt-in)
aegis hypo generate --project uniswap-fork

# 8. Render bounty-ready report
aegis report --project uniswap-fork --format md > report.md
```

### 3.3 Where things live

```
~/.config/aegis3/
  config.toml                # non-secret settings
  policy.json                # signed policy bundle
~/.local/share/aegis3/
  pg/                        # if using bundled postgres data dir
  artifacts/                 # raw tool outputs, traces, corpora
  runs/<job_id>/
    spec.json
    artifacts/
    findings.json
    report.md
    manifest.json            # sha256 + signatures
~/.cache/aegis3/
  solc/                      # solc-select cache
  forks/                     # anvil fork state
  corpora/                   # echidna/medusa corpora (reusable across runs)
```

### 3.4 Stopping / cleaning up

```bash
aegis job cancel <job_id>          # cooperative cancel
aegis prune --older-than 30d       # delete old run bundles
aegis down                         # stops API + orchestrator
```

---

## 4. Until the MVP exists: the manual flow

You can already get most of Aegis3's value today by orchestrating the tools
yourself. Here's a minimal, reproducible recipe for one repo.

```bash
# 0. Pin the host context
cd ~/code/target-repo
solc-select use 0.8.24

# 1. Static
slither . --json slither.json --sarif slither.sarif

# 2. Symbolic
myth analyze ./src --solv 0.8.24 -o json > mythril.json || true
halmos --json-output halmos.json || true

# 3. Property tests / invariants
forge test -vv --json > foundry.json
echidna . --config echidna.yaml --format json > echidna.json || true
medusa fuzz --config medusa.json || true

# 4. Manually consolidate (poor-analyst version of the normalizer)
jq -s '{slither:.[0], mythril:.[1], halmos:.[2], foundry:.[3], echidna:.[4]}' \
  slither.json mythril.json halmos.json foundry.json echidna.json > combined.json
```

Then write findings into a spreadsheet or markdown table indexed by
`(detector, contract, file:line)`. That's the manual stand-in for the
`findings` table in the schema, and the markdown table is the stand-in for
the report.

This is deliberately the same shape as the eventual MVP — when the platform
ships, the only thing that changes is automation + sandboxing + dedup.

---

## 5. Common pitfalls

- **Don't run analyzers on mainnet contracts you do not have authorization
  to test.** Aegis3's `scope` field exists for a reason.
- **Don't expose the API on the LAN.** It's loopback by default. Use SSH
  local forwarding (`ssh -L 8787:localhost:8787 host`) if you need remote
  access.
- **Echidna and Halmos are RAM-hungry.** Cap with `--memory=8g` per worker;
  prefer running them sequentially on smaller hosts.
- **solc version drift.** Always pin via `solc-select` or `foundry.toml`'s
  `solc_version`. Mismatched solc is the #1 cause of phantom Slither
  findings.
- **Etherscan / RPC keys.** Keep them in the OS keychain, never in
  `.env`/`.envrc` files committed to git.
- **Rosetta on macOS** silently runs `linux/amd64` images; performance is
  fine for Slither/Mythril, painful for Echidna fuzzing campaigns.

---

## 6. Quick decision tree

```
Are you on Linux?            ─► Yes ─► §2 (host setup) ─► §3 (run MVP)
                             └► No  ─► macOS dev only? ─► §2 in a Linux VM
                                       Windows?         ─► WSL2 + §2
Is the MVP implemented yet?  ─► No  ─► §4 (manual flow)
                             └► Yes ─► §3 (`aegis` CLI)
```
