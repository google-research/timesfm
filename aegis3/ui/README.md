# aegis3-ui

Next.js + React Flow UI is **scaffolding TBD**. The MVP path is API-first:
the CLI and `/v1` API are sufficient to drive scans, view findings, and
render reports. The UI will:

- bind to `127.0.0.1:8787`,
- consume `/v1/projects/{id}/graph` (Cytoscape JSON) for the attack graph,
- present findings tables with severity / OWASP-SC-2026 filters,
- never make outbound requests except to the local API.

When this is built, expect a `pnpm` workspace here with `app/`, `components/`,
and `lib/api.ts` for the typed client.
