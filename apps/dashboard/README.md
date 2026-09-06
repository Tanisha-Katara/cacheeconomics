# Hosted dashboard

This is the browser application for the networked control plane. It is separate
from both the installed `cacheeconomics` package and the existing static local
demo in `web/`.

## Security boundary

- The browser requests only organization-scoped, prompt-free API responses.
- OIDC sign-in uses Authorization Code with PKCE (`S256`).
- The access token exists only in JavaScript memory. It is not written to local
  storage, session storage, cookies, or the URL. Refreshing signs the user out.
- PKCE state and verifier values are short-lived in session storage and are
  deleted before code exchange.
- The dashboard never calculates a hidden cost. It displays the analysis
  contract's `display` and release state, so withheld figures stay withheld.
- The API—not this interface—enforces organization membership and RBAC.

The identity provider must allow a public browser client, require PKCE, permit
the configured redirect URI, and allow cross-origin token exchange from the
dashboard origin. The Nginx content-security policy must allow exactly that
identity provider origin through `DASHBOARD_CONNECT_SRC`, for example:

```text
'self' https://identity.example.com
```

Do not use a wildcard. The default is `'self'`, so external token exchange is
blocked until the operator chooses the identity origin.

## Local development

The Compose stack serves the dashboard at `http://127.0.0.1:8080`. Development
token entry is enabled there only to make testing possible before an identity
provider is selected. Production settings reject that mode.

The dashboard reverse-proxies `/api/` to the API container, keeping application
requests same-origin and avoiding a broad API CORS policy.

Basic asset checks need no JavaScript packages:

```bash
node --check apps/dashboard/app.js
python -m pytest -q apps/dashboard/tests
```
