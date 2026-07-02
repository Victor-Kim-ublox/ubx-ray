# Public access via Cloudflare Tunnel + Access

How the internal-only ubx-ray service (`ubxray` on `localhost:8000`) is exposed
to the internet at **https://ubx-ray.com** for free, with HTTPS and an email
login gate — **no router port-forwarding, no public IP, no paid plan**.

Two pieces:

1. **Cloudflare Tunnel** (`cloudflared`) — an outbound-only tunnel from this
   host to Cloudflare's edge; Cloudflare serves `ubx-ray.com` over HTTPS and
   forwards requests down the tunnel to `localhost:8000`.
2. **Cloudflare Access** (Zero Trust) — an identity gate in front of the app so
   only allowed people (here: `@u-blox.com` emails, via one-time email code)
   can reach it.

Both `cloudflared` and the app run as auto-start Windows services, so the site
survives reboots.

---

## Prerequisites (already done)

- A Cloudflare account with the domain **`ubx-ray.com`** added (its
  nameservers delegated to Cloudflare).
- The `ubxray` app service running on `localhost:8000` (see
  [`install_service.md`](install_service.md)).

---

## 1. Install cloudflared

```powershell
winget install --id Cloudflare.cloudflared
```

Installed to `C:\Program Files (x86)\cloudflared\cloudflared.exe`.

## 2. Authenticate + create the tunnel

```powershell
$cf = "C:\Program Files (x86)\cloudflared\cloudflared.exe"

# Opens a browser; log in and authorize the ubx-ray.com zone.
# Writes cert.pem to %USERPROFILE%\.cloudflared\
& $cf tunnel login

# Create a named tunnel (writes a <tunnel-id>.json credentials file)
& $cf tunnel create ubxray

# Route the hostnames to the tunnel (creates proxied CNAMEs in the zone)
& $cf tunnel route dns ubxray ubx-ray.com
& $cf tunnel route dns ubxray www.ubx-ray.com
```

The current tunnel id is `fc3f1b96-dd81-4d6d-8c46-cd2e94daf3e2`; its credentials
live at `C:\Users\kim45\.cloudflared\fc3f1b96-...json` (**keep secret** —
deleting the tunnel revokes them).

## 3. Config file

`C:\Users\kim45\.cloudflared\config.yml`:

```yaml
tunnel: fc3f1b96-dd81-4d6d-8c46-cd2e94daf3e2
credentials-file: C:\Users\kim45\.cloudflared\fc3f1b96-dd81-4d6d-8c46-cd2e94daf3e2.json

ingress:
  - hostname: ubx-ray.com
    service: http://localhost:8000
  - hostname: www.ubx-ray.com
    service: http://localhost:8000
  - service: http_status:404
```

Validate: `cloudflared tunnel --config <path> ingress validate` → `OK`.

## 4. Run as a service — via NSSM (not `cloudflared service install`)

> **Why NSSM:** on this host `cloudflared service install` registered the
> service with only the bare exe (no `--config`, no `run`), so it started as
> `LocalSystem`, could not find the config in the system profile, and
> crash-looped (Event Log showed only "Cloudflared service starting" with
> `arguments: [cloudflared.exe]`, and `https://ubx-ray.com` returned 530).
> Registering it under **NSSM** with an explicit `--config ... run ubxray`
> command line is reliable and mirrors the existing `ubxray` service.

Run from an **Administrator** PowerShell:

```powershell
$nssm = "C:\ClaudeWorkspace\ubx-ray\tools\nssm\nssm.exe"
$cf   = "C:\Program Files (x86)\cloudflared\cloudflared.exe"

# Remove any broken native cloudflared service first
& $cf service uninstall 2>$null
sc.exe delete Cloudflared 2>$null

# Register cloudflared under NSSM with an explicit config + tunnel
& $nssm install cloudflared-ubxray $cf "tunnel --config C:\Users\kim45\.cloudflared\config.yml run ubxray"
& $nssm set cloudflared-ubxray Start SERVICE_AUTO_START
& $nssm set cloudflared-ubxray AppStdout "C:\ClaudeWorkspace\ubx-ray\logs\cloudflared.out.log"
& $nssm set cloudflared-ubxray AppStderr "C:\ClaudeWorkspace\ubx-ray\logs\cloudflared.err.log"
& $nssm start cloudflared-ubxray
```

Success looks like four `Registered tunnel connection` lines in
`logs\cloudflared.err.log`. Both services should read `Running` / `Automatic`:

```powershell
Get-Service ubxray, cloudflared-ubxray | Select-Object Name, Status, StartType
```

---

## 5. Cloudflare Access (identity gate)

Configured in the **Zero Trust** dashboard (<https://one.dash.cloudflare.com>).
Team domain: `hidden-bird-1d47.cloudflareaccess.com`.

1. **Access controls → Applications → Add an application → Self-hosted**
   - Public hostnames: `ubx-ray.com` and `www.ubx-ray.com`
   - Application name: `ubx-ray`
2. **Policy**
   - Name: `u-blox employees`, Action: `Allow`
   - Include → **Emails ending in** `@u-blox.com`
     (use **Emails** + a specific address to restrict to individuals)
3. **Login methods** (Access controls → **Identity provider integrations**)
   - Ensure **One-time PIN** is present. This is what lets visitors log in with
     just an email + a 6-digit code **without needing a Cloudflare account**.
     Without it, the only option is "Login with Cloudflare" (which *does*
     require an account).
   - The app's Authentication tab has *Accept all available identity providers*
     **ON**, so One-time PIN is offered automatically. To show *only* the email
     code option, turn that off and select only **One-time PIN**.

Login flow for a visitor: `ubx-ray.com` → Cloudflare Access page → enter email
→ enter the emailed code → app. Only `@u-blox.com` emails pass the policy.

---

## Verification

```powershell
# Should 302-redirect to <team>.cloudflareaccess.com/... (gate active),
# not serve the app directly:
curl.exe -sS -I https://ubx-ray.com | Select-String '^HTTP/|^location:'

# Following redirects lands on "Sign in - Cloudflare Access"
curl.exe -sSL https://ubx-ray.com | Select-String 'Cloudflare Access|Log in to ubx-ray'
```

---

## Operations

```powershell
# Tunnel service
Get-Service cloudflared-ubxray
Restart-Service cloudflared-ubxray            # admin
Get-Content C:\ClaudeWorkspace\ubx-ray\logs\cloudflared.err.log -Tail 40 -Wait

# App service (see install_service.md)
Restart-Service ubxray                        # admin, after code changes
```

- **Change who can access:** Zero Trust → Access controls → Applications →
  `ubx-ray` → Policies.
- **After a reboot:** confirm both `ubxray` and `cloudflared-ubxray` are
  Running.
- **Corporate-account caveat:** the domain lives under a `u-blox.com` Cloudflare
  account, so some Zero Trust / IdP settings may be governed by org policy.
