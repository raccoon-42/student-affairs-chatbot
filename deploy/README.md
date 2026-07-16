# Deploying İyteBot on a DigitalOcean droplet

One droplet runs the whole compose stack (api + qdrant + daily corpus
check) behind host nginx with Let's Encrypt TLS.

## 1. Droplet

- Ubuntu 24.04 LTS, Basic / Regular, **2 GB RAM / 1 vCPU / 50 GB disk**
  (~$12/mo). 1 GB is not enough: the image build pulls torch and needs
  the headroom. Region: Frankfurt (closest to TR).
- Add your SSH key at creation.

```bash
ssh root@<droplet-ip>

# basics + firewall
apt update && apt upgrade -y
ufw allow OpenSSH && ufw allow 80 && ufw allow 443 && ufw enable

# docker (official convenience script installs compose v2 too)
curl -fsSL https://get.docker.com | sh
```

## 2. App

```bash
git clone https://github.com/raccoon-42/rag-chatbot.git /opt/iytebot
cd /opt/iytebot
```

Create `/opt/iytebot/.env` — start from your local one, then change:

- `ABUSE_EXEMPT=` and `RATELIMIT_EXEMPT=` — **empty**, no dev IPs in prod
- `COOKIE_SECURE=1` — auth cookie only over HTTPS
- keep: `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `GOOGLE_CLIENT_ID`,
  `EMBEDDING_BACKEND=openrouter` etc.
- In Google Cloud Console → the OAuth client → add
  `https://<your-domain>` to Authorized JavaScript origins, or sign-in
  will fail in prod.

```bash
docker compose -f docker-compose.prod.yml up -d --build   # ~10 min first time
```

## 3. Indexing happens by itself

Qdrant starts empty. The processed corpora are in git; on first start the
check-updates service finds no recorded index state and runs a baseline
check, indexing everything (one-time full embed via OpenRouter, cost is
small) and re-downloading the gitignored raw PDFs by itself:

```bash
docker compose -f docker-compose.prod.yml logs -f check-updates
```

Every corpus should end with a recorded baseline. From then on the
service re-checks daily at 09:30 TR.

## 4. Domain + nginx + TLS

Point an A record for your domain at the droplet IP, then:

```bash
apt install -y nginx certbot python3-certbot-nginx
cp deploy/nginx.conf /etc/nginx/sites-available/iytebot
# edit server_name in that file to your domain
ln -s /etc/nginx/sites-available/iytebot /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx
certbot --nginx -d <your-domain>
```

## 5. Verify

- `https://<domain>` loads the UI; ask something; citation chips render.
- Google sign-in works (needs the origin registered, step 2).
- `docker compose -f docker-compose.prod.yml logs -f check-updates`
  says "next corpus check in Ns".
- Mic + clipboard buttons work (they require HTTPS).

## Updating the app later

```bash
cd /opt/iytebot && git pull
docker compose -f docker-compose.prod.yml up -d --build
```

SQLite (`data/`), corpora and index state (`preprocessing/data/`) and the
qdrant volume all survive rebuilds — they live outside the image.

## Backup

`data/app.db` is the only irreplaceable state (users + conversations);
corpora and the qdrant index can always be rebuilt from source. A daily
`sqlite3 data/app.db ".backup ..."` to somewhere off-droplet is enough.
