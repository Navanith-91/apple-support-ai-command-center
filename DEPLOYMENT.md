# Deployment Guide: AI Customer Support Command Center

This guide covers step-by-step instructions for deploying the **AI Customer Support Command Center** across multiple environments, from 1-click free cloud hosting to containerized production deployments.

---

## 📋 Deployment Options Overview

| Platform | Best For | Effort | Cost | URL Format |
| :--- | :--- | :---: | :---: | :--- |
| **1. Streamlit Community Cloud** | Quick demos, portfolios, interviewers | 🟢 2 mins | Free | `https://<your-app>.streamlit.app` |
| **2. Docker / Cloud Containers** | Production, AWS / GCP / Azure | 🟡 5 mins | Low | Custom domain or Cloud Run URL |
| **3. Hugging Face Spaces** | ML community showcase | 🟢 3 mins | Free | `https://huggingface.co/spaces/<user>/<app>` |
| **4. Render / Railway** | Managed cloud hosting with custom domain | 🟢 3 mins | Free/Low | `https://<app>.onrender.com` |
| **5. Linux VPS (Nginx + SSL)** | Self-hosted enterprise server | 🟠 15 mins | VPS ($5/mo) | `https://support-ai.yourdomain.com` |

---

## Option 1: Streamlit Community Cloud (Recommended & Easiest)

Streamlit Community Cloud is 100% free, automatically builds from your GitHub repository, and provides an instant public HTTPS URL.

### Step 1: Push Code to GitHub
```bash
# Initialize git if not already initialized
git init
git add .
git commit -m "feat: AI Customer Support Command Center with full backend and evaluations"

# Create a new repository on GitHub (public or private) and push:
git remote add origin https://github.com/<YOUR_USERNAME>/customer-support-ai.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `<YOUR_USERNAME>/customer-support-ai`.
4. Set the **Main file path** to:
   ```text
   dashboard/app.py
   ```
5. Click **"Advanced settings"** (optional):
   - Python Version: `3.10`
   - Secrets: (Optional) Add your `GEMINI_API_KEY` or `OPENAI_API_KEY` if using external LLM providers.
6. Click **"Deploy!"**.
7. In ~2 minutes, your dashboard will be live at `https://<your-app>.streamlit.app`.

---

## Option 2: Docker Container Deployment

The repository includes a production-ready `Dockerfile` and `docker-compose.yml` with pre-cached embeddings and healthchecks.

### Local Docker Run
```bash
# Build the Docker image
docker build -t apple-support-command-center .

# Run the container on port 8501
docker run -d -p 8501:8501 --name support-agent apple-support-command-center

# Access the app:
http://localhost:8501
```

### Docker Compose Run
```bash
# Single-command build and launch
docker-compose up -d

# Check health and logs
docker-compose ps
docker-compose logs -f
```

---

## Option 3: Google Cloud Run (Serverless Container)

GCP Cloud Run provides automatic scaling (scales to 0 when idle to save costs) and HTTPS termination.

```bash
# 1. Authenticate with Google Cloud
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID

# 2. Build and submit container image
gcloud builds submit --tag gcr.io/YOUR_GCP_PROJECT_ID/support-command-center

# 3. Deploy to Cloud Run (allow unauthenticated access for public dashboard)
gcloud run deploy support-command-center \
    --image gcr.io/YOUR_GCP_PROJECT_ID/support-command-center \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8501 \
    --memory 2Gi \
    --cpu 2
```

---

## Option 4: Render / Railway PaaS

### Deploy on Render:
1. Connect your GitHub repository on [render.com](https://render.com).
2. Create a new **Web Service**.
3. Settings:
   - **Environment**: `Python 3` (or `Docker`)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run dashboard/app.py --server.port=$PORT --server.address=0.0.0.0`
4. Click **Create Web Service**.

### Deploy on Railway:
1. Install Railway CLI: `npm i -g @railway/cli`
2. Run `railway login`
3. Run `railway init` and select Dockerfile or Python
4. Run `railway up`

---

## Option 5: Self-Hosted Linux VPS (Ubuntu + Systemd + Nginx + SSL)

For hosting on an AWS EC2, DigitalOcean Droplet, or Linode server:

### 1. Install System Dependencies
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv nginx certbot python3-certbot-nginx git
```

### 2. Clone & Set Up Project
```bash
git clone https://github.com/<YOUR_USERNAME>/customer-support-ai.git /var/www/customer-support-ai
cd /var/www/customer-support-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Create Systemd Service
Create `/etc/systemd/system/support-ai.service`:
```ini
[Unit]
Description=AppleSupport AI Command Center Dashboard
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/var/www/customer-support-ai
ExecStart=/var/www/customer-support-ai/.venv/bin/streamlit run dashboard/app.py --server.port=8501 --server.address=127.0.0.1 --server.headless=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable support-ai
sudo systemctl start support-ai
```

### 4. Configure Nginx Reverse Proxy
Create `/etc/nginx/sites-available/support-ai`:
```nginx
server {
    listen 80;
    server_name support-ai.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
    }
}
```

Enable site and acquire free SSL certificate:
```bash
sudo ln -s /etc/nginx/sites-available/support-ai /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx
sudo certbot --nginx -d support-ai.yourdomain.com
```

---

## 🔒 Environment Variables (Optional)

If integrating external live LLMs (Gemini / OpenAI), configure the following environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | Google Gemini API Key for generation / LLM Judge | *Optional (offline templates used by default)* |
| `OPENAI_API_KEY` | OpenAI API Key for GPT-4 fallback | *Optional* |
| `STREAMLIT_SERVER_PORT` | Port for the web dashboard | `8501` |
| `STREAMLIT_SERVER_HEADLESS` | Headless execution mode | `true` |
