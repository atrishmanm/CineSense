# 🚀 Deployment Guide - CineSense

Multiple deployment options for your AI movie recommendation platform.

---

## ❌ Why Not Vercel?

Vercel is designed for **serverless** and **static sites**. CineSense requires:
- Long-running Flask server ❌
- MySQL database ❌  
- Persistent storage ❌
- Background jobs ❌

**Vercel won't work without major restructuring.**

---

## ✅ Recommended Platforms

### Option 1: **Render.com** (⭐ Best - Free Tier)

**Pros**: Free MySQL, automatic deploys, easy setup  
**Setup Time**: 10 minutes

#### Steps:

1. **Push to GitHub** (if not done):
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/CineSense.git
   git branch -M main
   git push -u origin main
   ```

2. **Go to Render**: https://render.com (Sign up with GitHub)

3. **Create Database**:
   - Click "New +" → "PostgreSQL" or use external MySQL
   - Name: `cinesense-db`
   - Free tier selected
   - Click "Create Database"
   - Copy connection details

4. **Create Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repo
   - Name: `cinesense`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Add environment variables:
     - `TMDB_API_KEY` = your_api_key
     - `DB_HOST` = from step 3
     - `DB_PORT` = 3306
     - `DB_USER` = from step 3
     - `DB_PASSWORD` = from step 3
     - `DB_NAME` = cinesense

5. **Initialize Database**:
   - Use Render shell to run: `python -c "from database.db_manager import DatabaseManager; db = DatabaseManager(); print('DB Ready')"`
   - Run data fetch: `python scripts/fetch_tmdb_data.py --count 1000`

6. **Access**: Your app at `https://cinesense-xxxxx.onrender.com`

---

### Option 2: **Railway.app** (Very Easy)

**Pros**: One-click MySQL, great DX  
**Cost**: $5/month after free trial

#### Steps:

1. Go to https://railway.app
2. "New Project" → "Deploy from GitHub"
3. Select CineSense repo
4. Add MySQL service: "+ New" → "Database" → "MySQL"
5. Railway auto-detects Python and deploys
6. Add env vars in project settings
7. Done! 🎉

---

### Option 3: **PythonAnywhere** (Free, Python-Specific)

**Pros**: Free MySQL included, Python-friendly  
**Setup Time**: 15 minutes

#### Steps:

1. Sign up: https://www.pythonanywhere.com
2. Upload code via Git or Files
3. Create MySQL database in "Databases" tab
4. Configure web app in "Web" tab
5. Set WSGI file to point to `app.py`
6. Add environment variables

---

### Option 4: **AWS / Azure / GCP** (Production-Grade)

**For serious deployment**:

**AWS**:
- EC2 for Flask app
- RDS for MySQL
- S3 for static assets
- CloudFront CDN

**Azure**:
- App Service for Flask
- Azure Database for MySQL
- Blob Storage for assets

**GCP**:
- Cloud Run for containers
- Cloud SQL for MySQL
- Cloud Storage for assets

---

## 🐳 Docker Deployment (Any Platform)

<function_calls>
<invoke name="create_file">
<parameter name="content">FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]
