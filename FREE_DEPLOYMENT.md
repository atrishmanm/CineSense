# 🆓 100% FREE Deployment Guide

Deploy CineSense **completely free** with no credit card required.

---

## 🥇 BEST: PythonAnywhere (100% Free Forever)

**✅ Includes**: Free MySQL, Free hosting, No credit card  
**⏱️ Setup**: 15 minutes  
**Perfect for**: Academic projects, portfolios

### Step-by-Step:

#### 1. Sign Up (No Card)
- Go to https://www.pythonanywhere.com
- Click "Start running Python online in less than a minute"
- Create free account (no payment info needed)

#### 2. Upload Code
Open a **Bash console**:
```bash
git clone https://github.com/YOUR_USERNAME/CineSense.git
cd CineSense
```

#### 3. Create Virtual Environment
```bash
mkvirtualenv cinesense --python=python3.10
pip install -r requirements.txt
```

#### 4. Setup MySQL Database
- Go to **"Databases"** tab
- MySQL password: Create one (remember it!)
- Database name: Create `YOUR_USERNAME$cinesense`
- Note your database hostname: `YOUR_USERNAME.mysql.pythonanywhere-services.com`

#### 5. Initialize Database Schema
In Bash console:
```bash
mysql -u YOUR_USERNAME -h YOUR_USERNAME.mysql.pythonanywhere-services.com -p YOUR_USERNAME$cinesense < database/schema.sql
# Enter your MySQL password when prompted
```

#### 6. Create .env File
```bash
cd ~/CineSense
nano .env
```

Add:
```env
TMDB_API_KEY=your_tmdb_api_key_here
DB_HOST=YOUR_USERNAME.mysql.pythonanywhere-services.com
DB_PORT=3306
DB_NAME=YOUR_USERNAME$cinesense
DB_USER=YOUR_USERNAME
DB_PASSWORD=your_mysql_password_here
```

Save: `Ctrl+X`, `Y`, `Enter`

#### 7. Fetch Movie Data
```bash
workon cinesense
cd ~/CineSense
python scripts/fetch_tmdb_data.py --count 500
```
(Takes 5-10 minutes)

#### 8. Configure Web App
- Go to **"Web"** tab
- Click **"Add a new web app"**
- Domain: `YOUR_USERNAME.pythonanywhere.com` (free subdomain)
- Manual configuration → Python 3.10
- Click through defaults

#### 9. Set Paths
On Web tab, set:
- **Source code**: `/home/YOUR_USERNAME/CineSense`
- **Working directory**: `/home/YOUR_USERNAME/CineSense`
- **Virtualenv**: `/home/YOUR_USERNAME/.virtualenvs/cinesense`

#### 10. Configure WSGI File
Click on WSGI configuration file link, delete all content and replace with:

```python
import sys
import os

# Add your project directory
path = '/home/YOUR_USERNAME/CineSense'
if path not in sys.path:
    sys.path.insert(0, path)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(path, '.env'))

# Import Flask app
from app import app as application
```

Replace `YOUR_USERNAME` with your actual username!

#### 11. Reload and Launch
- Click green **"Reload"** button
- Visit: `https://YOUR_USERNAME.pythonanywhere.com`

### ✅ Done! Your app is live 24/7 for FREE! 🎉

---

## 🥈 Alternative: Render + Free External Database

**✅ Includes**: Fast hosting, auto-deploy from GitHub  
**⏱️ Setup**: 10 minutes  
**Limitation**: Need separate free database

### Quick Steps:

#### 1. Get Free MySQL Database

**Option A: Clever Cloud (Recommended)**
- Go to https://www.clever-cloud.com
- Sign up (no card)
- Create add-on → MySQL → Dev plan (FREE)
- Copy connection details

**Option B: FreeSQLDatabase.com**
- Go to https://www.freesqldatabase.com
- Get free database (5MB limit)
- Copy credentials

**Option C: db4free.net**
- Go to https://www.db4free.net
- Register database (200MB limit)
- Copy credentials

#### 2. Initialize Database
Use any MySQL client or command line:
```bash
mysql -h YOUR_DB_HOST -P 3306 -u YOUR_DB_USER -p YOUR_DB_NAME < database/schema.sql
```

#### 3. Deploy to Render
- Go to https://render.com (sign up with GitHub - no card)
- New → Web Service
- Connect your CineSense repo
- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app --workers 2`
- Plan: **FREE**

#### 4. Add Environment Variables
In Render dashboard:
- `TMDB_API_KEY` = your_key
- `DB_HOST` = from step 1
- `DB_PORT` = 3306
- `DB_NAME` = from step 1
- `DB_USER` = from step 1
- `DB_PASSWORD` = from step 1

#### 5. Fetch Data
Once deployed, use Render Shell:
```bash
python scripts/fetch_tmdb_data.py --count 500
```

### ✅ Live at: `https://cinesense-xxxxx.onrender.com`

---

## 🥉 Free Credits: Railway

**🎁 $5 free credit** (lasts ~1 month for this app)

### Ultra-Quick Deploy:
```bash
# Install Railway CLI
npm i -g @railway/cli

# Deploy (one command!)
railway login
railway init
railway up

# Add MySQL
railway add mysql
```

**Done!** Railway auto-configures everything.

**After credit runs out**: $5-10/month (optional to continue)

---

## 📊 Free Tier Comparison

| Platform | Hosting | Database | Card Required | Best For |
|----------|---------|----------|---------------|----------|
| **PythonAnywhere** | ✅ Free | ✅ Free MySQL | ❌ No | Students, long-term |
| **Render.com** | ✅ Free | ❌ Need external | ❌ No | Fast demos |
| **Railway** | 🎁 $5 credit | 🎁 $5 credit | ❌ No (initially) | Easiest setup |
| **Clever Cloud** | ❌ Paid | ✅ Free MySQL | ❌ No | Just database |

---

## 🎯 Recommendation

### For Academic/Student Project:
→ **Use PythonAnywhere** (100% free, MySQL included, forever)

### For Quick Demo:
→ **Use Railway** (easiest, $5 free credit, 5 minutes)

### For Professional Portfolio:
→ **Use Render + Clever Cloud** (both free, looks professional)

---

## 🆘 Troubleshooting

**PythonAnywhere: "Bash console not responding"**
→ Reload page, open new console

**Render: "Application failed to start"**
→ Check logs, ensure all env vars set correctly

**Database: "Access denied"**
→ Check DB_HOST, DB_USER, DB_PASSWORD match credentials exactly

**TMDB: "Invalid API key"**
→ Get new key from https://www.themoviedb.org/settings/api

---

## 🎓 For Your Project Submission

**Deployed URL**: `https://YOUR_USERNAME.pythonanywhere.com`  
**Cost**: $0  
**Uptime**: 24/7  
**Database**: MySQL (free tier)  
**Hosting**: PythonAnywhere (free tier)

**Perfect for BTech/MTech project demonstrations!** ✨

---

**Need help?** Check the main [DEPLOYMENT.md](DEPLOYMENT.md) for more options.
