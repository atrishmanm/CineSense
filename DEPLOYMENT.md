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

### Option 1: **Render.com** (⭐ Best - Free Web Hosting)

**Pros**: Free web service (no card needed), automatic deploys  
**Note**: Render's managed MySQL requires payment. Use free external database.

#### Quick Deploy Steps:

1. **Get Free Database** (Choose one):
   
   **A) Clever Cloud (FREE MySQL - Recommended)**
   - Go to https://clever-cloud.com
   - Sign up → Create MySQL addon (Free tier: 256MB)
   - Copy: Host, Port, Database, User, Password

   **B) Railway MySQL (FREE $5 credit)**
   - Go to https://railway.app → New Project → MySQL
   - Copy connection details

   **C) FreeSQLDatabase.com**
   - Go to https://freesqldatabase.com
   - Create free MySQL database
   - Copy credentials

2. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add deployment configs"
   git push origin main
   ```

3. **Deploy on Render**: https://render.com
   - Sign up with GitHub (NO CARD REQUIRED)
   - Click "New +" → "Web Service"FREE $5 Credit)

**Pros**: One command deploy, free MySQL included, auto-setup  
**Setup Time**: 5 minutes  
**Cost**: FREE $5 credit (lasts ~1 month), then $5/month

#### Steps:

1. Install Railway CLI:
   ```bash
   npm i -g @railway/cli
   ```

2. Deploy:
   ```bash
   railway login
   railway init
   railway up
   ```

3. Add MySQL:
   ```bash
   railway add mysql
   ```

4. Done! Railway handles everything automatically.

**Note**: $5 free credit on signup - no card needed initially!
**Pros**: One command deploy, auto MySQL  
**Setup Time**: 5 minutes

#### Steps:

1. Install Railway CLI:
   ```bash
   npm i -g @railway/cli
   ```

2. Deploy:
   ```bash
   railway login
   railway init
   railway up
   ```

3. Add MySQL:
   ```bash
   railway add mysql
   ```

4. Done! Railway handles everything automatically.

---

### Option 3: **Heroku** (Classic)

**Pros**: Mature platform, great docs  
**Cost**: ~$7/month for database

#### Steps:

1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli

2. Deploy:
   ```bash
   heroku login
   heroku create cinesense-app
   heroku addons:create jawsdb:kitefin
   git push heroku main
   ```

3. Set config:
   ```bash
   heroku config:set TMDB_API_KEY=your_key
   ```

4. Initialize:
   ```bash
   heroku run python scripts/fetch_tmdb_data.py --count 1000
   ```
⭐ 100% FREE FOREVER)

**Pros**: Free MySQL included, no credit card ever, Python-optimized  
**Setup Time**: 15 minutes  
**Best for**: Academic projects, portfolios, demowhere** (Free Forever)

**Pros**: Free MySQL included, no credit card  
**Setup Time**: 15 minutes

#### Steps:

1. Sign up: https://www.pythonanywhere.com (Free account)

2. Upload code:
   - Bash console: `git clone https://github.com/YOUR_USERNAME/CineSense.git`

3. Create virtualenv:
   ```bash
   mkvirtualenv cinesense --python=python3.10
   cd CineSense
   pip install -r requirements.txt
   ```

4. Setup MySQL:
   - "Databases" tab → Create database `cinesense`
   - Note hostname and credentials

5. Configure web app:
   - "Web" tab → "Add a new web app"
   - Manual configuration → Python 3.10
   - Set source code: `/home/YOUR_USERNAME/CineSense`
   - Set virtualenv: `/home/YOUR_USERNAME/.virtualenvs/cinesense`

6. Edit WSGI file:
   ```python
   import sys
   path = '/home/YOUR_USERNAME/CineSense'
   if path not in sys.path:
       sys.path.append(path)
   
   from app import app as application
   ```

7. Set environment variables in WSGI file or .env

8. Reload web app

---

## 🐳 Docker Deployment (Universal)

Deploy anywhere with Docker:

```bash
# Build and run locally
docker-compose up

# Or deploy to any cloud with Docker support
docker build -t cinesense .
docker run -p 5000:5000 cinesense
```

**Works on**: DigitalOcean, Linode, AWS ECS, Azure Container Apps, GCP Cloud Run

---Cost | Why |
|----------|----------|------|-----|
| **Quick Demo** | Railway | FREE $5 credit | Easiest setup, one command |
| **Academic Project** | PythonAnywhere | FREE forever | No card, MySQL included |
| **Professional Portfolio** | Render + Clever Cloud | FREE | Looks professional, fast |
| **Production App** | AWS/Azure | Paid | Scalable, enterprise-ready |
| **Maximum Control** | Docker + VPS | Varies | Full customization |

### 💯 Completely FREE Options (No Card Required):
1. **PythonAnywhere** - Best for students (MySQL included)
2. **Render.com** - Best for demos (fast, but need external DB)
3. **Railway** - $5 free credit (easiest, lasts ~1 month)
| **Academic Project** | PythonAnywhere | Free forever, no card needed |
| **Professional Portfolio** | Railway | Best UX, worth $5 |
| **Production App** | AWS/Azure | Scalable, enterprise-ready |
| **Maximum Control** | Docker + VPS | Full customization |

---

## 📋 Pre-Deployment Checklist

- [ ] TMDB API key ready
- [ ] Code pushed to GitHub
- [ ] `.env.example` has all required variables
- [ ] `requirements.txt` includes `gunicorn`
- [ ] Database schema ready (`database/schema.sql`)
- [ ] Choose platform from above

---

## 🔧 Quick Fixes

**"Module not found"**  
→ Add to `requirements.txt`, redeploy

**"Database connection failed"**  
→ Check environment variables match database credentials

**"502 Bad Gateway"**  
→ App crashed. Check logs for Python errors

**"Out of memory"**  
→ Reduce worker count in Procfile: `gunicorn app:app --workers 2`

---

## 🚀 Next Steps After Deployment

1. Test signup/login flow
2. Run comparison 5-10 times
3. Verify recommendations appear
4. Share your live link!

**Need help?** Open an issue on GitHub with your platform and error logs.

---

**Recommended: Start with Render.com** - It's free, fast, and designed for apps like CineSense! 🎬
