# 🚀 Deploy CineSense to Render (100% FREE)

Complete guide to deploy your AI movie recommendation platform on Render's free tier.

## ⚡ Quick Overview

**Total Cost**: $0/month  
**Setup Time**: 15 minutes  
**Requirements**: GitHub account + Render account

---

## 📋 Prerequisites

### 1. Get Your TMDB API Key
1. Go to https://www.themoviedb.org/signup
2. Verify email and login
3. Go to Settings → API → Create API Key
4. Copy your API key (you'll need this later)

### 2. Get a Free MySQL Database

**Option A: Aiven (Recommended - 100% Free Forever)**
1. Sign up at https://aiven.io
2. Create "MySQL" service
3. Select "Free Plan" (no credit card needed)
4. Copy connection details:
   - Host
   - Port (usually 3306)
   - Username
   - Password
   - Database name

**Option B: FreeSQLDatabase.com**
1. Go to https://www.freesqldatabase.com
2. Fill the form and create database
3. Check email for credentials

---

## 🔧 Step-by-Step Deployment

### Step 1: Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial CineSense deployment"

# Create GitHub repo and push
git remote add origin https://github.com/YOUR_USERNAME/CineSense.git
git branch -M main
git push -u origin main
```

### Step 2: Create Render Account

1. Go to https://render.com
2. Click "Get Started for Free"
3. Sign up with your GitHub account (recommended)

### Step 3: Deploy Web Service

1. Click "New +" → "Web Service"
2. Connect your GitHub repository
3. Select the CineSense repository
4. Configure the service:

   **Name**: `cinesense` (or your choice)  
   **Region**: Choose closest to you  
   **Branch**: `main`  
   **Runtime**: `Python 3`  
   **Build Command**: `pip install -r requirements.txt`  
   **Start Command**: `gunicorn app:app`  
   **Plan**: **Free** (important!)

5. Click "Advanced" and add environment variables:

   | Key | Value | Example |
   |-----|-------|---------|
   | `TMDB_API_KEY` | Your TMDB API key | `a1b2c3d4e5f6...` |
   | `DB_HOST` | MySQL host from Step 2 | `mysql-abc.aivencloud.com` |
   | `DB_PORT` | MySQL port | `3306` |
   | `DB_USER` | MySQL username | `avnadmin` |
   | `DB_PASSWORD` | MySQL password | `AVNS_...` |
   | `DB_NAME` | Database name | `defaultdb` |
   | `SECRET_KEY` | Random secret string | `your-random-secret-key-123` |
   | `FLASK_ENV` | Environment | `production` |

6. Click "Create Web Service"

### Step 4: Initialize Database

Once deployed, Render will give you a URL like: `https://cinesense-xxxx.onrender.com`

1. In Render dashboard, go to your web service
2. Click "Shell" tab (opens terminal)
3. Run database initialization:

```bash
# Test database connection
python -c "from database.db_manager import DatabaseManager; db = DatabaseManager(); print('✓ Database Connected')"

# Fetch initial movie data (this will take 5-10 minutes)
python scripts/fetch_tmdb_data.py --count 500
```

### Step 5: Test Your Deployment

Visit your Render URL: `https://cinesense-xxxx.onrender.com`

You should see the CineSense homepage with movies loaded!

---

## 🎯 Important Notes

### Free Tier Limitations

- **Sleep Mode**: Service goes to sleep after 15 minutes of inactivity
- **First Request**: May take 30-50 seconds to wake up
- **750 Hours/Month**: Service shuts down after monthly limit (usually enough for personal use)
- **No Credit Card Required**: 100% free, no hidden charges

### Keeping Your App Awake (Optional)

If you want to prevent sleep mode, use a free uptime monitor:

1. Go to https://uptimerobot.com
2. Create free account
3. Add monitor:
   - Type: HTTP(s)
   - URL: Your Render app URL
   - Interval: 5 minutes

### Database Size Limit

- Aiven free tier: 1GB storage (plenty for 50,000+ movies)
- Monitor usage in Aiven dashboard

---

## 🔒 Security Checklist

✅ Never commit `.env` file to Git  
✅ Use strong `SECRET_KEY` in environment variables  
✅ Keep `FLASK_ENV=production` on Render  
✅ Regularly rotate API keys  
✅ Enable GitHub branch protection

---

## 🐛 Troubleshooting

### Build Failed
- Check `requirements.txt` is present
- Verify Python version in `runtime.txt` is 3.10.0

### Database Connection Error
- Verify all DB environment variables are correct
- Check database is running in Aiven
- Test connection using Render shell

### App Not Loading
- Check Render logs (Logs tab in dashboard)
- Ensure `gunicorn` is in `requirements.txt`
- Verify `app.py` exists in root directory

### Movies Not Showing
- Run `fetch_tmdb_data.py` in Render shell
- Check TMDB API key is valid
- Verify database has data: `SELECT COUNT(*) FROM movies`

---

## 📊 Post-Deployment Tasks

### 1. Custom Domain (Optional, Free)
1. In Render dashboard → Settings → Custom Domain
2. Add your domain (e.g., `cinesense.yourdomain.com`)
3. Update DNS records as instructed

### 2. Enable HTTPS
- Render provides free SSL certificates automatically
- Your app is HTTPS by default: `https://cinesense-xxxx.onrender.com`

### 3. Monitor Performance
- Check Render dashboard for metrics
- View logs for errors
- Set up email alerts in Render settings

---

## 🎉 Success!

Your CineSense platform is now live and accessible worldwide at:

**`https://cinesense-xxxx.onrender.com`**

Share it with friends and start getting movie recommendations! 🎬🍿

---

## 📞 Need Help?

- **Render Docs**: https://render.com/docs
- **TMDB API Docs**: https://developers.themoviedb.org
- **Aiven Docs**: https://docs.aiven.io

---

## 💡 Next Steps

1. **Add More Movies**: Run fetch script with `--count 1000` or higher
2. **Customize UI**: Edit templates and redeploy (auto-deploy on git push)
3. **Share Your App**: Get feedback from users
4. **Monitor Usage**: Check Render analytics

Enjoy your free, production-ready movie recommendation platform! 🚀
