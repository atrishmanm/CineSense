# 🚀 CineSense - Quick Start (5 Minutes)

Get CineSense running in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.10+ installed
- [ ] MySQL 8.0+ installed and running
- [ ] TMDB API key (get free at https://www.themoviedb.org/settings/api)

## Quick Setup

### 1. Install Dependencies (1 min)

```bash
pip install -r requirements.txt
```

### 2. Configure Environment (1 min)

Copy `.env.example` to `.env` and edit:

```env
TMDB_API_KEY=your_api_key_here
DB_PASSWORD=your_mysql_password
```

### 3. Setup Database (1 min)

```bash
# Create database
mysql -u root -p -e "CREATE DATABASE cinesense"

# Run schema
mysql -u root -p cinesense < database/schema.sql
```

### 4. Fetch Movie Data (2 min for test mode)

```bash
# Quick test with 50 movies
python scripts/fetch_tmdb_data.py --test
```

**Note**: For full demo, use `--count 3000` (takes 20-30 minutes)

### 5. Run Application

```bash
python app.py
```

Open: **http://localhost:5000**

---

## Test Flow

1. **Sign up** → Create account
2. **Compare** → Choose between 5-10 movie pairs
3. **Home** → See personalized recommendations!

---

## Common Issues

**"TMDB API key not found"**
→ Add your key to `.env` file

**"Database connection failed"**
→ Check MySQL is running & password is correct

**"No movies found"**
→ Run: `python scripts/fetch_tmdb_data.py --test`

---

## Files Overview

```
CineSense/
├── app.py                    # Main Flask app - START HERE
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
│
├── database/
│   ├── schema.sql           # Database structure
│   └── db_manager.py        # Database operations
│
├── ai/
│   ├── pairwise_learning.py # AI Layer 1
│   ├── embeddings.py        # AI Layer 2
│   ├── reinforcement.py     # AI Layer 3
│   └── recommender.py       # Main AI engine
│
├── api/
│   └── routes.py            # REST API endpoints
│
├── tmdb/
│   └── fetcher.py           # TMDB API integration
│
├── templates/               # HTML pages
└── static/                  # CSS/JS
```

---

## Quick Demo Script

```bash
# 1. Setup (one-time)
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your TMDB API key and MySQL password

# 2. Database (one-time)
mysql -u root -p -e "CREATE DATABASE cinesense"
mysql -u root -p cinesense < database/schema.sql

# 3. Data (one-time, takes 2 min)
python scripts/fetch_tmdb_data.py --test

# 4. Run (every time)
python app.py
```

---

## Next Steps

For full documentation, see:
- **SETUP_GUIDE.md** - Detailed setup instructions
- **ARCHITECTURE.md** - System architecture
- **README.md** - Project overview

---

**Ready to impress with your AI-powered movie platform!** 🎬
