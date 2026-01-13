# 🎬 CineSense - Complete Setup Guide

This guide will help you set up and run the CineSense AI-based movie recommendation platform.

## 📋 Prerequisites

Before starting, ensure you have:

1. **Python 3.10+** installed
2. **MySQL 8.0+** installed and running
3. **TMDB API Key** (free from https://www.themoviedb.org/settings/api)
4. **Git** (optional, for version control)

---

## 🚀 Step-by-Step Setup

### 1. Install Python Dependencies

Open a terminal in the project directory and run:

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

1. Copy the example environment file:
```bash
copy .env.example .env
```

2. Edit `.env` file and add your credentials:

```env
# TMDB API Configuration
TMDB_API_KEY=your_actual_tmdb_api_key_here

# Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=cinesense

# Flask Configuration
SECRET_KEY=generate_a_random_secret_key
PORT=5000
```

**Important**: Replace `your_actual_tmdb_api_key_here` with your real TMDB API key!

### 3. Set Up MySQL Database

1. **Start MySQL** (if not already running):
   - Windows: Start MySQL service from Services
   - Mac: `brew services start mysql`
   - Linux: `sudo systemctl start mysql`

2. **Login to MySQL**:
```bash
mysql -u root -p
```

3. **Create the database**:
```sql
CREATE DATABASE cinesense CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
EXIT;
```

4. **Run the schema script**:
```bash
mysql -u root -p cinesense < database/schema.sql
```

You should see: `CineSense database schema created successfully!`

### 4. Fetch Movie Data from TMDB

**This step is crucial!** It populates your database with movie data.

```bash
# Test mode: Fetch 50 movies (quick test)
python scripts/fetch_tmdb_data.py --test

# Full mode: Fetch 3000 movies (recommended)
python scripts/fetch_tmdb_data.py --count 3000
```

**This will take 15-30 minutes** due to API rate limits. Be patient!

The script will:
- ✓ Fetch diverse movies from TMDB
- ✓ Enrich with detailed information
- ✓ Store in database with proper relationships

### 5. Verify Database Setup

Test the database connection:

```bash
python -m database.db_manager
```

You should see:
```
✓ Connection successful!
  Movies in database: 3000
  Users in database: 0
```

### 6. Run the Application

```bash
python app.py
```

You should see:
```
============================================================
🎬 CINESENSE - AI Movie Recommendation Platform
============================================================
Server running on: http://localhost:5000
API documentation: http://localhost:5000/api
============================================================
```

### 7. Open in Browser

Navigate to: **http://localhost:5000**

---

## 🧪 Testing the Application

### Test Flow:

1. **Sign Up**
   - Go to http://localhost:5000/signup
   - Create an account
   - You'll be redirected to the comparison page

2. **Make Comparisons**
   - Choose between movie pairs
   - Make at least 5-10 comparisons
   - Watch the AI learning progress bar increase

3. **View Recommendations**
   - Go to home page (http://localhost:5000)
   - See personalized recommendations
   - Notice the "🎯 Recommended For You" section appears

4. **Explore Movies**
   - Click on any movie card
   - See detailed information
   - Read AI-generated explanations

---

## 📊 Understanding the AI System

### Three AI Layers:

**Layer 1: Pairwise Learning (ELO System)**
- Every comparison updates movie scores
- Uses Bradley-Terry model
- Stores in `user_interactions` table

**Layer 2: Vector Embeddings**
- Each movie = vector of features
- Each user = preference vector
- Uses cosine similarity
- Stores in `movie_embeddings` and `user_embeddings` tables

**Layer 3: Reinforcement Learning**
- Multi-Armed Bandit algorithm
- Balances exploration vs exploitation
- Chooses which movies to show next

---

## 🗄️ Database Structure

The system uses a fully normalized (3NF) schema:

```
users → user_interactions → movies
              ↓
        user_embeddings
              
movies → movie_genres → genres
      → movie_directors → directors
      → movie_actors → actors
      → movie_embeddings
```

Key tables:
- **users**: User accounts
- **movies**: Movie metadata
- **user_interactions**: Pairwise choices (the core!)
- **user_embeddings**: AI-learned user preferences
- **movie_embeddings**: AI-computed movie features

---

## 🔧 Troubleshooting

### Issue: "TMDB API key not found"
**Solution**: Make sure you've added your API key to `.env` file

### Issue: "Database connection failed"
**Solution**: 
- Check MySQL is running
- Verify credentials in `.env`
- Ensure database `cinesense` exists

### Issue: "No movies found"
**Solution**: Run the data fetching script:
```bash
python scripts/fetch_tmdb_data.py --test
```

### Issue: "Module not found"
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Port 5000 already in use
**Solution**: Change port in `.env`:
```env
PORT=8000
```

---

## 🎯 Project Features Checklist

✅ User registration and authentication
✅ Pairwise movie comparison interface
✅ AI learning from user choices
✅ Personalized recommendations
✅ Netflix-style UI with hero banner
✅ Movie detail pages
✅ Search functionality
✅ Top-rated and trending sections
✅ Explainable AI (shows why movies are recommended)
✅ Real-time learning progress tracking
✅ Fully normalized database (3NF)
✅ Vector embeddings for content-based filtering
✅ Reinforcement learning (Multi-Armed Bandit)
✅ ELO-based pairwise preference learning

---

## 📝 API Endpoints

### User
- `POST /api/user/signup` - Create account
- `POST /api/user/login` - Login
- `POST /api/user/logout` - Logout
- `GET /api/user/profile` - Get user stats

### Recommendations
- `GET /api/recommendations` - Get personalized recommendations
- `GET /api/featured` - Get featured movie
- `GET /api/compare` - Get comparison pair
- `POST /api/feedback` - Submit pairwise choice

### Movies
- `GET /api/movie/<id>` - Get movie details
- `GET /api/movie/search?q=<query>` - Search movies
- `GET /api/movie/top-rated` - Get top-rated movies
- `GET /api/movie/by-genre/<genre>` - Get movies by genre

---

## 🎓 For BTech Project Presentation

### Key Points to Highlight:

1. **Real AI Implementation**
   - Not just simple filtering
   - Three integrated AI techniques
   - Continuous online learning

2. **Database Design**
   - Fully normalized (3NF)
   - Efficient indexing
   - Proper foreign key relationships

3. **Novel Approach**
   - Pairwise comparison (like Netflix)
   - No rating fatigue
   - More accurate preference learning

4. **Scalability**
   - Connection pooling
   - Indexed queries
   - Efficient vector storage

5. **User Experience**
   - Netflix-quality UI
   - Responsive design
   - Explainable recommendations

### Demo Flow:

1. Show database schema (3NF design)
2. Explain AI architecture (three layers)
3. Live demo: signup → compare → recommendations
4. Show learning progress
5. Explain recommendation with AI reasoning

---

## 🚀 Next Steps (Optional Enhancements)

1. **Add Deep Learning**
   - Use PyTorch for neural collaborative filtering
   - Implement autoencoders for embeddings

2. **Social Features**
   - User profiles
   - Friend recommendations
   - Sharing capabilities

3. **Analytics Dashboard**
   - User engagement metrics
   - Popular genres analysis
   - Recommendation accuracy tracking

4. **Mobile App**
   - React Native frontend
   - Same Flask backend
   - Real-time push notifications

---

## 📞 Support

If you encounter any issues:

1. Check this guide thoroughly
2. Review error messages in terminal
3. Verify database setup
4. Ensure all dependencies are installed

---

## 🎉 Success!

If everything is working:
- ✓ Database populated with movies
- ✓ Flask server running
- ✓ AI making recommendations
- ✓ UI looks like Netflix

**You're ready to demo your project!** 🎬

---

Built with ❤️ for BTech CSE DBMS + AI Project
