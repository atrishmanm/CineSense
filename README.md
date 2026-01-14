# 🎬 CineSense - AI-Powered Movie Recommendation Platform

An intelligent movie recommendation system that learns your taste through simple comparisons and delivers personalized suggestions using advanced machine learning.

## 🌟 Key Features

- **🧠 Smart AI Recommendations**: Learns your preferences without tedious ratings
- **🎯 Pairwise Comparison**: Just pick which movie you prefer - the AI does the rest
- **🔮 Explainable AI**: Every recommendation comes with an explanation of why it suits you
- **🎨 Netflix-Style UI**: Premium streaming platform design with smooth interactions
- **📊 Multi-Layer Intelligence**: Combines collaborative filtering, content analysis, and reinforcement learning
- **⚡ Real-Time Learning**: System adapts instantly as you make choices

## 🛠️ Technology Stack

### Backend
- Python 3.10+
- Flask (REST API)
- MySQL (Database)

### AI/ML
- NumPy
- scikit-learn
- Custom online learning algorithms

### Frontend
- HTML5, CSS3, JavaScript
- Tailwind CSS
- Swiper.js (carousels)

### External APIs
- TMDB API (movie metadata)

## 🤖 How the AI Works

CineSense uses a **three-layer AI system** to understand your taste and recommend movies:

### 🎯 Layer 1: Preference Learning (Bradley-Terry Model)
When you choose between two movies, the system:
- Updates both movies' "preference scores" using an ELO-style algorithm
- Learns which features (genres, directors, themes) you favor
- Builds a mathematical model of your taste profile

**Example**: If you consistently pick action movies over comedies, the system boosts action scores in your profile.

### 🔍 Layer 2: Content Analysis (Vector Embeddings)
Every movie becomes a "feature vector" containing:
- **Genres** (Action, Drama, Comedy, etc.)
- **Cast & Directors** (Weighted by popularity)
- **Themes & Keywords** (from TMDB metadata)
- **Ratings & Popularity** (normalized scores)

Your preferences become a vector too. Recommendations use **cosine similarity** to find movies matching your unique taste fingerprint.

**Example**: Love Nolan + Sci-fi? Get *Interstellar*, *Arrival*, *Blade Runner 2049*.

### 🎲 Layer 3: Exploration (Multi-Armed Bandit)
The system balances:
- **Exploitation**: Show movies similar to what you love (safe bets)
- **Exploration**: Test new genres/directors to expand your taste (discovery)

Uses **Upper Confidence Bound (UCB)** algorithm to decide when to explore vs. exploit.

**Result**: You discover hidden gems without straying too far from your preferences.

## 🧪 Why This Approach?

Traditional recommendation systems have problems:
- ❌ **Ratings are tedious**: Who wants to rate 50 movies?
- ❌ **Cold start**: New users get generic suggestions
- ❌ **Filter bubbles**: You only see the same type of content

CineSense solves this:
- ✅ **Quick onboarding**: 5-10 comparisons reveal your taste
- ✅ **Continuous learning**: Every click improves recommendations
- ✅ **Explainable**: You see WHY each movie is recommended
- ✅ **Balanced discovery**: Smart mix of familiar + new

## 🗄️ Database Schema

Fully normalized (3NF) MySQL database:
- Users
- Movies
- Genres, Directors, Actors
- User Interactions (pairwise choices)
- User Embeddings (preference vectors)

## 🚀 Deployment

### Deploy to Render (100% FREE)

Complete step-by-step guide: **[RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md)**

**Quick Summary**:
1. Push code to GitHub
2. Create free MySQL database on Aiven
3. Deploy on Render (free tier)
4. Add environment variables
5. Initialize database via Render shell
6. Done! Your app is live 🎉

**Total Cost**: $0/month (no credit card required)

---

## 💻 Running CineSense from Scratch

### Prerequisites
- **Python 3.10+** ([Download](https://www.python.org/downloads/))
- **MySQL 8.0+** ([Download](https://dev.mysql.com/downloads/mysql/))
- **TMDB API Key** (Free - [Get here](https://www.themoviedb.org/settings/api))

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd CineSense
```

### Step 2: Set Up Python Environment

**Windows:**
```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Configure Database

**Start MySQL:**
```bash
# Windows: MySQL should auto-start as a service
# Mac: brew services start mysql
# Linux: sudo systemctl start mysql
```

**Create Database:**
```sql
CREATE DATABASE cinesense_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'cinesense_user'@'localhost' IDENTIFIED BY 'your_secure_password';
GRANT ALL PRIVILEGES ON cinesense_db.* TO 'cinesense_user'@'localhost';
FLUSH PRIVILEGES;
```

### Step 4: Configure Environment Variables

**Create `.env` file:**
```bash
# Copy the example
cp .env.example .env

# Or create manually with:
DB_HOST=localhost
DB_PORT=3306
DB_USER=cinesense_user
DB_PASSWORD=your_secure_password
DB_NAME=cinesense_db
TMDB_API_KEY=your_tmdb_api_key_here
```

**Get TMDB API Key:**
1. Go to [themoviedb.org](https://www.themoviedb.org/)
2. Create free account
3. Go to Settings → API → Request API Key
4. Choose "Developer" option
5. Copy your API key to `.env`

### Step 5: Initialize Database

```bash
# Run the database setup script
python setup_database.py
```

This will:
- Create all tables (users, movies, preferences, etc.)
- Fetch 5000+ movies from TMDB
- Generate embeddings for content-based filtering
- Set up initial indices

**⏱️ Time**: ~5-10 minutes depending on your internet speed

### Step 6: Start the Application

**Option A: Using PowerShell Script (Windows)**
```powershell
# Automated startup with browser launch
.\start-cinesense.ps1
```

**Option B: Manual Start**
```bash
# Make sure virtual environment is active
python app.py
```

Server starts at: **http://localhost:5000**

### Step 7: First-Time Setup

1. **Browse to** http://localhost:5000
2. **Click "Get Started"** to begin preference learning
3. **Make 5-10 comparisons** (pick which movie you prefer)
4. **Get personalized recommendations** on the homepage!

## 🎮 Quick Start with Demo Script

**Windows users** can use the automated script:

```powershell
.\start-cinesense.ps1
```

This script will:
- ✅ Check if MySQL is running
- ✅ Verify Python environment
- ✅ Activate virtual environment
- ✅ Start Flask server
- ✅ Auto-open browser after 3 seconds

### 3. Setup Database

```bash
# Login to MySQL
mysql -u root -p

# Create database
CREATE DATABASE cinesense;

# Run schema
mysql -u root -p cinesense < database/schema.sql
```

### 4. Fetch Movie Data

```bash
python scripts/fetch_tmdb_data.py --count 500
```

### 5. Run Application

```bash
# Start Flask server
python app.py

# Or use PowerShell script (Windows)
.\start-cinesense.ps1
```

Visit: http://localhost:5000

### 5. Run Application

```bash
python app.py
```

Visit: http://localhost:5000

## 📁 Project Structure

```
CineSense/
├── app.py                 # Main Flask application
├── config.py              # Configuration
├── requirements.txt       # Dependencies
├── .env                   # Environment variables
│
├── database/
│   ├── schema.sql        # Database schema
│   └── db_manager.py     # Database operations
│
├── ai/
│   ├── pairwise_learning.py   # Layer 1: Pairwise model
│   ├── embeddings.py          # Layer 2: Vector embeddings
│   ├── reinforcement.py       # Layer 3: RL bandit
│   └── recommender.py         # Main recommendation engine
│
├── api/
│   ├── routes.py         # API endpoints
│   └── auth.py           # User authentication
│
├── tmdb/
│   └── fetcher.py        # TMDB API integration
│
├── scripts/
│   └── fetch_tmdb_data.py  # Data acquisition script
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
└── templates/
    ├── index.html        # Home page
    ├── compare.html      # Pairwise comparison
    └── detail.html       # Movie detail
```

## 🎯 Usage Flow

1. **Sign Up**: Create user account
2. **Onboarding**: Select initial preferences (genres, directors)
3. **Pairwise Loop**: Choose between movie pairs
4. **AI Learning**: System adapts to your taste
5. **Recommendations**: Get personalized suggestions
6. **Watch**: Stream with provided links

## 🧠 AI Learning Process

```
User Choice (A > B)
    ↓
Update ELO Scores
    ↓
Adjust User Vector
    ↓
Update Movie Embeddings
    ↓
Bandit Policy Update
    ↓
Better Recommendations
```

## 📝 API Endpoints

- `GET /api/recommendations` - Get personalized recommendations
- `GET /api/featured` - Get featured movie for hero banner
- `GET /api/compare` - Get two movies for pairwise comparison
- `POST /api/feedback` - Submit user choice
- `POST /api/user/signup` - Create new user
- `GET /api/movie/<id>` - Get movie details

## 🎨 UI Design Philosophy

- **Dark theme** with high contrast
- **Big visuals** (posters, backdrops)
- **Smooth animations** for premium feel
- **Horizontal scrolling** rows (Netflix-style)
- **Responsive design** for all devices

## 📚 Database Design Highlights

- **Normalized structure** (3NF) prevents redundancy
- **Indexed tables** for fast queries
- **JSON storage** for vectors (efficient & flexible)
- **Timestamp tracking** for all interactions

## 🔐 Security Features

- Password hashing
- SQL injection prevention (parameterized queries)
- CORS configuration
- Environment variable protection

## 🧪 Testing

```bash
# Test database connection
python -m database.db_manager

# Test TMDB API
python -m tmdb.fetcher --test

# Test AI models
python -m ai.recommender --test
```

## 📈 Future Enhancements

- User reviews and comments
- Social features (friends, sharing)
- Deep learning models (LSTM, Transformers)
- Multi-modal embeddings (posters, trailers)
- A/B testing framework

## 👥 Team Contributions

Perfect for BTech CSE DBMS + AI projects:
- **Database**: Schema design, normalization, indexing
- **AI/ML**: Learning algorithms, embeddings, RL
- **Backend**: Flask API, business logic
- **Frontend**: UI/UX, responsive design
- **Integration**: TMDB API, data pipeline

## 📜 License

MIT License - Academic Project

## 🤝 Contributing

This is an educational project. Feel free to fork and enhance!

---

**Built with ❤️ for learning AI & DBMS concepts**
