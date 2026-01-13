# 🎬 CineSense - AI-Based Movie Recommendation Platform

An intelligent movie recommendation system that learns user preferences through pairwise comparisons and adaptive AI algorithms.

## 🌟 Key Features

- **Adaptive Learning**: System improves with every user interaction
- **Pairwise Comparison**: Choose between movies instead of rating
- **Vector Embeddings**: Content-based intelligent recommendations
- **Reinforcement Learning**: Multi-Armed Bandit exploration strategy
- **Netflix-Style UI**: Premium OTT platform experience
- **Explainable AI**: Know why movies are recommended

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

## 📊 AI Architecture

### Layer 1: Pairwise Preference Learning
- Bradley-Terry model
- ELO-style scoring system
- Learning-to-Rank algorithm

### Layer 2: Vector Embeddings
- Movie feature vectors (genres, directors, cast, ratings)
- User preference vectors
- Cosine similarity matching

### Layer 3: Reinforcement Learning
- Multi-Armed Bandit algorithm
- Exploration vs Exploitation balance
- Dynamic policy adaptation

## 🗄️ Database Schema

Fully normalized (3NF) MySQL database:
- Users
- Movies
- Genres, Directors, Actors
- User Interactions (pairwise choices)
- User Embeddings (preference vectors)

## 🚀 Setup Instructions

### 1. Clone and Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your:
# - TMDB API key
# - MySQL credentials
```

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
python scripts/fetch_tmdb_data.py
```

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
