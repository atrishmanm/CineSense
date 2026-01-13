# 🎬 CineSense - Project Architecture

## System Overview

CineSense is a sophisticated AI-based movie recommendation platform that learns user preferences through pairwise comparisons and adaptive algorithms.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                              │
│  (HTML/CSS/JavaScript + Tailwind + Swiper.js)              │
│                                                             │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌─────────┐  │
│  │   Home   │  │  Compare  │  │  Detail  │  │  Login  │  │
│  └──────────┘  └───────────┘  └──────────┘  └─────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP/REST API
┌───────────────────────┴─────────────────────────────────────┐
│                    FLASK BACKEND                            │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  API Routes                           │  │
│  │  • User Management  • Recommendations                │  │
│  │  • Feedback         • Movie Details                  │  │
│  └───────────┬──────────────────────────┬────────────────┘  │
│              │                          │                   │
│  ┌───────────▼──────────┐  ┌───────────▼───────────────┐  │
│  │   Database Manager   │  │   AI Recommendation Engine│  │
│  │  • Connection Pool   │  │   ┌───────────────────┐  │  │
│  │  • CRUD Operations   │  │   │ Layer 1: Pairwise│  │  │
│  │  • Query Optimization│  │   │ (ELO/Bradley-Terry)│  │  │
│  └──────────────────────┘  │   └───────────────────┘  │  │
│                            │   ┌───────────────────┐  │  │
│                            │   │ Layer 2: Embeddings│  │  │
│                            │   │ (Content-Based)   │  │  │
│                            │   └───────────────────┘  │  │
│                            │   ┌───────────────────┐  │  │
│                            │   │ Layer 3: RL Bandit│  │  │
│                            │   │ (Exploration)     │  │  │
│                            │   └───────────────────┘  │  │
│                            └───────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────────────┐
│                    MySQL DATABASE                           │
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌──────────────┐              │
│  │  Users  │  │ Movies  │  │ Interactions │              │
│  └────┬────┘  └────┬────┘  └──────┬───────┘              │
│       │            │               │                       │
│  ┌────▼─────┐ ┌───▼──────┐  ┌────▼────────┐             │
│  │User      │ │Movie     │  │ Genres      │             │
│  │Embeddings│ │Embeddings│  │ Directors   │             │
│  └──────────┘ └──────────┘  │ Actors      │             │
│                              └─────────────┘             │
└─────────────────────────────────────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────────────┐
│                    TMDB API                                 │
│  (External Movie Database - Data Source)                   │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Frontend Layer

**Technology**: HTML5, CSS3, JavaScript, Tailwind CSS, Swiper.js

**Pages**:
- **index.html**: Home page with hero banner and recommendation rows
- **compare.html**: Pairwise comparison interface (unique feature)
- **detail.html**: Movie detail view with AI explanations
- **login.html / signup.html**: Authentication pages

**Design Philosophy**:
- Netflix-inspired dark theme
- Responsive across all devices
- Smooth animations and transitions
- Horizontal scrolling carousels
- Big visuals (posters, backdrops)

### 2. Backend Layer

**Technology**: Python, Flask, Flask-CORS

**Modules**:

#### API Routes (`api/routes.py`)
- User authentication (signup, login, logout)
- Recommendation endpoints
- Comparison pair generation
- Feedback submission
- Movie details and search

#### Database Manager (`database/db_manager.py`)
- Connection pooling (10 connections)
- Context managers for safe operations
- CRUD operations for all entities
- Optimized queries with indexes
- Stored procedures for ELO updates

#### Configuration (`config.py`)
- Environment variable management
- Database configuration
- AI hyperparameters
- Feature dimensions

### 3. AI/ML Layer

**Three Integrated Layers**:

#### Layer 1: Pairwise Learning (`ai/pairwise_learning.py`)
- **ELO Rating System**: Like chess ratings
- **Bradley-Terry Model**: Probability-based ranking
- **User Preference Model**: Personalized taste tracking
- Learns from every comparison

#### Layer 2: Vector Embeddings (`ai/embeddings.py`)
- **Movie Embeddings**: 55-dimensional vectors
  - Genres (20 dim)
  - Directors (10 dim)
  - Actors (20 dim)
  - Metadata (5 dim: rating, popularity, year, votes, runtime)
- **User Embeddings**: Learned preference vectors
- **Content-Based Filtering**: Cosine similarity matching
- **Feature Encoders**: Multi-hot encoding for categorical data

#### Layer 3: Reinforcement Learning (`ai/reinforcement.py`)
- **Multi-Armed Bandit**: UCB (Upper Confidence Bound)
- **Exploration vs Exploitation**: Balances novelty and quality
- **Thompson Sampling**: Bayesian approach
- **Contextual Bandit**: User state-aware selection

#### Main Recommender (`ai/recommender.py`)
- Integrates all three layers
- Processes user choices
- Generates personalized recommendations
- Produces explainable AI outputs
- Handles cold start (new users)

### 4. Data Layer

**Technology**: MySQL 8.0+

**Schema Design**: Fully Normalized (3NF)

**Key Tables**:

```sql
users
├── user_id (PK)
├── username
├── email
├── password_hash
└── interaction_count

movies
├── movie_id (PK)
├── tmdb_id
├── title
├── overview
├── release_year
├── poster_path
├── backdrop_path
├── tmdb_rating
├── elo_score (AI-computed)
└── popularity

user_interactions (THE CORE!)
├── interaction_id (PK)
├── user_id (FK)
├── movie_1_id (FK)
├── movie_2_id (FK)
├── chosen_movie_id (FK)
├── rejected_movie_id (FK)
└── timestamp

user_embeddings
├── user_id (FK)
├── feature_index
└── feature_value

movie_embeddings
├── movie_id (FK)
├── feature_index
└── feature_value
```

**Relationships**:
- Many-to-Many: movies ↔ genres
- Many-to-Many: movies ↔ directors
- Many-to-Many: movies ↔ actors
- One-to-Many: users → interactions
- One-to-One: users ↔ embeddings

**Indexes**: Strategic indexes on:
- Foreign keys
- Frequently queried columns (rating, popularity, ELO)
- Composite indexes for common queries
- Full-text search on title/overview

### 5. External Services

**TMDB API**:
- Movie metadata
- Cast and crew information
- Images (posters, backdrops)
- Ratings and popularity

**Data Fetching Process**:
1. Fetch popular movies (50%)
2. Fetch top-rated movies (30%)
3. Fetch genre-diverse movies (20%)
4. Enrich with detailed information
5. Store in normalized database

## Data Flow

### User Registration Flow
```
User fills form → Frontend validates → POST /api/user/signup
→ Backend hashes password → Database stores user
→ Session created → Redirect to compare
```

### Pairwise Comparison Flow
```
User requests comparison → GET /api/compare
→ Bandit selects 2 movies → Backend fetches from DB
→ Frontend displays movies → User chooses one
→ POST /api/feedback → AI processes choice
→ Update ELO scores → Update user embedding
→ Update bandit statistics → Save to database
→ Return success → Show next pair
```

### Recommendation Flow
```
User requests recommendations → GET /api/recommendations
→ Backend checks user history → Load user embedding
→ AI Layer 1: Get pairwise preferences
→ AI Layer 2: Calculate content similarity
→ AI Layer 3: Apply bandit exploration
→ Combine scores (weighted sum) → Rank movies
→ Return top N → Frontend displays cards
```

## AI Algorithm Details

### Pairwise Preference Learning

**ELO Update Formula**:
```python
Expected_A = 1 / (1 + 10^((Rating_B - Rating_A) / 400))
New_Rating_A = Rating_A + K * (Actual - Expected_A)
```

**Bradley-Terry Model**:
```python
P(i > j) = strength_i / (strength_i + strength_j)
strength_i *= (1 + lr * (1 - P(i > j)))  # if i wins
```

### Vector Embeddings

**Movie Vector Creation**:
```python
vector = [
    genre_features[20],      # One-hot encoded genres
    director_features[10],   # Top directors
    actor_features[20],      # Top cast
    rating/10,               # Normalized rating
    log(popularity)/10,      # Log-normalized popularity
    (year-1900)/130,        # Normalized year
    log(votes)/15,          # Log-normalized votes
    runtime/300             # Normalized runtime
]
```

**User Embedding Update**:
```python
user_vector += learning_rate * chosen_movie_vector
user_vector -= learning_rate * rejected_movie_vector
```

### Reinforcement Learning

**UCB Selection**:
```python
UCB_value = avg_reward + c * sqrt(ln(total_trials) / arm_trials)
# Select arm with highest UCB
```

**Multi-Armed Bandit**:
- Epsilon-greedy: 80% exploit, 20% explore
- UCB: Automatic exploration bonus
- Thompson Sampling: Bayesian approach

## Performance Optimizations

1. **Database**:
   - Connection pooling (reuse connections)
   - Indexed queries (fast lookups)
   - Stored procedures (reduce round-trips)
   - Views for complex queries

2. **Backend**:
   - Singleton patterns (shared resources)
   - Context managers (safe cleanup)
   - Lazy loading (load on demand)
   - Caching (future enhancement)

3. **Frontend**:
   - Lazy image loading
   - Swiper.js for smooth carousels
   - Minimal DOM manipulation
   - Responsive breakpoints

4. **AI**:
   - Numpy vectorization
   - Efficient similarity calculations
   - Batch processing where possible
   - Sparse vector storage

## Security Measures

1. **Authentication**:
   - Password hashing (werkzeug)
   - Session management
   - CSRF protection (Flask built-in)

2. **Database**:
   - Parameterized queries (SQL injection prevention)
   - Foreign key constraints
   - Input validation

3. **API**:
   - Rate limiting (can be added)
   - Error handling
   - Proper HTTP status codes

## Scalability Considerations

**Current Design** (Single Server):
- ✓ Handles 100s of concurrent users
- ✓ Database connection pooling
- ✓ Efficient queries

**Future Enhancements**:
- Redis for caching
- Load balancing
- Database replication
- Microservices architecture
- CDN for static assets

## Testing Strategy

1. **Unit Tests**: Individual components
2. **Integration Tests**: API endpoints
3. **Database Tests**: Schema validation
4. **AI Tests**: Algorithm correctness
5. **UI Tests**: Selenium/Playwright

## Deployment Considerations

**Development**:
- Flask development server
- Local MySQL
- Debug mode enabled

**Production**:
- Gunicorn/uWSGI
- Nginx reverse proxy
- MySQL with replication
- Environment-based config
- SSL/TLS certificates
- Monitoring (Prometheus, Grafana)

---

## Key Innovations

1. **Pairwise Comparison**: More accurate than rating scales
2. **Three-Layer AI**: Comprehensive learning approach
3. **Online Learning**: Improves with every interaction
4. **Explainable AI**: Users understand recommendations
5. **Cold Start Handling**: Works for new users
6. **Content + Collaborative**: Best of both worlds

---

**Built for**: BTech CSE DBMS + AI Project
**Focus**: Real AI implementation, not toy project
**Result**: Production-ready recommendation system
