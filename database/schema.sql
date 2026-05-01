-- Active: 1768449834747@@127.0.0.1@3306@cinesense

USE cinesense;

-- ============================================================================
-- DROP EXISTING OBJECTS (For Clean Setup)
-- ============================================================================
DROP TABLE IF EXISTS genre_win_counts;
DROP TABLE IF EXISTS movie_reviews;
DROP TABLE IF EXISTS watchlist;
DROP TABLE IF EXISTS ab_test_metrics;
DROP TABLE IF EXISTS ab_test_assignments;
DROP TABLE IF EXISTS recommendation_feedback;
DROP TABLE IF EXISTS user_moods;
DROP TABLE IF EXISTS search_logs;
DROP TABLE IF EXISTS semantic_embeddings;
DROP TABLE IF EXISTS model_versions;
DROP TABLE IF EXISTS movie_keywords;
DROP TABLE IF EXISTS keywords;
DROP TABLE IF EXISTS recommendation_log;
DROP TABLE IF EXISTS candidate_generation_log;
DROP TABLE IF EXISTS cache_stats;
DROP TABLE IF EXISTS user_preferences;
DROP TABLE IF EXISTS search_history;
DROP TABLE IF EXISTS user_interactions;
DROP TABLE IF EXISTS user_embeddings;
DROP TABLE IF EXISTS movie_embeddings;
DROP TABLE IF EXISTS movie_actors;
DROP TABLE IF EXISTS movie_directors;
DROP TABLE IF EXISTS movie_genres;
DROP TABLE IF EXISTS actors;
DROP TABLE IF EXISTS directors;
DROP TABLE IF EXISTS genres;
DROP TABLE IF EXISTS movies;
DROP TABLE IF EXISTS users;

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- USERS TABLE
CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    interaction_count INT DEFAULT 0,

    -- CHECK constraints
    CONSTRAINT chk_interaction_count CHECK (interaction_count >= 0),
    CONSTRAINT chk_username_length   CHECK (CHAR_LENGTH(username) >= 3),

    INDEX idx_username (username),
    INDEX idx_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- MOVIES TABLE
CREATE TABLE movies (
    movie_id INT PRIMARY KEY,
    tmdb_id INT NOT NULL UNIQUE,
    title VARCHAR(255) NOT NULL,
    original_title VARCHAR(255),
    overview TEXT,
    release_year INT,
    runtime INT,
    poster_path VARCHAR(255),
    backdrop_path VARCHAR(255),
    tmdb_rating DECIMAL(3,1),
    vote_count INT,
    popularity DECIMAL(10,3),
    watch_link VARCHAR(500),
    elo_score INT DEFAULT 1500,
    comparison_count INT DEFAULT 0,
    movie_source ENUM('tmdb_api', 'user_interaction', 'cache', 'database') DEFAULT 'database',
    is_persisted BOOLEAN DEFAULT FALSE,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    access_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- CHECK constraints (MySQL 8.0.16+)
    CONSTRAINT chk_tmdb_rating      CHECK (tmdb_rating      IS NULL OR (tmdb_rating BETWEEN 0.0 AND 10.0)),
    CONSTRAINT chk_release_year     CHECK (release_year     IS NULL OR (release_year BETWEEN 1888 AND 2100)),
    CONSTRAINT chk_runtime          CHECK (runtime          IS NULL OR runtime > 0),
    CONSTRAINT chk_vote_count       CHECK (vote_count       IS NULL OR vote_count >= 0),
    CONSTRAINT chk_elo_score        CHECK (elo_score        >= 0),
    CONSTRAINT chk_comparison_count CHECK (comparison_count >= 0),
    CONSTRAINT chk_access_count     CHECK (access_count >= 0),

    INDEX idx_title (title),
    INDEX idx_release_year (release_year),
    INDEX idx_elo_score (elo_score),
    INDEX idx_popularity (popularity),
    INDEX idx_movie_source (movie_source),
    INDEX idx_is_persisted (is_persisted),
    INDEX idx_last_accessed (last_accessed),
    INDEX idx_access_count (access_count),
    INDEX idx_persisted_accessed (is_persisted, last_accessed DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- GENRES TABLE (Normalized)
CREATE TABLE genres (
    genre_id INT AUTO_INCREMENT PRIMARY KEY,
    genre_name VARCHAR(50) NOT NULL UNIQUE,
    tmdb_genre_id INT UNIQUE,
    INDEX idx_genre_name (genre_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- DIRECTORS TABLE (Normalized)
CREATE TABLE directors (
    director_id INT AUTO_INCREMENT PRIMARY KEY,
    director_name VARCHAR(100) NOT NULL,
    tmdb_person_id INT UNIQUE,
    popularity DECIMAL(10,3),
    INDEX idx_director_name (director_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ACTORS TABLE (Normalized)
CREATE TABLE actors (
    actor_id INT AUTO_INCREMENT PRIMARY KEY,
    actor_name VARCHAR(100) NOT NULL,
    tmdb_person_id INT UNIQUE,
    popularity DECIMAL(10,3),
    INDEX idx_actor_name (actor_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- MOVIE_GENRES TABLE (Many-to-Many Relationship)
CREATE TABLE movie_genres (
    movie_id INT NOT NULL,
    genre_id INT NOT NULL,
    PRIMARY KEY (movie_id, genre_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    FOREIGN KEY (genre_id) REFERENCES genres(genre_id) ON DELETE CASCADE,
    INDEX idx_genre_lookup (genre_id, movie_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- MOVIE_DIRECTORS TABLE (Many-to-Many Relationship)
CREATE TABLE movie_directors (
    movie_id INT NOT NULL,
    director_id INT NOT NULL,
    PRIMARY KEY (movie_id, director_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    FOREIGN KEY (director_id) REFERENCES directors(director_id) ON DELETE CASCADE,
    INDEX idx_director_lookup (director_id, movie_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- MOVIE_ACTORS TABLE (Many-to-Many Relationship)
CREATE TABLE movie_actors (
    movie_id INT NOT NULL,
    actor_id INT NOT NULL,
    cast_order INT DEFAULT 0,
    character_name VARCHAR(255),
    PRIMARY KEY (movie_id, actor_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    FOREIGN KEY (actor_id) REFERENCES actors(actor_id) ON DELETE CASCADE,
    INDEX idx_actor_lookup (actor_id, movie_id),
    INDEX idx_cast_order (movie_id, cast_order)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- USER_INTERACTIONS TABLE (Pairwise Comparison History)
CREATE TABLE user_interactions (
    interaction_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    movie_1_id INT NOT NULL,
    movie_2_id INT NOT NULL,
    chosen_movie_id INT NOT NULL,
    rejected_movie_id INT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR(100),
    interaction_type ENUM('comparison', 'recommendation', 'search', 'view', 'click') DEFAULT 'comparison',
    is_lazy_loaded BOOLEAN DEFAULT FALSE,
    cache_hit BOOLEAN DEFAULT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (movie_1_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    FOREIGN KEY (movie_2_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    FOREIGN KEY (chosen_movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    FOREIGN KEY (rejected_movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    INDEX idx_user_interactions (user_id, timestamp DESC),
    INDEX idx_chosen_movie (chosen_movie_id),
    INDEX idx_rejected_movie (rejected_movie_id),
    INDEX idx_session (session_id),
    INDEX idx_lazy_loaded (is_lazy_loaded, timestamp DESC),
    INDEX idx_cache_hit (cache_hit)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- USER_EMBEDDINGS TABLE (User Preference Vectors)
CREATE TABLE user_embeddings (
    user_id INT NOT NULL,
    feature_index INT NOT NULL,
    feature_value DECIMAL(10,6) NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, feature_index),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX idx_user_vector (user_id, feature_index)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- MOVIE_EMBEDDINGS TABLE (Movie Feature Vectors)
CREATE TABLE movie_embeddings (
    movie_id INT NOT NULL,
    feature_index INT NOT NULL,
    feature_value DECIMAL(10,6) NOT NULL,
    PRIMARY KEY (movie_id, feature_index),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    INDEX idx_movie_vector (movie_id, feature_index)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- ENHANCEMENT TABLES (Advanced Features)
-- ============================================================================

-- SEARCH HISTORY TABLE (Track what users search for)
CREATE TABLE search_history (
    search_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    search_query VARCHAR(500) NOT NULL,
    search_type ENUM('title', 'storyline', 'semantic', 'advanced') DEFAULT 'title',
    results_count INT DEFAULT 0,
    search_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL,
    INDEX idx_search_query (search_query(255)),
    INDEX idx_search_timestamp (search_timestamp),
    INDEX idx_user_searches (user_id, search_timestamp DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- MOVIE KEYWORDS TABLE (For semantic search)
CREATE TABLE movie_keywords (
    keyword_id INT AUTO_INCREMENT PRIMARY KEY,
    movie_id INT NOT NULL,
    keyword VARCHAR(100) NOT NULL,
    relevance_score DECIMAL(5,2) DEFAULT 1.0,
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    CONSTRAINT chk_relevance_score CHECK (relevance_score BETWEEN 0.0 AND 10.0),
    UNIQUE KEY unique_movie_keyword (movie_id, keyword),
    INDEX idx_keyword (keyword),
    INDEX idx_movie_keywords (movie_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- USER PREFERENCES TABLE (Aggregated preferences for quick access)
CREATE TABLE user_preferences (
    user_id INT PRIMARY KEY,
    favorite_genre VARCHAR(50),
    favorite_actor VARCHAR(100),
    favorite_director VARCHAR(100),
    avg_rating_preference DECIMAL(3,1),
    preferred_decade INT,
    total_interactions INT DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- CACHE STATS TABLE (Track caching performance)
CREATE TABLE cache_stats (
    stat_id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    stat_date DATE NOT NULL,
    cache_type ENUM('movie', 'vector', 'user', 'recommendation') NOT NULL,
    cache_hits INT DEFAULT 0,
    cache_misses INT DEFAULT 0,
    movie_cache_size INT DEFAULT 0,
    vector_cache_size INT DEFAULT 0,
    movie_hits INT DEFAULT 0,
    movie_misses INT DEFAULT 0,
    vector_hits INT DEFAULT 0,
    vector_misses INT DEFAULT 0,
    movie_hit_rate DECIMAL(5,2) DEFAULT 0.00,
    vector_hit_rate DECIMAL(5,2) DEFAULT 0.00,
    refill_count INT DEFAULT 0,
    eviction_count INT DEFAULT 0,
    avg_response_time_ms DECIMAL(10,2) DEFAULT 0,
    memory_usage_kb INT DEFAULT 0,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_cache_stat (stat_date, cache_type),
    INDEX idx_stat_date (stat_date),
    INDEX idx_timestamp (timestamp DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- CANDIDATE GENERATION LOG TABLE (Lazy loading analytics)
CREATE TABLE candidate_generation_log (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    strategy ENUM('mixed', 'genre', 'popularity', 'exploration', 'cache') DEFAULT 'mixed',
    candidate_count INT DEFAULT 0,
    genre_candidates INT DEFAULT 0,
    popularity_candidates INT DEFAULT 0,
    exploration_candidates INT DEFAULT 0,
    cache_candidates INT DEFAULT 0,
    generation_time_ms INT DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX idx_user_timestamp (user_id, timestamp DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- RECOMMENDATION LOG TABLE (Track recommendations given to users)
CREATE TABLE recommendation_log (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    movie_id INT NOT NULL,
    recommendation_score DECIMAL(5,2),
    recommendation_reason VARCHAR(255),
    was_clicked BOOLEAN DEFAULT FALSE,
    was_watched BOOLEAN DEFAULT FALSE,
    recommended_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    INDEX idx_user_recommendations (user_id, recommended_at DESC),
    INDEX idx_movie_recommendations (movie_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- SAMPLE DATA: Pre-populate TMDB Genres
-- ============================================================================
INSERT INTO genres (genre_id, genre_name, tmdb_genre_id) VALUES
(1, 'Action', 28),
(2, 'Adventure', 12),
(3, 'Animation', 16),
(4, 'Comedy', 35),
(5, 'Crime', 80),
(6, 'Documentary', 99),
(7, 'Drama', 18),
(8, 'Family', 10751),
(9, 'Fantasy', 14),
(10, 'History', 36),
(11, 'Horror', 27),
(12, 'Music', 10402),
(13, 'Mystery', 9648),
(14, 'Romance', 10749),
(15, 'Science Fiction', 878),
(16, 'TV Movie', 10770),
(17, 'Thriller', 53),
(18, 'War', 10752),
(19, 'Western', 37);

-- ============================================================================
-- WATCHLIST & REVIEWS TABLES (DBMS Features: Constraints, Triggers, Functions)
-- ============================================================================

-- WATCHLIST TABLE (Comprehensive constraints demo)
CREATE TABLE watchlist (
    watchlist_id   INT            AUTO_INCREMENT,
    user_id        INT            NOT NULL,
    movie_id       INT            NOT NULL,
    priority       TINYINT        NOT NULL DEFAULT 5,
    status         ENUM('planned','watching','completed','dropped')
                                  NOT NULL DEFAULT 'planned',
    personal_note  VARCHAR(500),
    added_at       TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    watched_at     TIMESTAMP      NULL,
    user_rating    DECIMAL(3,1)   NULL,

    PRIMARY KEY (watchlist_id),
    UNIQUE KEY  uq_user_movie   (user_id, movie_id),
    FOREIGN KEY (user_id)  REFERENCES users(user_id)  ON DELETE CASCADE,
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,

    CONSTRAINT chk_priority    CHECK (priority   BETWEEN 1 AND 10),
    CONSTRAINT chk_user_rating CHECK (user_rating IS NULL
                                   OR user_rating BETWEEN 0.0 AND 10.0),
    CONSTRAINT chk_watched_order
        CHECK (watched_at IS NULL OR watched_at >= added_at),

    INDEX idx_watchlist_user   (user_id, added_at DESC),
    INDEX idx_watchlist_movie  (movie_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- MOVIE REVIEWS TABLE
CREATE TABLE movie_reviews (
    review_id     INT          AUTO_INCREMENT PRIMARY KEY,
    user_id       INT          NOT NULL,
    movie_id      INT          NOT NULL,
    rating        DECIMAL(3,1) NOT NULL,
    review_text   TEXT,
    is_spoiler    BOOLEAN      NOT NULL DEFAULT FALSE,
    helpful_votes INT          NOT NULL DEFAULT 0,
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
                               ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY  uq_user_movie_review (user_id, movie_id),
    FOREIGN KEY (user_id)  REFERENCES users(user_id)  ON DELETE CASCADE,
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,

    CONSTRAINT chk_review_rating CHECK (rating BETWEEN 0.0 AND 10.0),
    CONSTRAINT chk_helpful_votes CHECK (helpful_votes >= 0),

    FULLTEXT INDEX ft_review_text (review_text),
    INDEX idx_reviews_movie (movie_id, rating DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- GENRE WIN COUNTS TABLE (Materialized genre performance data)
CREATE TABLE genre_win_counts (
    genre_id   INT  NOT NULL PRIMARY KEY,
    win_count  INT  NOT NULL DEFAULT 0,
    loss_count INT  NOT NULL DEFAULT 0,
    FOREIGN KEY (genre_id) REFERENCES genres(genre_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Seed genre_win_counts
INSERT INTO genre_win_counts (genre_id, win_count, loss_count)
SELECT genre_id, 0, 0 FROM genres;

-- ============================================================================
-- VIEWS FOR EASIER QUERYING
-- ============================================================================

-- Complete movie information with all metadata
DROP VIEW IF EXISTS movie_details;
CREATE VIEW movie_details AS
SELECT 
    m.movie_id,
    m.tmdb_id,
    m.title,
    m.overview,
    m.release_year,
    m.runtime,
    m.poster_path,
    m.backdrop_path,
    m.tmdb_rating,
    m.vote_count,
    m.popularity,
    m.watch_link,
    m.elo_score,
    m.comparison_count,
    GROUP_CONCAT(DISTINCT g.genre_name ORDER BY g.genre_name SEPARATOR ', ') AS genres,
    GROUP_CONCAT(DISTINCT d.director_name ORDER BY d.director_name SEPARATOR ', ') AS directors,
    GROUP_CONCAT(DISTINCT a.actor_name ORDER BY ma.cast_order SEPARATOR ', ') AS cast
FROM movies m
LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
LEFT JOIN genres g ON mg.genre_id = g.genre_id
LEFT JOIN movie_directors md ON m.movie_id = md.movie_id
LEFT JOIN directors d ON md.director_id = d.director_id
LEFT JOIN movie_actors ma ON m.movie_id = ma.movie_id
LEFT JOIN actors a ON ma.actor_id = a.actor_id
GROUP BY m.movie_id;

-- User statistics
DROP VIEW IF EXISTS user_stats;
CREATE VIEW user_stats AS
SELECT 
    u.user_id,
    u.username,
    u.interaction_count,
    u.created_at,
    u.last_active,
    COUNT(DISTINCT ui.interaction_id) AS total_comparisons,
    COUNT(DISTINCT ui.chosen_movie_id) AS unique_movies_chosen,
    DATEDIFF(CURRENT_DATE, u.created_at) AS days_active
FROM users u
LEFT JOIN user_interactions ui ON u.user_id = ui.user_id
GROUP BY u.user_id;

-- Comprehensive Movie Details with All Metadata (Enhanced)
DROP VIEW IF EXISTS comprehensive_movie_view;
CREATE VIEW comprehensive_movie_view AS
SELECT 
    m.movie_id,
    m.tmdb_id,
    m.title,
    m.original_title,
    m.overview,
    m.release_year,
    m.runtime,
    m.poster_path,
    m.backdrop_path,
    m.tmdb_rating,
    m.vote_count,
    m.popularity,
    m.watch_link,
    m.elo_score,
    m.comparison_count,
    m.created_at,
    GROUP_CONCAT(DISTINCT g.genre_name ORDER BY g.genre_name SEPARATOR ', ') AS genres,
    GROUP_CONCAT(DISTINCT d.director_name ORDER BY d.director_name SEPARATOR ', ') AS directors,
    GROUP_CONCAT(DISTINCT a.actor_name ORDER BY ma.cast_order SEPARATOR ', ') AS cast,
    GROUP_CONCAT(DISTINCT mk.keyword ORDER BY mk.relevance_score DESC SEPARATOR ', ') AS keywords,
    (SELECT COUNT(*) + 1 FROM movies m2 WHERE m2.popularity > m.popularity) AS popularity_rank,
    (SELECT COUNT(*) + 1 FROM movies m2 WHERE m2.tmdb_rating > m.tmdb_rating) AS rating_rank
FROM movies m
LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
LEFT JOIN genres g ON mg.genre_id = g.genre_id
LEFT JOIN movie_directors md ON m.movie_id = md.movie_id
LEFT JOIN directors d ON md.director_id = d.director_id
LEFT JOIN movie_actors ma ON m.movie_id = ma.movie_id
LEFT JOIN actors a ON ma.actor_id = a.actor_id
LEFT JOIN movie_keywords mk ON m.movie_id = mk.movie_id
GROUP BY m.movie_id;

-- Advanced User Statistics View
DROP VIEW IF EXISTS advanced_user_stats;
CREATE VIEW advanced_user_stats AS
SELECT 
    u.user_id,
    u.username,
    u.email,
    u.created_at,
    u.last_active,
    u.interaction_count,
    COUNT(DISTINCT ui.interaction_id) AS total_comparisons,
    COUNT(DISTINCT ui.chosen_movie_id) AS unique_movies_chosen,
    COUNT(DISTINCT ui.rejected_movie_id) AS unique_movies_rejected,
    DATEDIFF(CURRENT_DATE, u.created_at) AS days_since_joined,
    DATEDIFF(CURRENT_DATE, u.last_active) AS days_since_last_active,
    AVG(m.tmdb_rating) AS avg_chosen_rating,
    AVG(m.release_year) AS avg_chosen_year,
    MIN(m.release_year) AS oldest_movie_chosen,
    MAX(m.release_year) AS newest_movie_chosen,
    CASE 
        WHEN COUNT(ui.interaction_id) = 0 THEN 'New User'
        WHEN COUNT(ui.interaction_id) < 10 THEN 'Beginner'
        WHEN COUNT(ui.interaction_id) < 50 THEN 'Intermediate'
        WHEN COUNT(ui.interaction_id) < 200 THEN 'Advanced'
        ELSE 'Expert'
    END AS user_level
FROM users u
LEFT JOIN user_interactions ui ON u.user_id = ui.user_id
LEFT JOIN movies m ON ui.chosen_movie_id = m.movie_id
GROUP BY u.user_id;

-- Genre Popularity View
DROP VIEW IF EXISTS genre_popularity_view;
CREATE VIEW genre_popularity_view AS
SELECT 
    g.genre_id,
    g.genre_name,
    COUNT(DISTINCT mg.movie_id) AS total_movies,
    AVG(m.tmdb_rating) AS avg_rating,
    AVG(m.popularity) AS avg_popularity,
    COUNT(DISTINCT ui.chosen_movie_id) AS times_chosen,
    SUM(m.vote_count) AS total_votes,
    RANK() OVER (ORDER BY COUNT(DISTINCT ui.chosen_movie_id) DESC) AS popularity_rank
FROM genres g
INNER JOIN movie_genres mg ON g.genre_id = mg.genre_id
INNER JOIN movies m ON mg.movie_id = m.movie_id
LEFT JOIN user_interactions ui ON m.movie_id = ui.chosen_movie_id
GROUP BY g.genre_id
HAVING COUNT(DISTINCT mg.movie_id) > 0;

-- Director Performance View
DROP VIEW IF EXISTS director_performance_view;
CREATE VIEW director_performance_view AS
SELECT 
    d.director_id,
    d.director_name,
    COUNT(DISTINCT md.movie_id) AS total_movies,
    AVG(m.tmdb_rating) AS avg_rating,
    MAX(m.tmdb_rating) AS highest_rated_movie_rating,
    AVG(m.popularity) AS avg_popularity,
    SUM(m.vote_count) AS total_votes,
    COUNT(DISTINCT ui.chosen_movie_id) AS times_chosen_in_comparisons,
    YEAR(CURRENT_DATE) - MIN(m.release_year) AS years_active
FROM directors d
INNER JOIN movie_directors md ON d.director_id = md.director_id
INNER JOIN movies m ON md.movie_id = m.movie_id
LEFT JOIN user_interactions ui ON m.movie_id = ui.chosen_movie_id
GROUP BY d.director_id
HAVING COUNT(DISTINCT md.movie_id) > 0;

-- Actor Performance View
DROP VIEW IF EXISTS actor_performance_view;
CREATE VIEW actor_performance_view AS
SELECT 
    a.actor_id,
    a.actor_name,
    COUNT(DISTINCT ma.movie_id) AS total_movies,
    AVG(m.tmdb_rating) AS avg_movie_rating,
    MAX(m.tmdb_rating) AS highest_rated_movie,
    AVG(m.popularity) AS avg_popularity,
    COUNT(DISTINCT ui.chosen_movie_id) AS times_chosen_in_comparisons
FROM actors a
INNER JOIN movie_actors ma ON a.actor_id = ma.actor_id
INNER JOIN movies m ON ma.movie_id = m.movie_id
LEFT JOIN user_interactions ui ON m.movie_id = ui.chosen_movie_id
GROUP BY a.actor_id
HAVING COUNT(DISTINCT ma.movie_id) > 0;

-- Daily Activity Statistics View
DROP VIEW IF EXISTS daily_activity_stats;
CREATE VIEW daily_activity_stats AS
SELECT 
    DATE(ui.timestamp) AS activity_date,
    COUNT(DISTINCT ui.user_id) AS unique_users,
    COUNT(ui.interaction_id) AS total_comparisons,
    COUNT(DISTINCT ui.chosen_movie_id) AS unique_movies_chosen,
    AVG(m.tmdb_rating) AS avg_rating_of_chosen_movies,
    COUNT(ui.interaction_id) - LAG(COUNT(ui.interaction_id)) OVER (ORDER BY DATE(ui.timestamp)) AS comparison_change
FROM user_interactions ui
INNER JOIN movies m ON ui.chosen_movie_id = m.movie_id
GROUP BY DATE(ui.timestamp)
ORDER BY activity_date DESC;

-- Trending Movies View (Based on recent comparisons)
DROP VIEW IF EXISTS trending_movies_view;
CREATE VIEW trending_movies_view AS
SELECT 
    m.movie_id,
    m.title,
    m.poster_path,
    m.tmdb_rating,
    m.release_year,
    COUNT(DISTINCT ui.interaction_id) AS recent_comparisons,
    COUNT(DISTINCT CASE WHEN ui.chosen_movie_id = m.movie_id THEN ui.interaction_id END) AS times_chosen,
    COUNT(DISTINCT CASE WHEN ui.rejected_movie_id = m.movie_id THEN ui.interaction_id END) AS times_rejected,
    ROUND((COUNT(DISTINCT CASE WHEN ui.chosen_movie_id = m.movie_id THEN ui.interaction_id END) /
           NULLIF(COUNT(DISTINCT ui.interaction_id), 0)) * 100, 2) AS win_rate_percentage
FROM movies m
LEFT JOIN user_interactions ui ON (m.movie_id = ui.movie_1_id OR m.movie_id = ui.movie_2_id)
    AND ui.timestamp >= DATE_SUB(CURRENT_DATE, INTERVAL 7 DAY)
GROUP BY m.movie_id
HAVING recent_comparisons > 0
ORDER BY recent_comparisons DESC, win_rate_percentage DESC;

-- Elite Movies View (Set Operations Example)
DROP VIEW IF EXISTS elite_movies;
CREATE VIEW elite_movies AS
SELECT movie_id, title, 'Elite' AS category FROM movies
WHERE tmdb_rating >= 8.0 AND popularity >= 100
UNION
SELECT movie_id, title, 'Rising Star' AS category FROM movies
WHERE comparison_count > 50 AND elo_score > 1600;

-- INTERSECT concept (MySQL-compatible emulation)
-- "Highly rated" ∩ "Frequently compared"
DROP VIEW IF EXISTS intersect_movies_emulation;
CREATE VIEW intersect_movies_emulation AS
SELECT hr.movie_id, hr.title
FROM (
    SELECT movie_id, title
    FROM movies
    WHERE tmdb_rating >= 8.0
) hr
INNER JOIN (
    SELECT movie_id
    FROM movies
    WHERE comparison_count >= 20
) fc ON hr.movie_id = fc.movie_id;

-- EXCEPT concept (MySQL-compatible emulation)
-- "Highly rated" \ "Frequently compared"
DROP VIEW IF EXISTS except_movies_emulation;
CREATE VIEW except_movies_emulation AS
SELECT hr.movie_id, hr.title
FROM (
    SELECT movie_id, title
    FROM movies
    WHERE tmdb_rating >= 8.0
) hr
LEFT JOIN (
    SELECT movie_id
    FROM movies
    WHERE comparison_count >= 20
) fc ON hr.movie_id = fc.movie_id
WHERE fc.movie_id IS NULL;

-- User Network View (Complex Join Example)
DROP VIEW IF EXISTS user_network_view;
CREATE VIEW user_network_view AS
SELECT 
    u1.user_id,
    u1.username,
    up1.favorite_genre,
    COUNT(DISTINCT u2.user_id) AS similar_users_count,
    GROUP_CONCAT(DISTINCT u2.username SEPARATOR ', ') AS similar_users
FROM users u1
INNER JOIN user_preferences up1 ON u1.user_id = up1.user_id
LEFT JOIN user_preferences up2 ON up1.favorite_genre = up2.favorite_genre 
    AND up1.user_id != up2.user_id
LEFT JOIN users u2 ON up2.user_id = u2.user_id
GROUP BY u1.user_id;

-- Power Users View (Subquery with ALL, ANY, EXISTS Example)
DROP VIEW IF EXISTS power_users;
CREATE VIEW power_users AS
SELECT u.user_id, u.username, u.interaction_count
FROM users u
WHERE u.interaction_count > ALL (
    SELECT AVG(interaction_count) FROM users
)
AND EXISTS (
    SELECT 1 FROM user_preferences up
    WHERE up.user_id = u.user_id
    AND up.total_interactions > 20
);

-- ============================================================================
-- STORED PROCEDURES
-- ============================================================================

-- Update user interaction count
DROP PROCEDURE IF EXISTS update_user_interaction_count;
DELIMITER //
CREATE PROCEDURE update_user_interaction_count(IN p_user_id INT)
BEGIN
    DECLARE interaction_total INT;
    
    SELECT COUNT(*) INTO interaction_total
    FROM user_interactions
    WHERE user_id = p_user_id;
    
    UPDATE users
    SET interaction_count = interaction_total,
        last_active = CURRENT_TIMESTAMP
    WHERE user_id = p_user_id;
END //
DELIMITER ;

-- Record user interaction with Elo score update
DROP PROCEDURE IF EXISTS record_user_interaction;
DELIMITER //
CREATE PROCEDURE record_user_interaction(
    IN p_user_id INT,
    IN p_movie_1_id INT,
    IN p_movie_2_id INT,
    IN p_chosen_movie_id INT,
    IN p_session_id VARCHAR(100)
)
BEGIN
    DECLARE v_rejected_movie_id INT;
    DECLARE v_elo_winner INT;
    DECLARE v_elo_loser INT;
    DECLARE v_expected_score DECIMAL(10,6);
    DECLARE v_k_factor INT DEFAULT 32;
    DECLARE v_new_elo_winner INT;
    DECLARE v_new_elo_loser INT;
    
    IF p_chosen_movie_id = p_movie_1_id THEN
        SET v_rejected_movie_id = p_movie_2_id;
    ELSE
        SET v_rejected_movie_id = p_movie_1_id;
    END IF;
    
    INSERT INTO user_interactions (
        user_id, movie_1_id, movie_2_id, 
        chosen_movie_id, rejected_movie_id, session_id
    ) VALUES (
        p_user_id, p_movie_1_id, p_movie_2_id,
        p_chosen_movie_id, v_rejected_movie_id, p_session_id
    );
    
    SELECT elo_score INTO v_elo_winner FROM movies WHERE movie_id = p_chosen_movie_id;
    SELECT elo_score INTO v_elo_loser FROM movies WHERE movie_id = v_rejected_movie_id;
    
    SET v_expected_score = 1.0 / (1.0 + POWER(10, (v_elo_loser - v_elo_winner) / 400.0));
    
    SET v_new_elo_winner = v_elo_winner + ROUND(v_k_factor * (1 - v_expected_score));
    SET v_new_elo_loser = v_elo_loser + ROUND(v_k_factor * (0 - (1 - v_expected_score)));
    
    UPDATE movies 
    SET elo_score = v_new_elo_winner,
        comparison_count = comparison_count + 1
    WHERE movie_id = p_chosen_movie_id;
    
    UPDATE movies 
    SET elo_score = v_new_elo_loser,
        comparison_count = comparison_count + 1
    WHERE movie_id = v_rejected_movie_id;
    
    CALL update_user_interaction_count(p_user_id);
END //
DELIMITER ;

-- Update user preferences based on interactions
DROP PROCEDURE IF EXISTS update_user_preferences;
DELIMITER //
CREATE PROCEDURE update_user_preferences(IN p_user_id INT)
BEGIN
    DECLARE v_favorite_genre VARCHAR(50);
    DECLARE v_favorite_actor VARCHAR(100);
    DECLARE v_favorite_director VARCHAR(100);
    DECLARE v_avg_rating DECIMAL(3,1);
    DECLARE v_preferred_decade INT;
    DECLARE v_total_interactions INT;
    
    SELECT g.genre_name INTO v_favorite_genre
    FROM user_interactions ui
    INNER JOIN movie_genres mg ON ui.chosen_movie_id = mg.movie_id
    INNER JOIN genres g ON mg.genre_id = g.genre_id
    WHERE ui.user_id = p_user_id
    GROUP BY g.genre_name
    ORDER BY COUNT(*) DESC
    LIMIT 1;
    
    SELECT a.actor_name INTO v_favorite_actor
    FROM user_interactions ui
    INNER JOIN movie_actors ma ON ui.chosen_movie_id = ma.movie_id
    INNER JOIN actors a ON ma.actor_id = a.actor_id
    WHERE ui.user_id = p_user_id
    GROUP BY a.actor_name
    ORDER BY COUNT(*) DESC
    LIMIT 1;
    
    SELECT d.director_name INTO v_favorite_director
    FROM user_interactions ui
    INNER JOIN movie_directors md ON ui.chosen_movie_id = md.movie_id
    INNER JOIN directors d ON md.director_id = d.director_id
    WHERE ui.user_id = p_user_id
    GROUP BY d.director_name
    ORDER BY COUNT(*) DESC
    LIMIT 1;
    
    SELECT AVG(m.tmdb_rating) INTO v_avg_rating
    FROM user_interactions ui
    INNER JOIN movies m ON ui.chosen_movie_id = m.movie_id
    WHERE ui.user_id = p_user_id;
    
    SELECT FLOOR(AVG(m.release_year) / 10) * 10 INTO v_preferred_decade
    FROM user_interactions ui
    INNER JOIN movies m ON ui.chosen_movie_id = m.movie_id
    WHERE ui.user_id = p_user_id;
    
    SELECT COUNT(*) INTO v_total_interactions
    FROM user_interactions
    WHERE user_id = p_user_id;
    
    INSERT INTO user_preferences (
        user_id, favorite_genre, favorite_actor, favorite_director,
        avg_rating_preference, preferred_decade, total_interactions
    ) VALUES (
        p_user_id, v_favorite_genre, v_favorite_actor, v_favorite_director,
        v_avg_rating, v_preferred_decade, v_total_interactions
    ) ON DUPLICATE KEY UPDATE
        favorite_genre = v_favorite_genre,
        favorite_actor = v_favorite_actor,
        favorite_director = v_favorite_director,
        avg_rating_preference = v_avg_rating,
        preferred_decade = v_preferred_decade,
        total_interactions = v_total_interactions,
        last_updated = CURRENT_TIMESTAMP;
END //
DELIMITER ;

-- Get personalized movie recommendations using cursor
DROP PROCEDURE IF EXISTS get_personalized_recommendations;
DELIMITER //
CREATE PROCEDURE get_personalized_recommendations(
    IN p_user_id INT,
    IN p_limit INT
)
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE v_movie_id INT;
    DECLARE v_score DECIMAL(10,2);
    DECLARE v_favorite_genre VARCHAR(50);
    
    DECLARE movie_cursor CURSOR FOR
        SELECT DISTINCT m.movie_id,
            (
                (CASE WHEN FIND_IN_SET(v_favorite_genre, 
                    (SELECT GROUP_CONCAT(g.genre_name) 
                     FROM movie_genres mg 
                     INNER JOIN genres g ON mg.genre_id = g.genre_id 
                     WHERE mg.movie_id = m.movie_id)) > 0 
                THEN 50 ELSE 0 END) +
                (m.popularity / 100) +
                (m.tmdb_rating * 5) +
                ((m.elo_score - 1500) / 10) +
                (CASE WHEN m.release_year >= YEAR(CURRENT_DATE) - 5 THEN 10 ELSE 0 END)
            ) AS relevance_score
        FROM movies m
        WHERE m.movie_id NOT IN (
            SELECT chosen_movie_id FROM user_interactions WHERE user_id = p_user_id
            UNION
            SELECT rejected_movie_id FROM user_interactions WHERE user_id = p_user_id
        )
        AND m.tmdb_rating IS NOT NULL
        ORDER BY relevance_score DESC
        LIMIT p_limit;
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
    
    SELECT favorite_genre INTO v_favorite_genre
    FROM user_preferences
    WHERE user_id = p_user_id;
    
    CREATE TEMPORARY TABLE IF NOT EXISTS temp_recommendations (
        movie_id INT,
        relevance_score DECIMAL(10,2),
        PRIMARY KEY (movie_id)
    );
    
    DELETE FROM temp_recommendations;
    
    OPEN movie_cursor;
    
    read_loop: LOOP
        FETCH movie_cursor INTO v_movie_id, v_score;
        IF done THEN
            LEAVE read_loop;
        END IF;
        
        INSERT INTO temp_recommendations (movie_id, relevance_score)
        VALUES (v_movie_id, v_score);
    END LOOP;
    
    CLOSE movie_cursor;
    
    SELECT m.*, tr.relevance_score
    FROM temp_recommendations tr
    INNER JOIN comprehensive_movie_view m ON tr.movie_id = m.movie_id
    ORDER BY tr.relevance_score DESC;
    
    DROP TEMPORARY TABLE IF EXISTS temp_recommendations;
END //
DELIMITER ;

-- Search movies with semantic/storyline support
DROP PROCEDURE IF EXISTS search_movies_advanced;
DELIMITER //
CREATE PROCEDURE search_movies_advanced(
    IN p_search_query VARCHAR(500),
    IN p_search_type VARCHAR(20),
    IN p_user_id INT,
    IN p_limit INT
)
BEGIN
    DECLARE v_search_lower VARCHAR(500);
    
    SET v_search_lower = LOWER(p_search_query);
    
    INSERT INTO search_history (user_id, search_query, search_type)
    VALUES (p_user_id, p_search_query, p_search_type);
    
    IF p_search_type = 'storyline' OR p_search_type = 'semantic' THEN
        SELECT DISTINCT m.*, 
            (
                (CASE WHEN LOWER(m.title) = v_search_lower THEN 1000 ELSE 0 END) +
                (CASE WHEN LOWER(m.title) LIKE CONCAT(v_search_lower, '%') THEN 500 ELSE 0 END) +
                (CASE WHEN LOWER(m.title) LIKE CONCAT('%', v_search_lower, '%') THEN 250 ELSE 0 END) +
                (CASE WHEN LOWER(m.overview) LIKE CONCAT('%', v_search_lower, '%') THEN 200 ELSE 0 END) +
                (CASE WHEN LOWER(m.keywords) LIKE CONCAT('%', v_search_lower, '%') THEN 150 ELSE 0 END) +
                (m.popularity / 10) +
                (m.tmdb_rating * 10)
            ) AS relevance_score
        FROM comprehensive_movie_view m
        WHERE LOWER(m.title) LIKE CONCAT('%', v_search_lower, '%')
           OR LOWER(m.overview) LIKE CONCAT('%', v_search_lower, '%')
           OR LOWER(m.keywords) LIKE CONCAT('%', v_search_lower, '%')
           OR LOWER(m.genres) LIKE CONCAT('%', v_search_lower, '%')
           OR LOWER(m.cast) LIKE CONCAT('%', v_search_lower, '%')
           OR LOWER(m.directors) LIKE CONCAT('%', v_search_lower, '%')
        ORDER BY relevance_score DESC, m.popularity DESC
        LIMIT p_limit;
    ELSE
        SELECT DISTINCT m.*,
            (
                (CASE WHEN LOWER(m.title) = v_search_lower THEN 1000 ELSE 0 END) +
                (CASE WHEN LOWER(m.title) LIKE CONCAT(v_search_lower, '%') THEN 500 ELSE 0 END) +
                (CASE WHEN LOWER(m.title) LIKE CONCAT('%', v_search_lower, '%') THEN 250 ELSE 0 END)
            ) AS relevance_score
        FROM comprehensive_movie_view m
        WHERE LOWER(m.title) LIKE CONCAT('%', v_search_lower, '%')
        ORDER BY relevance_score DESC, m.popularity DESC
        LIMIT p_limit;
    END IF;
    
    UPDATE search_history
    SET results_count = (SELECT FOUND_ROWS())
    WHERE search_id = LAST_INSERT_ID();
END //
DELIMITER ;

-- Update cache statistics
DROP PROCEDURE IF EXISTS update_cache_stats;
DELIMITER //
CREATE PROCEDURE update_cache_stats(
    IN p_cache_type VARCHAR(20),
    IN p_hits INT,
    IN p_misses INT,
    IN p_response_time_ms DECIMAL(10,2),
    IN p_memory_usage_kb INT
)
BEGIN
    INSERT INTO cache_stats (
        stat_date, cache_type, cache_hits, cache_misses,
        avg_response_time_ms, memory_usage_kb
    ) VALUES (
        CURRENT_DATE, p_cache_type, p_hits, p_misses,
        p_response_time_ms, p_memory_usage_kb
    ) ON DUPLICATE KEY UPDATE
        cache_hits = cache_hits + p_hits,
        cache_misses = cache_misses + p_misses,
        avg_response_time_ms = (avg_response_time_ms + p_response_time_ms) / 2,
        memory_usage_kb = p_memory_usage_kb,
        recorded_at = CURRENT_TIMESTAMP;
END //
DELIMITER ;

-- Get comparison pair for user (smart selection)
DROP PROCEDURE IF EXISTS get_comparison_pair;
DELIMITER //
CREATE PROCEDURE get_comparison_pair(IN p_user_id INT)
BEGIN
    DECLARE v_interaction_count INT;
    DECLARE v_favorite_genre VARCHAR(50);
    
    SELECT interaction_count INTO v_interaction_count
    FROM users
    WHERE user_id = p_user_id;
    
    SELECT favorite_genre INTO v_favorite_genre
    FROM user_preferences
    WHERE user_id = p_user_id;
    
    IF v_interaction_count < 10 THEN
        SELECT m.* FROM comprehensive_movie_view m
        WHERE m.tmdb_rating >= 7.0
        AND m.popularity > 50
        AND m.movie_id NOT IN (
            SELECT chosen_movie_id FROM user_interactions WHERE user_id = p_user_id
            UNION
            SELECT rejected_movie_id FROM user_interactions WHERE user_id = p_user_id
        )
        ORDER BY RAND()
        LIMIT 2;
    ELSE
        (
            SELECT m.* FROM comprehensive_movie_view m
            WHERE FIND_IN_SET(v_favorite_genre, m.genres) > 0
            AND m.movie_id NOT IN (
                SELECT chosen_movie_id FROM user_interactions WHERE user_id = p_user_id
                UNION
                SELECT rejected_movie_id FROM user_interactions WHERE user_id = p_user_id
            )
            ORDER BY RAND()
            LIMIT 1
        )
        UNION ALL
        (
            SELECT m.* FROM comprehensive_movie_view m
            WHERE (v_favorite_genre IS NULL OR FIND_IN_SET(v_favorite_genre, m.genres) = 0)
            AND m.movie_id NOT IN (
                SELECT chosen_movie_id FROM user_interactions WHERE user_id = p_user_id
                UNION
                SELECT rejected_movie_id FROM user_interactions WHERE user_id = p_user_id
            )
            ORDER BY m.popularity DESC, RAND()
            LIMIT 1
        );
    END IF;
END //
DELIMITER ;

-- Update movie ELO scores (standalone procedure for direct Elo updates)
DROP PROCEDURE IF EXISTS update_movie_elo;
DELIMITER //
CREATE PROCEDURE update_movie_elo(
    IN p_winner_id INT,
    IN p_loser_id INT,
    IN p_k_factor INT
)
BEGIN
    DECLARE v_elo_winner INT;
    DECLARE v_elo_loser INT;
    DECLARE v_expected_score DECIMAL(10,6);
    DECLARE v_new_elo_winner INT;
    DECLARE v_new_elo_loser INT;
    
    SELECT elo_score INTO v_elo_winner FROM movies WHERE movie_id = p_winner_id;
    SELECT elo_score INTO v_elo_loser FROM movies WHERE movie_id = p_loser_id;
    
    SET v_expected_score = 1.0 / (1.0 + POWER(10, (v_elo_loser - v_elo_winner) / 400.0));
    
    SET v_new_elo_winner = v_elo_winner + ROUND(p_k_factor * (1 - v_expected_score));
    SET v_new_elo_loser = v_elo_loser + ROUND(p_k_factor * (0 - (1 - v_expected_score)));
    
    UPDATE movies 
    SET elo_score = v_new_elo_winner,
        comparison_count = comparison_count + 1
    WHERE movie_id = p_winner_id;
    
    UPDATE movies 
    SET elo_score = v_new_elo_loser,
        comparison_count = comparison_count + 1
    WHERE movie_id = p_loser_id;
END //
DELIMITER ;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Calculate user similarity score between two users
DROP FUNCTION IF EXISTS calculate_user_similarity;
DELIMITER //
CREATE FUNCTION calculate_user_similarity(
    p_user1_id INT,
    p_user2_id INT
) RETURNS DECIMAL(5,2)
DETERMINISTIC
READS SQL DATA
BEGIN
    DECLARE v_common_choices INT;
    DECLARE v_total_choices INT;
    DECLARE v_similarity DECIMAL(5,2);
    
    SELECT COUNT(DISTINCT ui1.chosen_movie_id) INTO v_common_choices
    FROM user_interactions ui1
    INNER JOIN user_interactions ui2
        ON ui1.chosen_movie_id = ui2.chosen_movie_id
    WHERE ui1.user_id = p_user1_id
      AND ui2.user_id = p_user2_id;
    
    SELECT COUNT(DISTINCT chosen_movie_id) INTO v_total_choices
    FROM user_interactions
    WHERE user_id IN (p_user1_id, p_user2_id);
    
    IF v_total_choices > 0 THEN
        SET v_similarity = (v_common_choices / v_total_choices) * 100;
    ELSE
        SET v_similarity = 0;
    END IF;
    
    RETURN v_similarity;
END //
DELIMITER ;

-- Get movie diversity score (based on genre variety)
DROP FUNCTION IF EXISTS get_movie_diversity_score;
DELIMITER //
CREATE FUNCTION get_movie_diversity_score(p_movie_id INT)
RETURNS INT
DETERMINISTIC
READS SQL DATA
BEGIN
    DECLARE v_genre_count INT;
    
    SELECT COUNT(DISTINCT genre_id) INTO v_genre_count
    FROM movie_genres
    WHERE movie_id = p_movie_id;
    
    RETURN v_genre_count;
END //
DELIMITER ;

-- Check if user has watched movie (based on interactions)
DROP FUNCTION IF EXISTS has_user_interacted_with_movie;
DELIMITER //
CREATE FUNCTION has_user_interacted_with_movie(
    p_user_id INT,
    p_movie_id INT
) RETURNS BOOLEAN
DETERMINISTIC
READS SQL DATA
BEGIN
    DECLARE v_count INT;
    
    SELECT COUNT(*) INTO v_count
    FROM user_interactions
    WHERE user_id = p_user_id
    AND (chosen_movie_id = p_movie_id OR rejected_movie_id = p_movie_id);
    
    RETURN v_count > 0;
END //
DELIMITER ;

-- Calculate cache hit rate
DROP FUNCTION IF EXISTS calculate_cache_hit_rate;
DELIMITER //
CREATE FUNCTION calculate_cache_hit_rate(
    p_cache_type VARCHAR(20)
) RETURNS DECIMAL(5,2)
DETERMINISTIC
READS SQL DATA
BEGIN
    DECLARE v_hit_rate DECIMAL(5,2);
    
    SELECT 
        CASE 
            WHEN (cache_hits + cache_misses) > 0 
            THEN (cache_hits / (cache_hits + cache_misses)) * 100
            ELSE 0
        END INTO v_hit_rate
    FROM cache_stats
    WHERE cache_type = p_cache_type
    AND stat_date = CURRENT_DATE;
    
    RETURN COALESCE(v_hit_rate, 0);
END //
DELIMITER ;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Trigger: Auto-update user preferences after interaction
DROP TRIGGER IF EXISTS after_interaction_insert;
DELIMITER //
CREATE TRIGGER after_interaction_insert
AFTER INSERT ON user_interactions
FOR EACH ROW
BEGIN
    UPDATE users
    SET interaction_count = interaction_count + 1,
        last_active = CURRENT_TIMESTAMP
    WHERE user_id = NEW.user_id;
    
    IF (SELECT interaction_count FROM users WHERE user_id = NEW.user_id) % 5 = 0 THEN
        CALL update_user_preferences(NEW.user_id);
    END IF;
END //
DELIMITER ;

-- Trigger: Validate movie rating before insert
DROP TRIGGER IF EXISTS before_movie_insert_validate;
DELIMITER //
CREATE TRIGGER before_movie_insert_validate
BEFORE INSERT ON movies
FOR EACH ROW
BEGIN
    IF NEW.tmdb_rating IS NOT NULL AND (NEW.tmdb_rating < 0 OR NEW.tmdb_rating > 10) THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'TMDB rating must be between 0 and 10';
    END IF;
    
    IF NEW.elo_score IS NULL THEN
        SET NEW.elo_score = 1500;
    END IF;
    
    IF NEW.comparison_count IS NULL THEN
        SET NEW.comparison_count = 0;
    END IF;
END //
DELIMITER ;

-- Trigger: Validate movie rating before update
DROP TRIGGER IF EXISTS before_movie_update_validate;
DELIMITER //
CREATE TRIGGER before_movie_update_validate
BEFORE UPDATE ON movies
FOR EACH ROW
BEGIN
    IF NEW.tmdb_rating IS NOT NULL AND (NEW.tmdb_rating < 0 OR NEW.tmdb_rating > 10) THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'TMDB rating must be between 0 and 10';
    END IF;
END //
DELIMITER ;

-- Trigger: Auto-create user preferences on user creation
DROP TRIGGER IF EXISTS after_user_insert;
DELIMITER //
CREATE TRIGGER after_user_insert
AFTER INSERT ON users
FOR EACH ROW
BEGIN
    INSERT INTO user_preferences (user_id, total_interactions)
    VALUES (NEW.user_id, 0);
END //
DELIMITER ;

-- Trigger: Record search in search history
DROP TRIGGER IF EXISTS after_search_history_insert;
DELIMITER //
CREATE TRIGGER after_search_history_insert
AFTER INSERT ON search_history
FOR EACH ROW
BEGIN
    SET @last_search_id = NEW.search_id;
END //
DELIMITER ;

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

CREATE INDEX idx_movies_rating_popularity ON movies(tmdb_rating DESC, popularity DESC);
CREATE INDEX idx_movies_year_rating ON movies(release_year DESC, tmdb_rating DESC);
CREATE INDEX idx_user_interactions_user_timestamp ON user_interactions(user_id, timestamp DESC);
CREATE INDEX idx_recommendation_log_user_movie ON recommendation_log(user_id, movie_id);

-- Full-text search index
ALTER TABLE movies ADD FULLTEXT INDEX ft_title_overview (title, overview);

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================
SELECT 'CineSense comprehensive database schema created successfully!' AS Status;
SELECT 'Tables: 16 | Views: 13 | Procedures: 8 | Functions: 4 | Triggers: 5 | CHECK Constraints: 9' AS Summary;

-- ============================================================================
-- EXTENDED TABLES (AI Features: Semantic Search, Mood, A/B Testing)
-- ============================================================================

CREATE TABLE IF NOT EXISTS keywords (
    keyword_id INT AUTO_INCREMENT PRIMARY KEY,
    keyword VARCHAR(100) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_keyword (keyword),
    FULLTEXT INDEX idx_fulltext_keyword (keyword)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS semantic_embeddings (
    movie_id INT PRIMARY KEY,
    embedding_vector BLOB NOT NULL,
    model_version VARCHAR(50) DEFAULT 'mpnet-base-v2',
    embedding_dim INT DEFAULT 768,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS search_logs (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    query TEXT NOT NULL,
    search_type ENUM('semantic', 'keyword', 'hybrid') DEFAULT 'hybrid',
    results_count INT DEFAULT 0,
    top_result_id INT,
    clicked_result_id INT,
    response_time_ms INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL,
    FOREIGN KEY (top_result_id) REFERENCES movies(movie_id) ON DELETE SET NULL,
    FOREIGN KEY (clicked_result_id) REFERENCES movies(movie_id) ON DELETE SET NULL,
    INDEX idx_user_search (user_id, created_at),
    INDEX idx_query (query(100))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS user_moods (
    mood_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    mood VARCHAR(50) NOT NULL,
    detected_from TEXT,
    confidence_score DECIMAL(3,2) DEFAULT 1.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX idx_user_mood (user_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS recommendation_feedback (
    feedback_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    movie_id INT NOT NULL,
    recommendation_type VARCHAR(50),
    feedback_type ENUM('thumbs_up', 'thumbs_down', 'not_interested', 'watched') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    INDEX idx_user_feedback (user_id, created_at),
    INDEX idx_movie_feedback (movie_id),
    INDEX idx_feedback_type_date (recommendation_type, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS ab_test_assignments (
    assignment_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    experiment_name VARCHAR(100) NOT NULL,
    variant VARCHAR(50) NOT NULL,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    UNIQUE KEY unique_user_experiment (user_id, experiment_name),
    INDEX idx_experiment (experiment_name, variant)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS ab_test_metrics (
    metric_id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_name VARCHAR(100) NOT NULL,
    variant VARCHAR(50) NOT NULL,
    user_id INT NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    value DECIMAL(10,4) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX idx_ab_experiment_variant (experiment_name, variant, metric_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS model_versions (
    version_id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50),
    rmse DECIMAL(6,4),
    mae DECIMAL(6,4),
    training_date TIMESTAMP,
    deployed_at TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE,
    model_path TEXT,
    hyperparameters JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_active_model (model_name, is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- EXTENDED VIEWS (AI Features)
-- ============================================================================

DROP VIEW IF EXISTS movie_search_index;
CREATE VIEW movie_search_index AS
SELECT
    m.movie_id, m.title, m.overview, m.release_year,
    GROUP_CONCAT(DISTINCT g.genre_name ORDER BY g.genre_name) AS genres,
    GROUP_CONCAT(DISTINCT d.director_name ORDER BY d.director_name) AS directors,
    GROUP_CONCAT(DISTINCT a.actor_name ORDER BY ma.cast_order) AS cast_members,
    GROUP_CONCAT(DISTINCT mk.keyword ORDER BY mk.keyword) AS keywords,
    m.popularity, m.tmdb_rating
FROM movies m
LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
LEFT JOIN genres g ON mg.genre_id = g.genre_id
LEFT JOIN movie_directors md ON m.movie_id = md.movie_id
LEFT JOIN directors d ON md.director_id = d.director_id
LEFT JOIN movie_actors ma ON m.movie_id = ma.movie_id
LEFT JOIN actors a ON ma.actor_id = a.actor_id
LEFT JOIN movie_keywords mk ON m.movie_id = mk.movie_id
GROUP BY m.movie_id;

DROP VIEW IF EXISTS search_analytics;
CREATE VIEW search_analytics AS
SELECT
    DATE(created_at) AS search_date, search_type,
    COUNT(*) AS total_searches,
    AVG(results_count) AS avg_results,
    AVG(response_time_ms) AS avg_response_time_ms,
    COUNT(DISTINCT user_id) AS unique_users,
    ROUND(100.0 * SUM(CASE WHEN clicked_result_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS click_through_rate
FROM search_logs
GROUP BY DATE(created_at), search_type;

DROP VIEW IF EXISTS user_mood_trends;
CREATE VIEW user_mood_trends AS
SELECT
    user_id, mood,
    COUNT(*) AS mood_frequency,
    AVG(confidence_score) AS avg_confidence,
    MIN(created_at) AS first_occurrence,
    MAX(created_at) AS last_occurrence
FROM user_moods
GROUP BY user_id, mood;

DROP VIEW IF EXISTS recommendation_performance;
CREATE VIEW recommendation_performance AS
SELECT
    recommendation_type,
    COUNT(*) AS total_recommendations,
    SUM(CASE WHEN feedback_type = 'thumbs_up'   THEN 1 ELSE 0 END) AS positive_feedback,
    SUM(CASE WHEN feedback_type = 'thumbs_down' THEN 1 ELSE 0 END) AS negative_feedback,
    SUM(CASE WHEN feedback_type = 'watched'     THEN 1 ELSE 0 END) AS movies_watched,
    ROUND(100.0 * SUM(CASE WHEN feedback_type = 'thumbs_up' THEN 1 ELSE 0 END) / COUNT(*), 2) AS satisfaction_rate
FROM recommendation_feedback
GROUP BY recommendation_type;

-- ============================================================================
-- EXTENDED STORED PROCEDURES (AI Features)
-- ============================================================================

DELIMITER //

DROP PROCEDURE IF EXISTS get_mood_recommendations //
CREATE PROCEDURE get_mood_recommendations(
    IN user_id_param INT,
    IN mood_param VARCHAR(50),
    IN limit_param INT
)
BEGIN
    DECLARE genre_list VARCHAR(500);
    SET genre_list = CASE mood_param
        WHEN 'happy'       THEN 'Comedy,Animation,Family,Musical'
        WHEN 'sad'         THEN 'Drama,Romance'
        WHEN 'stressed'    THEN 'Comedy,Animation'
        WHEN 'bored'       THEN 'Action,Adventure,Thriller,Sci-Fi'
        WHEN 'thoughtful'  THEN 'Drama,Documentary,Mystery,Sci-Fi'
        WHEN 'romantic'    THEN 'Romance,Drama,Comedy'
        WHEN 'adventurous' THEN 'Adventure,Action,Fantasy'
        ELSE 'Action,Comedy,Drama'
    END;
    INSERT INTO user_moods (user_id, mood, detected_from, created_at)
        VALUES (user_id_param, mood_param, 'recommendation_request', NOW());
    SELECT m.movie_id, m.title, m.overview, m.poster_path, m.tmdb_rating,
           GROUP_CONCAT(DISTINCT g.genre_name) AS genres,
           COUNT(DISTINCT g.genre_id) AS genre_match_count
    FROM movies m
    INNER JOIN movie_genres mg ON m.movie_id = mg.movie_id
    INNER JOIN genres g ON mg.genre_id = g.genre_id
    WHERE FIND_IN_SET(g.genre_name, genre_list) > 0
    GROUP BY m.movie_id
    ORDER BY genre_match_count DESC, m.tmdb_rating DESC
    LIMIT limit_param;
END //

DROP PROCEDURE IF EXISTS update_semantic_embedding //
CREATE PROCEDURE update_semantic_embedding(
    IN movie_id_param INT,
    IN embedding_data BLOB,
    IN model_ver VARCHAR(50),
    IN embed_dim INT
)
BEGIN
    INSERT INTO semantic_embeddings (movie_id, embedding_vector, model_version, embedding_dim)
    VALUES (movie_id_param, embedding_data, model_ver, embed_dim)
    ON DUPLICATE KEY UPDATE
        embedding_vector = embedding_data,
        model_version = model_ver,
        embedding_dim = embed_dim,
        updated_at = NOW();
END //

DROP PROCEDURE IF EXISTS log_recommendation_feedback //
CREATE PROCEDURE log_recommendation_feedback(
    IN user_id_param INT,
    IN movie_id_param INT,
    IN rec_type VARCHAR(50),
    IN feedback VARCHAR(50)
)
BEGIN
    INSERT INTO recommendation_feedback (user_id, movie_id, recommendation_type, feedback_type, created_at)
    VALUES (user_id_param, movie_id_param, rec_type, feedback, NOW());
    IF feedback = 'thumbs_up' THEN
        UPDATE movies SET popularity = popularity + 0.1 WHERE movie_id = movie_id_param;
    ELSEIF feedback = 'thumbs_down' THEN
        UPDATE movies SET popularity = GREATEST(popularity - 0.05, 0) WHERE movie_id = movie_id_param;
    END IF;
END //

DROP PROCEDURE IF EXISTS assign_ab_test_variant //
CREATE PROCEDURE assign_ab_test_variant(
    IN user_id_param INT,
    IN experiment_param VARCHAR(100)
)
BEGIN
    DECLARE assigned_variant VARCHAR(50);
    SELECT variant INTO assigned_variant
    FROM ab_test_assignments
    WHERE user_id = user_id_param AND experiment_name = experiment_param;
    IF assigned_variant IS NULL THEN
        SET assigned_variant = CASE
            WHEN MOD(user_id_param, 100) < 50 THEN 'control'
            ELSE 'treatment'
        END;
        INSERT INTO ab_test_assignments (user_id, experiment_name, variant)
        VALUES (user_id_param, experiment_param, assigned_variant);
    END IF;
    SELECT assigned_variant AS variant;
END //

DELIMITER ;



INSERT IGNORE INTO keywords (keyword) VALUES
('thriller'),('action'),('comedy'),('romance'),('drama'),
('sci-fi'),('fantasy'),('horror'),('mystery'),('adventure'),
('spy'),('time-travel'),('superhero'),('war'),('historical'),
('heist'),('revenge'),('survival'),('dystopian'),('space');

SELECT 'Extended AI feature tables created successfully!' AS Status;


DROP VIEW IF EXISTS user_review_summary;
CREATE VIEW user_review_summary AS
SELECT
    u.user_id, u.username,
    COUNT(r.review_id)       AS total_reviews,
    ROUND(AVG(r.rating), 2)  AS avg_given_rating,
    MAX(r.rating)            AS highest_rating,
    MIN(r.rating)            AS lowest_rating,
    SUM(r.helpful_votes)     AS total_helpful_votes,
    SUM(CASE WHEN r.is_spoiler THEN 1 ELSE 0 END) AS spoiler_reviews
FROM users u
LEFT JOIN movie_reviews r ON u.user_id = r.user_id
GROUP BY u.user_id;

-- View: watchlist with movie and review data
DROP VIEW IF EXISTS watchlist_details;
CREATE VIEW watchlist_details AS
SELECT
    w.watchlist_id, w.user_id, u.username, w.movie_id,
    m.title, m.release_year, m.tmdb_rating, m.poster_path,
    w.status, w.priority, w.user_rating, w.personal_note,
    w.added_at, w.watched_at,
    r.review_text,
    DATEDIFF(CURRENT_DATE, w.added_at) AS days_on_watchlist
FROM watchlist w
INNER JOIN users  u ON w.user_id  = u.user_id
INNER JOIN movies m ON w.movie_id = m.movie_id
LEFT  JOIN movie_reviews r ON r.user_id = w.user_id AND r.movie_id = w.movie_id;

-- View: genre head-to-head comparison statistics
DROP VIEW IF EXISTS genre_comparison_stats;
CREATE VIEW genre_comparison_stats AS
SELECT
    g.genre_id, g.genre_name,
    COUNT(DISTINCT mg.movie_id) AS total_movies,
    SUM(CASE WHEN ui.chosen_movie_id  = mg.movie_id THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN ui.rejected_movie_id= mg.movie_id THEN 1 ELSE 0 END) AS losses,
    ROUND(
        SUM(CASE WHEN ui.chosen_movie_id = mg.movie_id THEN 1 ELSE 0 END)
        / NULLIF(
            SUM(CASE WHEN ui.chosen_movie_id  = mg.movie_id THEN 1 ELSE 0 END) +
            SUM(CASE WHEN ui.rejected_movie_id= mg.movie_id THEN 1 ELSE 0 END)
          , 0) * 100, 2
    ) AS win_rate_pct
FROM genres g
INNER JOIN movie_genres mg ON g.genre_id = mg.genre_id
LEFT  JOIN user_interactions ui
    ON mg.movie_id IN (ui.chosen_movie_id, ui.rejected_movie_id)
GROUP BY g.genre_id;

-- View: recommendation effectiveness
DROP VIEW IF EXISTS recommendation_effectiveness;
CREATE VIEW recommendation_effectiveness AS
SELECT
    m.movie_id, m.title, m.tmdb_rating,
    COUNT(rl.log_id) AS times_recommended,
    SUM(rl.was_clicked) AS total_clicks,
    SUM(rl.was_watched) AS total_watches,
    ROUND(SUM(rl.was_clicked) / NULLIF(COUNT(rl.log_id), 0) * 100, 2) AS ctr_pct,
    ROUND(SUM(rl.was_watched) / NULLIF(COUNT(rl.log_id), 0) * 100, 2) AS watch_rate_pct,
    AVG(rl.recommendation_score) AS avg_score
FROM movies m
INNER JOIN recommendation_log rl ON m.movie_id = rl.movie_id
GROUP BY m.movie_id
HAVING times_recommended >= 1;

-- ── Advanced Functions ───────────────────────────────────────────────────────

-- Function: classify movie popularity tier
DROP FUNCTION IF EXISTS get_popularity_tier;
DELIMITER //
CREATE FUNCTION get_popularity_tier(p_popularity DECIMAL(10,3))
RETURNS VARCHAR(20)
DETERMINISTIC NO SQL
BEGIN
    IF p_popularity IS NULL THEN RETURN 'Unknown';
    ELSEIF p_popularity >= 500 THEN RETURN 'Blockbuster';
    ELSEIF p_popularity >= 200 THEN RETURN 'Popular';
    ELSEIF p_popularity >= 50 THEN RETURN 'Moderate';
    ELSE RETURN 'Niche';
    END IF;
END //
DELIMITER ;

-- Function: Jaccard genre similarity between two movies
DROP FUNCTION IF EXISTS jaccard_genre_similarity;
DELIMITER //
CREATE FUNCTION jaccard_genre_similarity(p_movie_a INT, p_movie_b INT)
RETURNS DECIMAL(5,4) DETERMINISTIC READS SQL DATA
BEGIN
    DECLARE v_intersection INT DEFAULT 0;
    DECLARE v_union_count  INT DEFAULT 0;
    SELECT COUNT(*) INTO v_intersection
    FROM movie_genres mga
    INNER JOIN movie_genres mgb
        ON mga.genre_id = mgb.genre_id
    WHERE mga.movie_id = p_movie_a
      AND mgb.movie_id = p_movie_b;
    SELECT COUNT(*) INTO v_union_count FROM (
        SELECT genre_id FROM movie_genres WHERE movie_id = p_movie_a
        UNION
        SELECT genre_id FROM movie_genres WHERE movie_id = p_movie_b
    ) AS combined;
    IF v_union_count = 0 THEN RETURN 0.0000; END IF;
    RETURN ROUND(v_intersection / v_union_count, 4);
END //
DELIMITER ;

-- Function: get user taste profile label
DROP FUNCTION IF EXISTS get_user_taste_profile;
DELIMITER //
CREATE FUNCTION get_user_taste_profile(p_user_id INT)
RETURNS VARCHAR(50) DETERMINISTIC READS SQL DATA
BEGIN
    DECLARE v_avg_rating DECIMAL(3,1);
    DECLARE v_fav_genre  VARCHAR(50);
    DECLARE v_interaction INT;
    DECLARE EXIT HANDLER FOR SQLEXCEPTION RETURN 'Profile Unavailable';
    SELECT avg_rating_preference, favorite_genre, total_interactions
    INTO v_avg_rating, v_fav_genre, v_interaction
    FROM user_preferences WHERE user_id = p_user_id;
    IF v_interaction IS NULL OR v_interaction = 0 THEN RETURN 'New Explorer'; END IF;
    RETURN CONCAT(
        CASE WHEN v_avg_rating >= 8.0 THEN 'Purist'
             WHEN v_avg_rating >= 6.0 THEN 'Balanced' ELSE 'Lenient' END,
        ' ', COALESCE(v_fav_genre, 'Mixed'), ' Fan');
END //
DELIMITER ;

-- Function: weighted recommendation score for user-movie pair
DROP FUNCTION IF EXISTS weighted_recommendation_score;
DELIMITER //
CREATE FUNCTION weighted_recommendation_score(p_user_id INT, p_movie_id INT)
RETURNS DECIMAL(10,4) DETERMINISTIC READS SQL DATA
BEGIN
    DECLARE v_elo DECIMAL(10,2) DEFAULT 1500;
    DECLARE v_popularity DECIMAL(10,3) DEFAULT 0;
    DECLARE v_rating DECIMAL(3,1) DEFAULT 0;
    DECLARE v_genre_match INT DEFAULT 0;
    DECLARE v_fav_genre VARCHAR(50) DEFAULT NULL;
    DECLARE EXIT HANDLER FOR SQLEXCEPTION RETURN -1.0000;
    SELECT elo_score, popularity, tmdb_rating INTO v_elo, v_popularity, v_rating
    FROM movies WHERE movie_id = p_movie_id;
    SELECT favorite_genre INTO v_fav_genre FROM user_preferences WHERE user_id = p_user_id;
    IF v_fav_genre IS NOT NULL THEN
        SELECT COUNT(*) INTO v_genre_match FROM movie_genres mg
        INNER JOIN genres g ON mg.genre_id = g.genre_id
        WHERE mg.movie_id = p_movie_id AND g.genre_name = v_fav_genre;
    END IF;
    RETURN ROUND(
        (0.40 * ((v_elo - 1500) / 10)) +
        (0.30 * (v_rating * 10)) +
        (0.20 * LEAST(v_popularity, 500) / 5) +
        (0.10 * IF(v_genre_match > 0, 100, 0)), 4);
END //
DELIMITER ;

-- ── Advanced Triggers (Watchlist/Reviews) ────────────────────────────────────

-- Trigger: after review INSERT, auto-complete watchlist + update recommendation_log
DROP TRIGGER IF EXISTS after_review_insert_update_log;
DELIMITER //
CREATE TRIGGER after_review_insert_update_log
AFTER INSERT ON movie_reviews FOR EACH ROW
BEGIN
    UPDATE recommendation_log SET was_watched = TRUE, was_clicked = TRUE
    WHERE user_id = NEW.user_id AND movie_id = NEW.movie_id;
    UPDATE watchlist SET status = 'completed', user_rating = NEW.rating, watched_at = CURRENT_TIMESTAMP
    WHERE user_id = NEW.user_id AND movie_id = NEW.movie_id AND status != 'completed';
END //
DELIMITER ;

-- Trigger: before review UPDATE — prevent dropping rating by > 3 pts
DROP TRIGGER IF EXISTS before_review_update_validate;
DELIMITER //
CREATE TRIGGER before_review_update_validate
BEFORE UPDATE ON movie_reviews FOR EACH ROW
BEGIN
    IF (OLD.rating - NEW.rating) > 3.0 THEN
        SIGNAL SQLSTATE '45002' SET MESSAGE_TEXT = 'Rating cannot be lowered by more than 3 points';
    END IF;
END //
DELIMITER ;

-- Trigger: after review DELETE — revert watchlist to 'watching'
DROP TRIGGER IF EXISTS after_review_delete_revert_watchlist;
DELIMITER //
CREATE TRIGGER after_review_delete_revert_watchlist
AFTER DELETE ON movie_reviews FOR EACH ROW
BEGIN
    UPDATE watchlist SET status = 'watching', user_rating = NULL
    WHERE user_id = OLD.user_id AND movie_id = OLD.movie_id AND status = 'completed';
END //
DELIMITER ;

-- Trigger: maintain genre win/loss counts on interactions
DROP TRIGGER IF EXISTS after_interaction_update_genre_wins;
DELIMITER //
CREATE TRIGGER after_interaction_update_genre_wins
AFTER INSERT ON user_interactions FOR EACH ROW
BEGIN
    UPDATE genre_win_counts gwc INNER JOIN movie_genres mg ON gwc.genre_id = mg.genre_id
    SET gwc.win_count = gwc.win_count + 1 WHERE mg.movie_id = NEW.chosen_movie_id;
    UPDATE genre_win_counts gwc INNER JOIN movie_genres mg ON gwc.genre_id = mg.genre_id
    SET gwc.loss_count = gwc.loss_count + 1 WHERE mg.movie_id = NEW.rejected_movie_id;
END //
DELIMITER ;

-- ── Cursor-Based Procedures ──────────────────────────────────────────────────

-- Procedure: seed watchlists for users who have none (cursor + exception handling)
DROP PROCEDURE IF EXISTS seed_watchlists_for_new_users;
DELIMITER //
CREATE PROCEDURE seed_watchlists_for_new_users(IN p_movies_per_user INT)
BEGIN
    DECLARE v_done INT DEFAULT FALSE;
    DECLARE v_user_id INT;
    DECLARE v_fav_genre VARCHAR(50);
    DECLARE v_seeded_count INT DEFAULT 0;
    DECLARE user_cursor CURSOR FOR
        SELECT u.user_id, COALESCE(up.favorite_genre, 'Action') AS fav_genre
        FROM users u LEFT JOIN user_preferences up ON u.user_id = up.user_id
        WHERE NOT EXISTS (SELECT 1 FROM watchlist w WHERE w.user_id = u.user_id);
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET v_done = TRUE;
    DECLARE CONTINUE HANDLER FOR SQLEXCEPTION BEGIN END;
    OPEN user_cursor;
    user_loop: LOOP
        FETCH user_cursor INTO v_user_id, v_fav_genre;
        IF v_done THEN LEAVE user_loop; END IF;
        INSERT IGNORE INTO watchlist (user_id, movie_id, priority, status)
        SELECT v_user_id, m.movie_id, 5, 'planned'
        FROM movies m INNER JOIN movie_genres mg ON m.movie_id = mg.movie_id
        INNER JOIN genres g ON mg.genre_id = g.genre_id
        WHERE g.genre_name = v_fav_genre AND m.tmdb_rating >= 7.0
        AND m.movie_id NOT IN (SELECT movie_id FROM watchlist WHERE user_id = v_user_id)
        ORDER BY m.popularity DESC LIMIT p_movies_per_user;
        SET v_seeded_count = v_seeded_count + ROW_COUNT();
    END LOOP user_loop;
    CLOSE user_cursor;
    SELECT CONCAT('Seeded ', v_seeded_count, ' watchlist entries.') AS result;
END //
DELIMITER ;

-- Procedure: batch re-score recommendations (cursor + EXIT HANDLER + transaction)
DROP PROCEDURE IF EXISTS batch_rescore_recommendations;
DELIMITER //
CREATE PROCEDURE batch_rescore_recommendations()
BEGIN
    DECLARE v_done INT DEFAULT FALSE;
    DECLARE v_log_id INT;
    DECLARE v_user_id INT;
    DECLARE v_movie_id INT;
    DECLARE v_new_score DECIMAL(10,4);
    DECLARE v_rows_updated INT DEFAULT 0;
    DECLARE rec_cursor CURSOR FOR
        SELECT log_id, user_id, movie_id FROM recommendation_log
        WHERE was_clicked = FALSE AND was_watched = FALSE
        ORDER BY recommended_at DESC;
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET v_done = TRUE;
    DECLARE EXIT HANDLER FOR SQLEXCEPTION BEGIN ROLLBACK; RESIGNAL; END;
    -- ACID — ISOLATION: READ COMMITTED prevents dirty reads without full serialization.
    -- Each SELECT inside the cursor sees only committed rows, avoiding phantom reads
    -- of recommendation_log entries inserted by concurrent sessions mid-loop.
    SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
    START TRANSACTION;
    OPEN rec_cursor;
    rec_loop: LOOP
        FETCH rec_cursor INTO v_log_id, v_user_id, v_movie_id;
        IF v_done THEN LEAVE rec_loop; END IF;
        SET v_new_score = weighted_recommendation_score(v_user_id, v_movie_id);
        IF v_new_score >= 0 THEN
            UPDATE recommendation_log SET recommendation_score = v_new_score WHERE log_id = v_log_id;
            SET v_rows_updated = v_rows_updated + 1;
        END IF;
    END LOOP rec_loop;
    CLOSE rec_cursor;
    COMMIT;
    SELECT CONCAT('Re-scored ', v_rows_updated, ' entries.') AS result;
END //
DELIMITER ;

-- Procedure: genre audit report (cursor + temp table)
DROP PROCEDURE IF EXISTS generate_genre_audit;
DELIMITER //
CREATE PROCEDURE generate_genre_audit()
BEGIN
    DECLARE v_done INT DEFAULT FALSE;
    DECLARE v_genre_id INT;
    DECLARE v_genre_name VARCHAR(50);
    DECLARE v_total_movies INT;
    DECLARE v_avg_rating DECIMAL(5,2);
    DECLARE v_win_count INT DEFAULT 0;
    DECLARE v_loss_count INT DEFAULT 0;
    DECLARE genre_cursor CURSOR FOR
        SELECT g.genre_id, g.genre_name, COUNT(DISTINCT mg.movie_id), ROUND(AVG(m.tmdb_rating), 2)
        FROM genres g LEFT JOIN movie_genres mg ON g.genre_id = mg.genre_id
        LEFT JOIN movies m ON mg.movie_id = m.movie_id GROUP BY g.genre_id;
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET v_done = TRUE;
    CREATE TEMPORARY TABLE IF NOT EXISTS tmp_genre_audit (
        genre_id INT, genre_name VARCHAR(50), total_movies INT,
        avg_rating DECIMAL(5,2), win_count INT, loss_count INT, win_rate_pct DECIMAL(5,2));
    DELETE FROM tmp_genre_audit;
    OPEN genre_cursor;
    audit_loop: LOOP
        FETCH genre_cursor INTO v_genre_id, v_genre_name, v_total_movies, v_avg_rating;
        IF v_done THEN LEAVE audit_loop; END IF;
        SELECT COALESCE(win_count, 0), COALESCE(loss_count, 0) INTO v_win_count, v_loss_count
        FROM genre_win_counts WHERE genre_id = v_genre_id;
        INSERT INTO tmp_genre_audit VALUES (v_genre_id, v_genre_name, v_total_movies, v_avg_rating,
            v_win_count, v_loss_count, ROUND(v_win_count / NULLIF(v_win_count + v_loss_count, 0) * 100, 2));
    END LOOP audit_loop;
    CLOSE genre_cursor;
    SELECT * FROM tmp_genre_audit ORDER BY win_rate_pct DESC;
    DROP TEMPORARY TABLE IF EXISTS tmp_genre_audit;
END //
DELIMITER ;

-- ── Exception Handling Procedures ────────────────────────────────────────────

-- Procedure: safe movie insert with full exception handling (SIGNAL, RESIGNAL, named conditions)
DROP PROCEDURE IF EXISTS safe_add_movie;
DELIMITER //
CREATE PROCEDURE safe_add_movie(
    IN p_movie_id INT, IN p_tmdb_id INT, IN p_title VARCHAR(255),
    IN p_rating DECIMAL(3,1), IN p_year INT, OUT p_status VARCHAR(100)
)
BEGIN
    DECLARE invalid_input CONDITION FOR SQLSTATE '45000';
    DECLARE EXIT HANDLER FOR 1062 BEGIN
        SET p_status = CONCAT('DUPLICATE: movie_id=', p_movie_id, ' already exists');
        ROLLBACK;
    END;
    DECLARE EXIT HANDLER FOR SQLEXCEPTION BEGIN
        GET DIAGNOSTICS CONDITION 1 @err_code = MYSQL_ERRNO, @err_msg = MESSAGE_TEXT;
        SET p_status = CONCAT('ERROR ', @err_code, ': ', @err_msg);
        ROLLBACK;
    END;
    IF p_title IS NULL OR TRIM(p_title) = '' THEN
        SIGNAL invalid_input SET MESSAGE_TEXT = 'Movie title cannot be empty';
    END IF;
    IF p_rating IS NOT NULL AND (p_rating < 0 OR p_rating > 10) THEN
        SIGNAL invalid_input SET MESSAGE_TEXT = 'Rating must be between 0.0 and 10.0';
    END IF;
    IF p_year IS NOT NULL AND (p_year < 1888 OR p_year > YEAR(CURRENT_DATE) + 5) THEN
        SIGNAL invalid_input SET MESSAGE_TEXT = 'Release year is out of realistic range';
    END IF;
    START TRANSACTION;
    INSERT INTO movies (movie_id, tmdb_id, title, tmdb_rating, release_year)
    VALUES (p_movie_id, p_tmdb_id, TRIM(p_title), p_rating, p_year);
    COMMIT;
    SET p_status = CONCAT('SUCCESS: "', TRIM(p_title), '" inserted');
END //
DELIMITER ;

-- Procedure: transactional Elo update with SAVEPOINT and exception handling
DROP PROCEDURE IF EXISTS transactional_elo_update;
DELIMITER //
CREATE PROCEDURE transactional_elo_update(
    IN p_winner_id INT, IN p_loser_id INT, OUT p_result VARCHAR(200)
)
BEGIN
    DECLARE v_elo_w INT;
    DECLARE v_elo_l INT;
    DECLARE v_expected DECIMAL(10,6);
    DECLARE v_k INT DEFAULT 32;
    DECLARE EXIT HANDLER FOR SQLEXCEPTION BEGIN
        ROLLBACK TO SAVEPOINT elo_update;
        GET DIAGNOSTICS CONDITION 1 @msg = MESSAGE_TEXT;
        SET p_result = CONCAT('ROLLED BACK — ', @msg);
        RELEASE SAVEPOINT elo_update;
    END;

    SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
    START TRANSACTION;
    SAVEPOINT elo_update;

    SELECT elo_score INTO v_elo_w FROM movies WHERE movie_id = p_winner_id FOR UPDATE;
    SELECT elo_score INTO v_elo_l FROM movies WHERE movie_id = p_loser_id FOR UPDATE;
    IF v_elo_w IS NULL OR v_elo_l IS NULL THEN
        SIGNAL SQLSTATE '45004' SET MESSAGE_TEXT = 'One or both movie IDs not found';
    END IF;
    SET v_expected = 1.0 / (1.0 + POWER(10, (v_elo_l - v_elo_w) / 400.0));
    UPDATE movies SET elo_score = v_elo_w + ROUND(v_k * (1 - v_expected)),
        comparison_count = comparison_count + 1 WHERE movie_id = p_winner_id;
    UPDATE movies SET elo_score = v_elo_l + ROUND(v_k * (0 - (1 - v_expected))),
        comparison_count = comparison_count + 1 WHERE movie_id = p_loser_id;
    RELEASE SAVEPOINT elo_update;
    COMMIT;
    SET p_result = CONCAT('SUCCESS: winner=', p_winner_id, ', loser=', p_loser_id);
END //
DELIMITER ;

DROP VIEW IF EXISTS cached_movies;
CREATE VIEW cached_movies AS
SELECT
    m.movie_id,
    m.tmdb_id,
    m.title,
    m.release_year,
    m.tmdb_rating,
    m.popularity,
    m.movie_source,
    m.is_persisted,
    m.last_accessed,
    m.access_count,
    m.elo_score,
    GROUP_CONCAT(DISTINCT g.genre_name ORDER BY g.genre_name SEPARATOR ', ') AS genres
FROM movies m
LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
LEFT JOIN genres g ON mg.genre_id = g.genre_id
WHERE m.is_persisted = TRUE OR m.movie_source IN ('user_interaction', 'cache')
GROUP BY m.movie_id
ORDER BY m.last_accessed DESC;

-- View: high-level lazy loading statistics
DROP VIEW IF EXISTS lazy_loading_stats;
CREATE VIEW lazy_loading_stats AS
SELECT
    COUNT(*) AS total_movies,
    SUM(CASE WHEN is_persisted = TRUE THEN 1 ELSE 0 END) AS persisted_movies,
    SUM(CASE WHEN is_persisted = FALSE THEN 1 ELSE 0 END) AS temporary_movies,
    SUM(CASE WHEN movie_source = 'tmdb_api' THEN 1 ELSE 0 END) AS api_movies,
    SUM(CASE WHEN movie_source = 'user_interaction' THEN 1 ELSE 0 END) AS interaction_movies,
    SUM(CASE WHEN movie_source = 'cache' THEN 1 ELSE 0 END) AS cache_movies,
    AVG(access_count) AS avg_access_count,
    MAX(access_count) AS max_access_count
FROM movies;

DELIMITER //

DROP PROCEDURE IF EXISTS cleanup_temporary_movies //
CREATE PROCEDURE cleanup_temporary_movies(IN days_old INT)
BEGIN
    DELETE FROM movies
    WHERE is_persisted = FALSE
      AND last_accessed < DATE_SUB(NOW(), INTERVAL days_old DAY);

    SELECT ROW_COUNT() AS deleted_count;
END //

DROP PROCEDURE IF EXISTS persist_movie //
CREATE PROCEDURE persist_movie(IN p_movie_id INT)
BEGIN
    UPDATE movies
    SET is_persisted = TRUE,
        movie_source = 'user_interaction',
        last_accessed = CURRENT_TIMESTAMP
    WHERE movie_id = p_movie_id;
END //

DROP PROCEDURE IF EXISTS update_movie_access //
CREATE PROCEDURE update_movie_access(IN p_movie_id INT)
BEGIN
    UPDATE movies
    SET access_count = access_count + 1,
        last_accessed = CURRENT_TIMESTAMP
    WHERE movie_id = p_movie_id;
END //

DROP PROCEDURE IF EXISTS log_cache_stats //
CREATE PROCEDURE log_cache_stats(
    IN p_movie_cache_size INT,
    IN p_vector_cache_size INT,
    IN p_movie_hits INT,
    IN p_movie_misses INT,
    IN p_vector_hits INT,
    IN p_vector_misses INT,
    IN p_refill_count INT,
    IN p_eviction_count INT
)
BEGIN
    DECLARE movie_total INT;
    DECLARE vector_total INT;
    DECLARE movie_rate DECIMAL(5,2);
    DECLARE vector_rate DECIMAL(5,2);

    SET movie_total = p_movie_hits + p_movie_misses;
    SET vector_total = p_vector_hits + p_vector_misses;
    SET movie_rate = IF(movie_total > 0, (p_movie_hits / movie_total * 100), 0);
    SET vector_rate = IF(vector_total > 0, (p_vector_hits / vector_total * 100), 0);

    INSERT INTO cache_stats (
        stat_date,
        cache_type,
        movie_cache_size,
        vector_cache_size,
        movie_hits,
        movie_misses,
        vector_hits,
        vector_misses,
        movie_hit_rate,
        vector_hit_rate,
        refill_count,
        eviction_count,
        cache_hits,
        cache_misses,
        recorded_at
    ) VALUES (
        CURRENT_DATE,
        'recommendation',
        p_movie_cache_size,
        p_vector_cache_size,
        p_movie_hits,
        p_movie_misses,
        p_vector_hits,
        p_vector_misses,
        movie_rate,
        vector_rate,
        p_refill_count,
        p_eviction_count,
        p_movie_hits + p_vector_hits,
        p_movie_misses + p_vector_misses,
        CURRENT_TIMESTAMP
    );
END //

-- Trigger: persist movies when they appear in pairwise interactions
DROP TRIGGER IF EXISTS persist_on_interaction //
CREATE TRIGGER persist_on_interaction
AFTER INSERT ON user_interactions
FOR EACH ROW
BEGIN
    CALL persist_movie(NEW.movie_1_id);
    CALL persist_movie(NEW.movie_2_id);
END //

-- Trigger: increment access count when last_accessed changes
DROP TRIGGER IF EXISTS track_movie_access //
CREATE TRIGGER track_movie_access
BEFORE UPDATE ON movies
FOR EACH ROW
BEGIN
    IF NEW.last_accessed > OLD.last_accessed THEN
        SET NEW.access_count = OLD.access_count + 1;
    END IF;
END //

DELIMITER ;

UPDATE movies
SET is_persisted = TRUE,
    movie_source = 'database',
    last_accessed = CURRENT_TIMESTAMP,
    access_count = COALESCE(access_count, 0)
WHERE movie_source IS NULL;

SELECT 'All advanced DBMS features created successfully!' AS Status;

DROP PROCEDURE IF EXISTS read_movie_elo_shared_lock;
DELIMITER //
CREATE PROCEDURE read_movie_elo_shared_lock(IN p_movie_id INT)
BEGIN
    -- Isolation level: REPEATABLE READ ensures the row snapshot is stable
    -- for the duration of this transaction.
    SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
    START TRANSACTION;

    -- LOCK IN SHARE MODE = Shared Row-Level Lock (S Lock)
    -- ✔ Multiple sessions can hold S locks on the same row concurrently
    -- ✔ Protects against dirty reads and non-repeatable reads
    -- ✘ Blocks any session that tries FOR UPDATE / UPDATE / DELETE on this row
    SELECT
        movie_id,
        title,
        elo_score,
        comparison_count,
        tmdb_rating
    FROM movies
    WHERE movie_id = p_movie_id
    LOCK IN SHARE MODE;

    COMMIT; -- Releases the shared lock
END //
DELIMITER ;

--
DROP PROCEDURE IF EXISTS generate_elo_report_table_read_lock;
DELIMITER //
CREATE PROCEDURE generate_elo_report_table_read_lock()
BEGIN
    -- Acquire shared TABLE-LEVEL read locks on all tables involved in the report.
    -- No writes can occur on movies/genres/movie_genres while this lock is held.
    LOCK TABLES
        movies      READ,
        genres      READ,
        movie_genres READ;

    -- Safe aggregate report — consistent snapshot guaranteed by the READ lock
    SELECT
        g.genre_name,
        COUNT(DISTINCT m.movie_id)          AS total_movies,
        ROUND(AVG(m.elo_score), 2)          AS avg_elo,
        MAX(m.elo_score)                    AS highest_elo,
        MIN(m.elo_score)                    AS lowest_elo,
        SUM(m.comparison_count)             AS total_comparisons
    FROM genres g
    INNER JOIN movie_genres mg ON g.genre_id = mg.genre_id
    INNER JOIN movies m       ON mg.movie_id = m.movie_id
    GROUP BY g.genre_id
    ORDER BY avg_elo DESC;

    -- Release all table-level locks — other sessions can write again
    UNLOCK TABLES;
END //
DELIMITER ;

--
DROP PROCEDURE IF EXISTS bulk_elo_reset_table_write_lock;
DELIMITER //
CREATE PROCEDURE bulk_elo_reset_table_write_lock(IN p_min_comparisons INT)
BEGIN
    DECLARE v_rows_reset INT DEFAULT 0;

    -- Acquire exclusive TABLE-LEVEL write lock on movies.
    -- Effect: ALL other sessions are blocked from reading or writing movies
    -- until UNLOCK TABLES is issued — guarantees no session sees a partial reset.
    LOCK TABLES movies WRITE;

    -- Bulk operation: reset ELO back to 1500 for under-compared movies
    UPDATE movies
    SET elo_score = 1500
    WHERE comparison_count < p_min_comparisons;

    SET v_rows_reset = ROW_COUNT();

    -- Release the exclusive table lock
    UNLOCK TABLES;

    SELECT v_rows_reset AS rows_reset,
           'ELO reset complete — table lock released' AS status;
END //
DELIMITER ;


SELECT 'Concurrency control procedures created successfully!' AS Status;
