-- CineSense Database Schema
-- Fully Normalized (3NF) Design
-- MySQL 8.0+

-- Drop existing tables if they exist (for clean setup)
DROP TABLE IF EXISTS user_interactions;
DROP TABLE IF EXISTS user_embeddings;
DROP TABLE IF EXISTS movie_actors;
DROP TABLE IF EXISTS movie_directors;
DROP TABLE IF EXISTS movie_genres;
DROP TABLE IF EXISTS actors;
DROP TABLE IF EXISTS directors;
DROP TABLE IF EXISTS genres;
DROP TABLE IF EXISTS movies;
DROP TABLE IF EXISTS users;

-- ============================================================================
-- USERS TABLE
-- ============================================================================
CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    interaction_count INT DEFAULT 0,
    INDEX idx_username (username),
    INDEX idx_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- MOVIES TABLE
-- ============================================================================
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_title (title),
    INDEX idx_release_year (release_year),
    INDEX idx_elo_score (elo_score),
    INDEX idx_popularity (popularity)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- GENRES TABLE (Normalized)
-- ============================================================================
CREATE TABLE genres (
    genre_id INT AUTO_INCREMENT PRIMARY KEY,
    genre_name VARCHAR(50) NOT NULL UNIQUE,
    tmdb_genre_id INT UNIQUE,
    INDEX idx_genre_name (genre_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- DIRECTORS TABLE (Normalized)
-- ============================================================================
CREATE TABLE directors (
    director_id INT AUTO_INCREMENT PRIMARY KEY,
    director_name VARCHAR(100) NOT NULL,
    tmdb_person_id INT UNIQUE,
    popularity DECIMAL(10,3),
    INDEX idx_director_name (director_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- ACTORS TABLE (Normalized)
-- ============================================================================
CREATE TABLE actors (
    actor_id INT AUTO_INCREMENT PRIMARY KEY,
    actor_name VARCHAR(100) NOT NULL,
    tmdb_person_id INT UNIQUE,
    popularity DECIMAL(10,3),
    INDEX idx_actor_name (actor_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- MOVIE_GENRES TABLE (Many-to-Many Relationship)
-- ============================================================================
CREATE TABLE movie_genres (
    movie_id INT NOT NULL,
    genre_id INT NOT NULL,
    PRIMARY KEY (movie_id, genre_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    FOREIGN KEY (genre_id) REFERENCES genres(genre_id) ON DELETE CASCADE,
    INDEX idx_genre_lookup (genre_id, movie_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- MOVIE_DIRECTORS TABLE (Many-to-Many Relationship)
-- ============================================================================
CREATE TABLE movie_directors (
    movie_id INT NOT NULL,
    director_id INT NOT NULL,
    PRIMARY KEY (movie_id, director_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    FOREIGN KEY (director_id) REFERENCES directors(director_id) ON DELETE CASCADE,
    INDEX idx_director_lookup (director_id, movie_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- MOVIE_ACTORS TABLE (Many-to-Many Relationship)
-- ============================================================================
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

-- ============================================================================
-- USER_INTERACTIONS TABLE (Pairwise Comparison History)
-- ============================================================================
CREATE TABLE user_interactions (
    interaction_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    movie_1_id INT NOT NULL,
    movie_2_id INT NOT NULL,
    chosen_movie_id INT NOT NULL,
    rejected_movie_id INT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR(100),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (movie_1_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    FOREIGN KEY (movie_2_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    FOREIGN KEY (chosen_movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    FOREIGN KEY (rejected_movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    INDEX idx_user_interactions (user_id, timestamp DESC),
    INDEX idx_chosen_movie (chosen_movie_id),
    INDEX idx_rejected_movie (rejected_movie_id),
    INDEX idx_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- USER_EMBEDDINGS TABLE (User Preference Vectors)
-- ============================================================================
CREATE TABLE user_embeddings (
    user_id INT NOT NULL,
    feature_index INT NOT NULL,
    feature_value DECIMAL(10,6) NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, feature_index),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    INDEX idx_user_vector (user_id, feature_index)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- MOVIE_EMBEDDINGS TABLE (Movie Feature Vectors)
-- ============================================================================
CREATE TABLE movie_embeddings (
    movie_id INT NOT NULL,
    feature_index INT NOT NULL,
    feature_value DECIMAL(10,6) NOT NULL,
    PRIMARY KEY (movie_id, feature_index),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    INDEX idx_movie_vector (movie_id, feature_index)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- SAMPLE GENRES (Pre-populate for TMDB compatibility)
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
-- VIEWS FOR EASIER QUERYING
-- ============================================================================

-- Complete movie information with all metadata
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

-- ============================================================================
-- STORED PROCEDURES
-- ============================================================================

DELIMITER //

-- Update user interaction count
CREATE PROCEDURE update_user_interaction_count(IN p_user_id INT)
BEGIN
    UPDATE users 
    SET interaction_count = interaction_count + 1,
        last_active = CURRENT_TIMESTAMP
    WHERE user_id = p_user_id;
END //

-- Update movie ELO scores after comparison
CREATE PROCEDURE update_movie_elo(
    IN p_winner_id INT, 
    IN p_loser_id INT,
    IN p_k_factor INT
)
BEGIN
    DECLARE winner_elo INT;
    DECLARE loser_elo INT;
    DECLARE expected_winner DECIMAL(5,4);
    DECLARE expected_loser DECIMAL(5,4);
    DECLARE winner_change INT;
    DECLARE loser_change INT;
    
    -- Get current ELO scores
    SELECT elo_score INTO winner_elo FROM movies WHERE movie_id = p_winner_id;
    SELECT elo_score INTO loser_elo FROM movies WHERE movie_id = p_loser_id;
    
    -- Calculate expected scores
    SET expected_winner = 1 / (1 + POW(10, (loser_elo - winner_elo) / 400));
    SET expected_loser = 1 / (1 + POW(10, (winner_elo - loser_elo) / 400));
    
    -- Calculate ELO changes
    SET winner_change = ROUND(p_k_factor * (1 - expected_winner));
    SET loser_change = ROUND(p_k_factor * (0 - expected_loser));
    
    -- Update scores
    UPDATE movies 
    SET elo_score = elo_score + winner_change,
        comparison_count = comparison_count + 1
    WHERE movie_id = p_winner_id;
    
    UPDATE movies 
    SET elo_score = elo_score + loser_change,
        comparison_count = comparison_count + 1
    WHERE movie_id = p_loser_id;
END //

DELIMITER ;

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Additional composite indexes for common queries
CREATE INDEX idx_movie_rating_popularity ON movies(tmdb_rating DESC, popularity DESC);
CREATE INDEX idx_movie_year_rating ON movies(release_year DESC, tmdb_rating DESC);
CREATE INDEX idx_user_interactions_timestamp ON user_interactions(user_id, timestamp DESC);

-- Full-text search indexes
ALTER TABLE movies ADD FULLTEXT INDEX ft_title_overview (title, overview);

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================
SELECT 'CineSense database schema created successfully!' AS Status;
