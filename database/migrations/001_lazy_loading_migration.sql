-- ============================================================================
-- LAZY LOADING MIGRATION v1.0
-- Adds support for selective storage and cache tracking
-- ============================================================================

-- Migration: 001_lazy_loading_migration.sql
-- Description: Adds columns and indexes to support lazy loading architecture
-- Date: 2026-01-15
-- Author: CineSense Team

-- ============================================================================
-- 1. ADD MOVIE SOURCE TRACKING
-- ============================================================================

-- Add column to track where movie data came from (cache, db, or api)
ALTER TABLE movies 
ADD COLUMN IF NOT EXISTS movie_source ENUM('tmdb_api', 'user_interaction', 'cache', 'database') 
DEFAULT 'database' 
AFTER comparison_count;

-- Add column to track if movie should be persisted
ALTER TABLE movies 
ADD COLUMN IF NOT EXISTS is_persisted BOOLEAN 
DEFAULT FALSE 
AFTER movie_source;

-- Add column to track last access time (for LRU eviction)
ALTER TABLE movies 
ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMP 
DEFAULT CURRENT_TIMESTAMP 
ON UPDATE CURRENT_TIMESTAMP 
AFTER is_persisted;

-- Add column to track access count
ALTER TABLE movies 
ADD COLUMN IF NOT EXISTS access_count INT 
DEFAULT 0 
AFTER last_accessed;

-- ============================================================================
-- 2. ADD INTERACTION TYPE TRACKING
-- ============================================================================

-- Add column to track type of user interaction
ALTER TABLE user_interactions 
ADD COLUMN IF NOT EXISTS interaction_type ENUM('comparison', 'recommendation', 'search', 'view', 'click') 
DEFAULT 'comparison' 
AFTER session_id;

-- Add column to track if interaction came from lazy loading
ALTER TABLE user_interactions 
ADD COLUMN IF NOT EXISTS is_lazy_loaded BOOLEAN 
DEFAULT FALSE 
AFTER interaction_type;

-- Add column to track cache hit/miss
ALTER TABLE user_interactions 
ADD COLUMN IF NOT EXISTS cache_hit BOOLEAN 
DEFAULT NULL 
AFTER is_lazy_loaded;

-- ============================================================================
-- 3. CREATE CACHE TRACKING TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS cache_stats (
    stat_id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
    INDEX idx_timestamp (timestamp DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- 4. CREATE CANDIDATE GENERATION TRACKING TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS candidate_generation_log (
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

-- ============================================================================
-- 5. ADD INDEXES FOR LAZY LOADING QUERIES
-- ============================================================================

-- Index for finding cached movies
CREATE INDEX IF NOT EXISTS idx_movie_source ON movies(movie_source);

-- Index for finding persisted movies
CREATE INDEX IF NOT EXISTS idx_is_persisted ON movies(is_persisted);

-- Index for LRU eviction (find least recently accessed)
CREATE INDEX IF NOT EXISTS idx_last_accessed ON movies(last_accessed ASC);

-- Index for access count (find most accessed)
CREATE INDEX IF NOT EXISTS idx_access_count ON movies(access_count DESC);

-- Composite index for selective storage queries
CREATE INDEX IF NOT EXISTS idx_persisted_accessed ON movies(is_persisted, last_accessed DESC);

-- Index for lazy loaded interactions
CREATE INDEX IF NOT EXISTS idx_lazy_loaded ON user_interactions(is_lazy_loaded, timestamp DESC);

-- Index for cache hit tracking
CREATE INDEX IF NOT EXISTS idx_cache_hit ON user_interactions(cache_hit);

-- ============================================================================
-- 6. CREATE VIEW FOR CACHED MOVIES
-- ============================================================================

CREATE OR REPLACE VIEW cached_movies AS
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
ORDER BY m.last_accessed DESC
LIMIT 100;

-- ============================================================================
-- 7. CREATE VIEW FOR LAZY LOADING STATISTICS
-- ============================================================================

CREATE OR REPLACE VIEW lazy_loading_stats AS
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

-- ============================================================================
-- 8. CREATE STORED PROCEDURE FOR CACHE CLEANUP
-- ============================================================================

DELIMITER //

CREATE PROCEDURE IF NOT EXISTS cleanup_temporary_movies(IN days_old INT)
BEGIN
    -- Delete non-persisted movies older than X days
    DELETE FROM movies 
    WHERE is_persisted = FALSE 
    AND last_accessed < DATE_SUB(NOW(), INTERVAL days_old DAY);
    
    SELECT ROW_COUNT() AS deleted_count;
END //

-- Procedure to mark movie as persisted after user interaction
CREATE PROCEDURE IF NOT EXISTS persist_movie(IN p_movie_id INT)
BEGIN
    UPDATE movies 
    SET is_persisted = TRUE,
        movie_source = 'user_interaction',
        last_accessed = CURRENT_TIMESTAMP
    WHERE movie_id = p_movie_id;
END //

-- Procedure to update movie access
CREATE PROCEDURE IF NOT EXISTS update_movie_access(IN p_movie_id INT)
BEGIN
    UPDATE movies 
    SET access_count = access_count + 1,
        last_accessed = CURRENT_TIMESTAMP
    WHERE movie_id = p_movie_id;
END //

-- Procedure to log cache stats
CREATE PROCEDURE IF NOT EXISTS log_cache_stats(
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
        movie_cache_size, 
        vector_cache_size, 
        movie_hits, 
        movie_misses, 
        vector_hits, 
        vector_misses,
        movie_hit_rate,
        vector_hit_rate,
        refill_count,
        eviction_count
    ) VALUES (
        p_movie_cache_size,
        p_vector_cache_size,
        p_movie_hits,
        p_movie_misses,
        p_vector_hits,
        p_vector_misses,
        movie_rate,
        vector_rate,
        p_refill_count,
        p_eviction_count
    );
END //

DELIMITER ;

-- ============================================================================
-- 9. UPDATE EXISTING MOVIES (Set defaults)
-- ============================================================================

-- Mark existing movies as persisted (they were fetched before lazy loading)
UPDATE movies 
SET is_persisted = TRUE,
    movie_source = 'database',
    last_accessed = CURRENT_TIMESTAMP,
    access_count = 0
WHERE movie_source IS NULL;

-- ============================================================================
-- 10. CREATE TRIGGERS FOR AUTOMATIC TRACKING
-- ============================================================================

DELIMITER //

-- Trigger to persist movies when they appear in interactions
CREATE TRIGGER IF NOT EXISTS persist_on_interaction
AFTER INSERT ON user_interactions
FOR EACH ROW
BEGIN
    -- Persist both movies in the comparison
    CALL persist_movie(NEW.movie_1_id);
    CALL persist_movie(NEW.movie_2_id);
END //

-- Trigger to update access count on movie retrieval
CREATE TRIGGER IF NOT EXISTS track_movie_access
BEFORE UPDATE ON movies
FOR EACH ROW
BEGIN
    IF NEW.last_accessed > OLD.last_accessed THEN
        SET NEW.access_count = OLD.access_count + 1;
    END IF;
END //

DELIMITER ;

-- ============================================================================
-- 11. PERFORMANCE OPTIMIZATION
-- ============================================================================

-- Optimize the movies table
OPTIMIZE TABLE movies;

-- Analyze tables for better query planning
ANALYZE TABLE movies;
ANALYZE TABLE user_interactions;
ANALYZE TABLE cache_stats;
ANALYZE TABLE candidate_generation_log;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

SELECT 'Lazy Loading Migration v1.0 completed successfully!' AS Status,
       (SELECT COUNT(*) FROM movies) AS total_movies,
       (SELECT COUNT(*) FROM movies WHERE is_persisted = TRUE) AS persisted_movies,
       (SELECT COUNT(*) FROM cache_stats) AS cache_stat_entries;
