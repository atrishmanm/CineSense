-- ============================================================================
-- CINESENSE ENHANCED DATABASE SCHEMA
-- For DBMS Course Evaluation - Demonstrating Complex SQL Concepts
-- ============================================================================
-- Implements: Triggers, Stored Procedures, Functions, Views, Cursors,
--            Aggregate Functions, Joins, Set Operations, Constraints
-- ============================================================================

-- Active: 1768449834747@@127.0.0.1@3306@cinesense

USE cinesense;

-- ============================================================================
-- ADDITIONAL TABLES FOR SEARCH ENHANCEMENT
-- ============================================================================

-- Search History Table (Track what users search for)
CREATE TABLE IF NOT EXISTS search_history (
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

-- Movie Keywords Table (For semantic search)
CREATE TABLE IF NOT EXISTS movie_keywords (
    keyword_id INT AUTO_INCREMENT PRIMARY KEY,
    movie_id INT NOT NULL,
    keyword VARCHAR(100) NOT NULL,
    relevance_score DECIMAL(5,2) DEFAULT 1.0,
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    UNIQUE KEY unique_movie_keyword (movie_id, keyword),
    INDEX idx_keyword (keyword),
    INDEX idx_movie_keywords (movie_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- User Preferences (Aggregated preferences for quick access)
CREATE TABLE IF NOT EXISTS user_preferences (
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

-- Cache Monitor Table (Track caching performance)
CREATE TABLE IF NOT EXISTS cache_stats (
    stat_id INT AUTO_INCREMENT PRIMARY KEY,
    stat_date DATE NOT NULL,
    cache_type ENUM('movie', 'vector', 'user', 'recommendation') NOT NULL,
    cache_hits INT DEFAULT 0,
    cache_misses INT DEFAULT 0,
    avg_response_time_ms DECIMAL(10,2) DEFAULT 0,
    memory_usage_kb INT DEFAULT 0,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_cache_stat (stat_date, cache_type),
    INDEX idx_stat_date (stat_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Recommendation Log (Track recommendations given to users)
CREATE TABLE IF NOT EXISTS recommendation_log (
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
-- COMPLEX VIEWS (Demonstrating Joins, Aggregations, and Subqueries)
-- ============================================================================

-- Comprehensive Movie Details with All Metadata
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
    -- Aggregated genres
    GROUP_CONCAT(DISTINCT g.genre_name ORDER BY g.genre_name SEPARATOR ', ') AS genres,
    -- Aggregated directors
    GROUP_CONCAT(DISTINCT d.director_name ORDER BY d.director_name SEPARATOR ', ') AS directors,
    -- Aggregated cast (top 10)
    GROUP_CONCAT(DISTINCT a.actor_name ORDER BY ma.cast_order LIMIT 10 SEPARATOR ', ') AS cast,
    -- Aggregated keywords for search
    GROUP_CONCAT(DISTINCT mk.keyword ORDER BY mk.relevance_score DESC SEPARATOR ', ') AS keywords,
    -- Computed popularity rank
    (SELECT COUNT(*) + 1 FROM movies m2 WHERE m2.popularity > m.popularity) AS popularity_rank,
    -- Computed rating rank
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
    -- Interaction statistics
    COUNT(DISTINCT ui.interaction_id) AS total_comparisons,
    COUNT(DISTINCT ui.chosen_movie_id) AS unique_movies_chosen,
    COUNT(DISTINCT ui.rejected_movie_id) AS unique_movies_rejected,
    -- Date statistics
    DATEDIFF(CURRENT_DATE, u.created_at) AS days_since_joined,
    DATEDIFF(CURRENT_DATE, u.last_active) AS days_since_last_active,
    -- Preference statistics
    AVG(m.tmdb_rating) AS avg_chosen_rating,
    AVG(m.release_year) AS avg_chosen_year,
    MIN(m.release_year) AS oldest_movie_chosen,
    MAX(m.release_year) AS newest_movie_chosen,
    -- Activity level
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
    -- Rank by popularity
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
    -- Activity trend
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
    -- Win rate
    ROUND((COUNT(DISTINCT CASE WHEN ui.chosen_movie_id = m.movie_id THEN ui.interaction_id END) /
           NULLIF(COUNT(DISTINCT ui.interaction_id), 0)) * 100, 2) AS win_rate_percentage
FROM movies m
LEFT JOIN user_interactions ui ON (m.movie_id = ui.movie_1_id OR m.movie_id = ui.movie_2_id)
    AND ui.timestamp >= DATE_SUB(CURRENT_DATE, INTERVAL 7 DAY)
GROUP BY m.movie_id
HAVING recent_comparisons > 0
ORDER BY recent_comparisons DESC, win_rate_percentage DESC;

-- ============================================================================
-- STORED PROCEDURES (Complex Business Logic)
-- ============================================================================

-- Update user interaction count and preferences
DROP PROCEDURE IF EXISTS update_user_interaction_count;
DELIMITER //
CREATE PROCEDURE update_user_interaction_count(IN p_user_id INT)
BEGIN
    DECLARE interaction_total INT;
    
    -- Count total interactions
    SELECT COUNT(*) INTO interaction_total
    FROM user_interactions
    WHERE user_id = p_user_id;
    
    -- Update user table
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
    
    -- Determine rejected movie
    IF p_chosen_movie_id = p_movie_1_id THEN
        SET v_rejected_movie_id = p_movie_2_id;
    ELSE
        SET v_rejected_movie_id = p_movie_1_id;
    END IF;
    
    -- Insert interaction record
    INSERT INTO user_interactions (
        user_id, movie_1_id, movie_2_id, 
        chosen_movie_id, rejected_movie_id, session_id
    ) VALUES (
        p_user_id, p_movie_1_id, p_movie_2_id,
        p_chosen_movie_id, v_rejected_movie_id, p_session_id
    );
    
    -- Get current Elo scores
    SELECT elo_score INTO v_elo_winner FROM movies WHERE movie_id = p_chosen_movie_id;
    SELECT elo_score INTO v_elo_loser FROM movies WHERE movie_id = v_rejected_movie_id;
    
    -- Calculate expected score for winner
    SET v_expected_score = 1.0 / (1.0 + POWER(10, (v_elo_loser - v_elo_winner) / 400.0));
    
    -- Calculate new Elo scores
    SET v_new_elo_winner = v_elo_winner + ROUND(v_k_factor * (1 - v_expected_score));
    SET v_new_elo_loser = v_elo_loser + ROUND(v_k_factor * (0 - (1 - v_expected_score)));
    
    -- Update Elo scores and comparison counts
    UPDATE movies 
    SET elo_score = v_new_elo_winner,
        comparison_count = comparison_count + 1
    WHERE movie_id = p_chosen_movie_id;
    
    UPDATE movies 
    SET elo_score = v_new_elo_loser,
        comparison_count = comparison_count + 1
    WHERE movie_id = v_rejected_movie_id;
    
    -- Update user interaction count
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
    
    -- Get favorite genre (most chosen)
    SELECT g.genre_name INTO v_favorite_genre
    FROM user_interactions ui
    INNER JOIN movie_genres mg ON ui.chosen_movie_id = mg.movie_id
    INNER JOIN genres g ON mg.genre_id = g.genre_id
    WHERE ui.user_id = p_user_id
    GROUP BY g.genre_name
    ORDER BY COUNT(*) DESC
    LIMIT 1;
    
    -- Get favorite actor
    SELECT a.actor_name INTO v_favorite_actor
    FROM user_interactions ui
    INNER JOIN movie_actors ma ON ui.chosen_movie_id = ma.movie_id
    INNER JOIN actors a ON ma.actor_id = a.actor_id
    WHERE ui.user_id = p_user_id
    GROUP BY a.actor_name
    ORDER BY COUNT(*) DESC
    LIMIT 1;
    
    -- Get favorite director
    SELECT d.director_name INTO v_favorite_director
    FROM user_interactions ui
    INNER JOIN movie_directors md ON ui.chosen_movie_id = md.movie_id
    INNER JOIN directors d ON md.director_id = d.director_id
    WHERE ui.user_id = p_user_id
    GROUP BY d.director_name
    ORDER BY COUNT(*) DESC
    LIMIT 1;
    
    -- Calculate average rating preference
    SELECT AVG(m.tmdb_rating) INTO v_avg_rating
    FROM user_interactions ui
    INNER JOIN movies m ON ui.chosen_movie_id = m.movie_id
    WHERE ui.user_id = p_user_id;
    
    -- Calculate preferred decade
    SELECT FLOOR(AVG(m.release_year) / 10) * 10 INTO v_preferred_decade
    FROM user_interactions ui
    INNER JOIN movies m ON ui.chosen_movie_id = m.movie_id
    WHERE ui.user_id = p_user_id;
    
    -- Get total interactions
    SELECT COUNT(*) INTO v_total_interactions
    FROM user_interactions
    WHERE user_id = p_user_id;
    
    -- Insert or update preferences
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
    
    -- Cursor for iterating through candidate movies
    DECLARE movie_cursor CURSOR FOR
        SELECT DISTINCT m.movie_id,
            -- Calculate relevance score
            (
                -- Genre match bonus
                (CASE WHEN FIND_IN_SET(v_favorite_genre, 
                    (SELECT GROUP_CONCAT(g.genre_name) 
                     FROM movie_genres mg 
                     INNER JOIN genres g ON mg.genre_id = g.genre_id 
                     WHERE mg.movie_id = m.movie_id)) > 0 
                THEN 50 ELSE 0 END) +
                -- Popularity score (normalized)
                (m.popularity / 100) +
                -- Rating score
                (m.tmdb_rating * 5) +
                -- Elo score (normalized)
                ((m.elo_score - 1500) / 10) +
                -- Recency bonus (newer movies slight preference)
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
    
    -- Get user's favorite genre
    SELECT favorite_genre INTO v_favorite_genre
    FROM user_preferences
    WHERE user_id = p_user_id;
    
    -- Create temporary table for results
    CREATE TEMPORARY TABLE IF NOT EXISTS temp_recommendations (
        movie_id INT,
        relevance_score DECIMAL(10,2),
        PRIMARY KEY (movie_id)
    );
    
    -- Clear temp table
    DELETE FROM temp_recommendations;
    
    -- Open cursor and fetch recommendations
    OPEN movie_cursor;
    
    read_loop: LOOP
        FETCH movie_cursor INTO v_movie_id, v_score;
        IF done THEN
            LEAVE read_loop;
        END IF;
        
        -- Insert into temp table
        INSERT INTO temp_recommendations (movie_id, relevance_score)
        VALUES (v_movie_id, v_score);
    END LOOP;
    
    CLOSE movie_cursor;
    
    -- Return recommendations with full details
    SELECT m.*, tr.relevance_score
    FROM temp_recommendations tr
    INNER JOIN comprehensive_movie_view m ON tr.movie_id = m.movie_id
    ORDER BY tr.relevance_score DESC;
    
    -- Clean up
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
    
    -- Record search history
    INSERT INTO search_history (user_id, search_query, search_type)
    VALUES (p_user_id, p_search_query, p_search_type);
    
    -- Perform search based on type
    IF p_search_type = 'storyline' OR p_search_type = 'semantic' THEN
        -- Search in overview/storyline and keywords
        SELECT DISTINCT m.*, 
            -- Calculate relevance score
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
        -- Standard title search
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
    
    -- Update search results count
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
    
    -- Get user interaction count
    SELECT interaction_count INTO v_interaction_count
    FROM users
    WHERE user_id = p_user_id;
    
    -- Get favorite genre if available
    SELECT favorite_genre INTO v_favorite_genre
    FROM user_preferences
    WHERE user_id = p_user_id;
    
    IF v_interaction_count < 10 THEN
        -- New users: show popular diverse movies
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
        -- Experienced users: mix of preference-based and exploration
        (
            -- 70% from favorite genre
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
            -- 30% exploration from other genres
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

-- ============================================================================
-- FUNCTIONS (Scalar and Aggregate Calculations)
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
    
    -- Count common movie choices
    SELECT COUNT(*) INTO v_common_choices
    FROM (
        SELECT chosen_movie_id FROM user_interactions WHERE user_id = p_user1_id
        INTERSECT
        SELECT chosen_movie_id FROM user_interactions WHERE user_id = p_user2_id
    ) AS common_movies;
    
    -- Count total unique choices
    SELECT COUNT(DISTINCT chosen_movie_id) INTO v_total_choices
    FROM user_interactions
    WHERE user_id IN (p_user1_id, p_user2_id);
    
    -- Calculate Jaccard similarity
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
-- TRIGGERS (Automatic Data Management)
-- ============================================================================

-- Trigger: Auto-update user preferences after interaction
DROP TRIGGER IF EXISTS after_interaction_insert;
DELIMITER //
CREATE TRIGGER after_interaction_insert
AFTER INSERT ON user_interactions
FOR EACH ROW
BEGIN
    -- Update user interaction count
    UPDATE users
    SET interaction_count = interaction_count + 1,
        last_active = CURRENT_TIMESTAMP
    WHERE user_id = NEW.user_id;
    
    -- Update user preferences (only after every 5 interactions for performance)
    IF (SELECT interaction_count FROM users WHERE user_id = NEW.user_id) % 5 = 0 THEN
        CALL update_user_preferences(NEW.user_id);
    END IF;
END //
DELIMITER ;

-- Trigger: Validate movie rating before insert/update
DROP TRIGGER IF EXISTS before_movie_insert_validate;
DELIMITER //
CREATE TRIGGER before_movie_insert_validate
BEFORE INSERT ON movies
FOR EACH ROW
BEGIN
    -- Ensure rating is between 0 and 10
    IF NEW.tmdb_rating IS NOT NULL AND (NEW.tmdb_rating < 0 OR NEW.tmdb_rating > 10) THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'TMDB rating must be between 0 and 10';
    END IF;
    
    -- Ensure elo_score has default if not provided
    IF NEW.elo_score IS NULL THEN
        SET NEW.elo_score = 1500;
    END IF;
    
    -- Ensure comparison_count has default
    IF NEW.comparison_count IS NULL THEN
        SET NEW.comparison_count = 0;
    END IF;
END //
DELIMITER ;

-- Trigger: Log movie updates
DROP TRIGGER IF EXISTS before_movie_update_validate;
DELIMITER //
CREATE TRIGGER before_movie_update_validate
BEFORE UPDATE ON movies
FOR EACH ROW
BEGIN
    -- Ensure rating is between 0 and 10
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
    -- Could add analytics or alerting here
    -- For now, just a placeholder for future enhancements
    SET @last_search_id = NEW.search_id;
END //
DELIMITER ;

-- ============================================================================
-- INDEXES FOR PERFORMANCE (Beyond Basic Indexes)
-- ============================================================================

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_movies_rating_popularity ON movies(tmdb_rating DESC, popularity DESC);
CREATE INDEX IF NOT EXISTS idx_movies_year_rating ON movies(release_year DESC, tmdb_rating DESC);
CREATE INDEX IF NOT EXISTS idx_user_interactions_user_timestamp ON user_interactions(user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_recommendation_log_user_movie ON recommendation_log(user_id, movie_id);

-- ============================================================================
-- SAMPLE DATA INSERTION FOR TESTING
-- ============================================================================

-- Insert sample keywords for existing movies (for semantic search)
INSERT IGNORE INTO movie_keywords (movie_id, keyword, relevance_score)
SELECT m.movie_id, 'action', 1.0
FROM movies m
INNER JOIN movie_genres mg ON m.movie_id = mg.movie_id
INNER JOIN genres g ON mg.genre_id = g.genre_id
WHERE g.genre_name = 'Action'
LIMIT 100;

-- ============================================================================
-- COMPLEX QUERY EXAMPLES (For DBMS Evaluation)
-- ============================================================================

-- Example 1: Set Operations (UNION, INTERSECT simulation)
-- Find movies that are both highly rated AND popular
DROP VIEW IF EXISTS elite_movies;
CREATE VIEW elite_movies AS
SELECT movie_id, title, 'Elite' AS category FROM movies
WHERE tmdb_rating >= 8.0 AND popularity >= 100
UNION
SELECT movie_id, title, 'Rising Star' AS category FROM movies
WHERE comparison_count > 50 AND elo_score > 1600;

-- Example 2: Complex Join with Multiple Tables
-- Get users with their favorite genres and similar users
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

-- Example 3: Subquery with ALL, ANY, EXISTS
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
-- ADMINISTRATION AND MONITORING QUERIES
-- ============================================================================

-- View all triggers
-- SHOW TRIGGERS;

-- View all stored procedures
-- SHOW PROCEDURE STATUS WHERE Db = 'cinesense';

-- View all functions
-- SHOW FUNCTION STATUS WHERE Db = 'cinesense';

-- View all views
-- SELECT TABLE_NAME FROM INFORMATION_SCHEMA.VIEWS WHERE TABLE_SCHEMA = 'cinesense';

-- ============================================================================
-- GRANT PERMISSIONS (If needed for web application)
-- ============================================================================

-- GRANT EXECUTE ON PROCEDURE cinesense.record_user_interaction TO 'cinesense_app'@'localhost';
-- GRANT EXECUTE ON PROCEDURE cinesense.get_personalized_recommendations TO 'cinesense_app'@'localhost';
-- GRANT EXECUTE ON PROCEDURE cinesense.search_movies_advanced TO 'cinesense_app'@'localhost';
-- GRANT SELECT ON cinesense.comprehensive_movie_view TO 'cinesense_app'@'localhost';

-- ============================================================================
-- END OF ENHANCED SCHEMA
-- ============================================================================
