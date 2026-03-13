"""
Database Schema Update for New Features
Run this script to add social features and other new tables
"""

import mysql.connector
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_schema():
    """Update database schema with new features"""
    
    logger.info("Connecting to database...")
    conn = mysql.connector.connect(**Config.DB_CONFIG)
    cursor = conn.cursor()
    
    # SQL for new tables
    updates = [
        # Social Features - Friend Requests
        """
        CREATE TABLE IF NOT EXISTS friend_requests (
            request_id INT PRIMARY KEY AUTO_INCREMENT,
            user_id INT NOT NULL,
            friend_id INT NOT NULL,
            status ENUM('pending', 'accepted', 'rejected') DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            FOREIGN KEY (friend_id) REFERENCES users(user_id) ON DELETE CASCADE,
            UNIQUE KEY unique_request (user_id, friend_id),
            INDEX idx_user_status (user_id, status),
            INDEX idx_friend_status (friend_id, status)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        
        # Social Features - Friendships
        """
        CREATE TABLE IF NOT EXISTS friendships (
            user_id INT NOT NULL,
            friend_id INT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, friend_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            FOREIGN KEY (friend_id) REFERENCES users(user_id) ON DELETE CASCADE,
            INDEX idx_user (user_id),
            INDEX idx_friend (friend_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        
        # Social Features - Watch Parties
        """
        CREATE TABLE IF NOT EXISTS watch_parties (
            party_id INT PRIMARY KEY AUTO_INCREMENT,
            movie_id INT NOT NULL,
            host_id INT NOT NULL,
            scheduled_time TIMESTAMP NOT NULL,
            status ENUM('scheduled', 'live', 'completed', 'cancelled') DEFAULT 'scheduled',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
            FOREIGN KEY (host_id) REFERENCES users(user_id) ON DELETE CASCADE,
            INDEX idx_movie (movie_id),
            INDEX idx_host (host_id),
            INDEX idx_scheduled (scheduled_time)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        
        # Social Features - Watch Party Invites
        """
        CREATE TABLE IF NOT EXISTS watch_party_invites (
            party_id INT NOT NULL,
            user_id INT NOT NULL,
            status ENUM('invited', 'accepted', 'declined') DEFAULT 'invited',
            invited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (party_id, user_id),
            FOREIGN KEY (party_id) REFERENCES watch_parties(party_id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            INDEX idx_user (user_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        
        # Social Features - Movie Lists
        """
        CREATE TABLE IF NOT EXISTS movie_lists (
            list_id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            owner_id INT NOT NULL,
            is_public BOOLEAN DEFAULT TRUE,
            is_collaborative BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (owner_id) REFERENCES users(user_id) ON DELETE CASCADE,
            INDEX idx_owner (owner_id),
            INDEX idx_public (is_public)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        
        # Social Features - List Movies
        """
        CREATE TABLE IF NOT EXISTS list_movies (
            list_id INT NOT NULL,
            movie_id INT NOT NULL,
            added_by INT NOT NULL,
            position INT NOT NULL DEFAULT 0,
            notes TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (list_id, movie_id),
            FOREIGN KEY (list_id) REFERENCES movie_lists(list_id) ON DELETE CASCADE,
            FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
            FOREIGN KEY (added_by) REFERENCES users(user_id) ON DELETE CASCADE,
            INDEX idx_list (list_id),
            INDEX idx_movie (movie_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        
        # Chat/Conversation History
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            chat_id INT PRIMARY KEY AUTO_INCREMENT,
            user_id INT,
            session_id VARCHAR(64) NOT NULL,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            INDEX idx_session (session_id),
            INDEX idx_user (user_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        
        # A/B Testing Experiments
        """
        CREATE TABLE IF NOT EXISTS ab_experiments (
            experiment_id VARCHAR(64) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            variants JSON NOT NULL,
            traffic_allocation JSON NOT NULL,
            start_date TIMESTAMP NOT NULL,
            end_date TIMESTAMP NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        
        # A/B Testing User Assignments
        """
        CREATE TABLE IF NOT EXISTS ab_user_assignments (
            user_id INT NOT NULL,
            experiment_id VARCHAR(64) NOT NULL,
            variant VARCHAR(64) NOT NULL,
            assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, experiment_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            INDEX idx_experiment (experiment_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        
        # A/B Testing Metrics
        """
        CREATE TABLE IF NOT EXISTS ab_metrics (
            metric_id INT PRIMARY KEY AUTO_INCREMENT,
            experiment_id VARCHAR(64) NOT NULL,
            user_id INT NOT NULL,
            variant VARCHAR(64) NOT NULL,
            metric_name VARCHAR(64) NOT NULL,
            metric_value FLOAT NOT NULL,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
            INDEX idx_experiment_variant (experiment_id, variant),
            INDEX idx_user (user_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    ]
    
    # Execute updates
    success_count = 0
    error_count = 0
    
    for i, sql in enumerate(updates, 1):
        try:
            cursor.execute(sql)
            conn.commit()
            table_name = sql.split("TABLE IF NOT EXISTS ")[1].split(" (")[0]
            logger.info(f"✓ [{i}/{len(updates)}] Created/verified table: {table_name}")
            success_count += 1
        except Exception as e:
            logger.error(f"✗ [{i}/{len(updates)}] Error: {e}")
            error_count += 1
    
    cursor.close()
    conn.close()
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Schema Update Complete")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info("=" * 60)
    
    return success_count, error_count


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("CineSense - Database Schema Update")
    print("=" * 60)
    print("\nThis will add new tables for:")
    print("  • Social features (friends, watch parties)")
    print("  • Chat history")
    print("  • A/B testing framework")
    print("\n" + "=" * 60)
    
    answer = input("\nContinue? (y/n): ")
    if answer.lower() == 'y':
        success, errors = update_schema()
        if errors == 0:
            print("\n✅ Database schema updated successfully!")
        else:
            print(f"\n⚠️ Completed with {errors} error(s)")
    else:
        print("\nCancelled.")
