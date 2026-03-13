"""
Social Features API Routes
Friend recommendations, watch parties, collaborative lists
"""

from flask import Blueprint, request, jsonify, session
from database.db_manager import db
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

social_bp = Blueprint('social', __name__, url_prefix='/api/social')


# ============================================================================
# FRIEND MANAGEMENT
# ============================================================================

@social_bp.route('/friends/add', methods=['POST'])
def add_friend():
    """Send friend request"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.json
        user_id = session['user_id']
        friend_id = data.get('friend_id')
        friend_username = data.get('friend_username')
        
        # Look up friend by username if id not provided
        if not friend_id and friend_username:
            friend_user = db.get_user_by_username(friend_username)
            if not friend_user:
                return jsonify({'error': f'User "{friend_username}" not found'}), 404
            friend_id = friend_user['user_id']
        
        if not friend_id:
            return jsonify({'error': 'friend_id or friend_username required'}), 400
        
        if user_id == friend_id:
            return jsonify({'error': 'Cannot add yourself as friend'}), 400
        
        # Check if friendship already exists
        existing = db.query("""
            SELECT * FROM friendships
            WHERE (user_id_1 = %s AND user_id_2 = %s)
               OR (user_id_1 = %s AND user_id_2 = %s)
        """, (user_id, friend_id, friend_id, user_id))
        
        if existing:
            return jsonify({'error': 'Friendship already exists'}), 400
        
        # Create friend request
        db.execute("""
            INSERT INTO friend_requests (from_user_id, to_user_id, status, created_at)
            VALUES (%s, %s, 'pending', NOW())
        """, (user_id, friend_id))
        
        return jsonify({
            'message': 'Friend request sent',
            'from_user_id': user_id,
            'to_user_id': friend_id
        }), 200
    
    except Exception as e:
        logger.error(f"Error sending friend request: {e}")
        return jsonify({'error': 'Failed to send friend request'}), 500


@social_bp.route('/friends/requests', methods=['GET'])
def get_friend_requests():
    """Get pending friend requests"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        user_id = session['user_id']
        
        requests = db.query("""
            SELECT 
                fr.request_id,
                fr.from_user_id,
                u.username,
                u.email,
                fr.created_at
            FROM friend_requests fr
            JOIN users u ON fr.from_user_id = u.user_id
            WHERE fr.to_user_id = %s AND fr.status = 'pending'
            ORDER BY fr.created_at DESC
        """, (user_id,))
        
        return jsonify({'requests': requests}), 200
    
    except Exception as e:
        logger.error(f"Error fetching friend requests: {e}")
        return jsonify({'error': 'Failed to fetch requests'}), 500


@social_bp.route('/friends/accept', methods=['POST'])
def accept_friend_request():
    """Accept friend request"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.json
        user_id = session['user_id']
        request_id = data.get('request_id')
        
        # Get request details
        request_data = db.query("""
            SELECT * FROM friend_requests
            WHERE request_id = %s AND to_user_id = %s AND status = 'pending'
        """, (request_id, user_id))
        
        if not request_data:
            return jsonify({'error': 'Request not found'}), 404
        
        from_user_id = request_data[0]['from_user_id']
        
        # Create friendship
        db.execute("""
            INSERT INTO friendships (user_id_1, user_id_2, status, created_at)
            VALUES (%s, %s, 'accepted', NOW())
        """, (from_user_id, user_id))
        
        # Update request status
        db.execute("""
            UPDATE friend_requests
            SET status = 'accepted'
            WHERE request_id = %s
        """, (request_id,))
        
        return jsonify({'message': 'Friend request accepted'}), 200
    
    except Exception as e:
        logger.error(f"Error accepting friend request: {e}")
        return jsonify({'error': 'Failed to accept request'}), 500


@social_bp.route('/friends/list', methods=['GET'])
def get_friends():
    """Get user's friends list"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        user_id = session['user_id']
        
        friends = db.query("""
            SELECT 
                CASE 
                    WHEN f.user_id_1 = %s THEN u2.user_id
                    ELSE u1.user_id
                END as friend_id,
                CASE 
                    WHEN f.user_id_1 = %s THEN u2.username
                    ELSE u1.username
                END as username,
                CASE 
                    WHEN f.user_id_1 = %s THEN u2.email
                    ELSE u1.email
                END as email,
                f.created_at
            FROM friendships f
            LEFT JOIN users u1 ON f.user_id_1 = u1.user_id
            LEFT JOIN users u2 ON f.user_id_2 = u2.user_id
            WHERE (f.user_id_1 = %s OR f.user_id_2 = %s)
              AND f.status = 'accepted'
            ORDER BY f.created_at DESC
        """, (user_id, user_id, user_id, user_id, user_id))
        
        return jsonify({'friends': friends}), 200
    
    except Exception as e:
        logger.error(f"Error fetching friends: {e}")
        return jsonify({'error': 'Failed to fetch friends'}), 500


# ============================================================================
# FRIEND RECOMMENDATIONS
# ============================================================================

@social_bp.route('/recommendations/friends', methods=['GET'])
def get_friend_recommendations():
    """Get movies your friends liked"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        user_id = session['user_id']
        limit = request.args.get('limit', 20, type=int)
        
        # Get friends
        friends = db.query("""
            SELECT 
                CASE 
                    WHEN f.user_id_1 = %s THEN f.user_id_2
                    ELSE f.user_id_1
                END as friend_id
            FROM friendships f
            WHERE (f.user_id_1 = %s OR f.user_id_2 = %s)
              AND f.status = 'accepted'
        """, (user_id, user_id, user_id))
        
        if not friends:
            return jsonify({'message': 'No friends yet', 'recommendations': []}), 200
        
        friend_ids = [f['friend_id'] for f in friends]
        
        # Get their top-rated movies (that user hasn't seen)
        recommendations = db.query("""
            SELECT 
                m.*,
                u.username as recommended_by,
                AVG(ui.implicit_rating) as friend_rating,
                COUNT(DISTINCT ui.user_id) as friend_count
            FROM movies m
            JOIN user_interactions ui ON m.movie_id = ui.winner_id
            JOIN users u ON ui.user_id = u.user_id
            WHERE ui.user_id IN (%s)
              AND ui.implicit_rating >= 4.0
              AND m.movie_id NOT IN (
                  SELECT winner_id FROM user_interactions WHERE user_id = %s
              )
            GROUP BY m.movie_id, u.username
            ORDER BY friend_rating DESC, friend_count DESC
            LIMIT %s
        """ % (','.join(['%s'] * len(friend_ids)), '%s', '%s'), 
        (*friend_ids, user_id, limit))
        
        return jsonify({
            'recommendations': recommendations,
            'friend_count': len(friend_ids)
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting friend recommendations: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500


# ============================================================================
# WATCH PARTIES
# ============================================================================

@social_bp.route('/watchparty/create', methods=['POST'])
def create_watch_party():
    """Create synchronized watch session"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.json
        user_id = session['user_id']
        
        movie_id = data.get('movie_id')
        scheduled_time = data.get('scheduled_time')  # ISO format
        invitees = data.get('invitees', [])  # List of user IDs
        
        if not movie_id or not scheduled_time:
            return jsonify({'error': 'movie_id and scheduled_time required'}), 400
        
        # Create watch party
        party_id = db.execute("""
            INSERT INTO watch_parties (host_id, movie_id, scheduled_time, status, created_at)
            VALUES (%s, %s, %s, 'pending', NOW())
        """, (user_id, movie_id, scheduled_time), return_lastrowid=True)
        
        # Invite friends
        for friend_id in invitees:
            db.execute("""
                INSERT INTO watch_party_invites (party_id, user_id, status, invited_at)
                VALUES (%s, %s, 'pending', NOW())
            """, (party_id, friend_id))
        
        return jsonify({
            'party_id': party_id,
            'message': 'Watch party created!',
            'invitees_count': len(invitees)
        }), 201
    
    except Exception as e:
        logger.error(f"Error creating watch party: {e}")
        return jsonify({'error': 'Failed to create watch party'}), 500


@social_bp.route('/watchparty/<int:party_id>', methods=['GET'])
def get_watch_party(party_id):
    """Get watch party details"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        party = db.query("""
            SELECT 
                wp.*,
                m.title as movie_title,
                m.poster_path,
                u.username as host_name
            FROM watch_parties wp
            JOIN movies m ON wp.movie_id = m.movie_id
            JOIN users u ON wp.host_id = u.user_id
            WHERE wp.party_id = %s
        """, (party_id,))
        
        if not party:
            return jsonify({'error': 'Watch party not found'}), 404
        
        # Get invitees
        invitees = db.query("""
            SELECT 
                wpi.*,
                u.username
            FROM watch_party_invites wpi
            JOIN users u ON wpi.user_id = u.user_id
            WHERE wpi.party_id = %s
        """, (party_id,))
        
        party[0]['invitees'] = invitees
        
        return jsonify(party[0]), 200
    
    except Exception as e:
        logger.error(f"Error fetching watch party: {e}")
        return jsonify({'error': 'Failed to fetch watch party'}), 500


@social_bp.route('/watchparty/<int:party_id>/join', methods=['POST'])
def join_watch_party(party_id):
    """Accept watch party invitation"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        user_id = session['user_id']
        
        # Update invitation status
        db.execute("""
            UPDATE watch_party_invites
            SET status = 'accepted'
            WHERE party_id = %s AND user_id = %s
        """, (party_id, user_id))
        
        return jsonify({'message': 'Joined watch party!'}), 200
    
    except Exception as e:
        logger.error(f"Error joining watch party: {e}")
        return jsonify({'error': 'Failed to join watch party'}), 500


@social_bp.route('/watchparty/upcoming', methods=['GET'])
def get_upcoming_watch_parties():
    """Get user's upcoming watch parties"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        user_id = session['user_id']
        
        parties = db.query("""
            SELECT 
                wp.*,
                m.title as movie_title,
                m.poster_path,
                u.username as host_name
            FROM watch_parties wp
            JOIN movies m ON wp.movie_id = m.movie_id
            JOIN users u ON wp.host_id = u.user_id
            WHERE wp.host_id = %s
               OR wp.party_id IN (
                   SELECT party_id FROM watch_party_invites
                   WHERE user_id = %s AND status = 'accepted'
               )
            AND wp.scheduled_time > NOW()
            ORDER BY wp.scheduled_time ASC
        """, (user_id, user_id))
        
        return jsonify({'parties': parties}), 200
    
    except Exception as e:
        logger.error(f"Error fetching watch parties: {e}")
        return jsonify({'error': 'Failed to fetch watch parties'}), 500


# ============================================================================
# COLLABORATIVE LISTS
# ============================================================================

@social_bp.route('/lists/create', methods=['POST'])
def create_collaborative_list():
    """Create a collaborative movie list"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.json
        user_id = session['user_id']
        
        list_name = data.get('name')
        description = data.get('description', '')
        is_public = data.get('is_public', False)
        
        if not list_name:
            return jsonify({'error': 'List name required'}), 400
        
        list_id = db.execute("""
            INSERT INTO movie_lists (creator_id, name, description, is_public, created_at)
            VALUES (%s, %s, %s, %s, NOW())
        """, (user_id, list_name, description, is_public), return_lastrowid=True)
        
        return jsonify({
            'list_id': list_id,
            'message': 'List created successfully'
        }), 201
    
    except Exception as e:
        logger.error(f"Error creating list: {e}")
        return jsonify({'error': 'Failed to create list'}), 500


@social_bp.route('/lists/<int:list_id>/add', methods=['POST'])
def add_movie_to_list(list_id):
    """Add movie to collaborative list"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.json
        movie_id = data.get('movie_id')
        
        if not movie_id:
            return jsonify({'error': 'movie_id required'}), 400
        
        db.execute("""
            INSERT INTO list_movies (list_id, movie_id, added_at)
            VALUES (%s, %s, NOW())
        """, (list_id, movie_id))
        
        return jsonify({'message': 'Movie added to list'}), 200
    
    except Exception as e:
        logger.error(f"Error adding movie to list: {e}")
        return jsonify({'error': 'Failed to add movie'}), 500


# Schema additions needed (add to database/schema.sql):
"""
-- Friend requests table
CREATE TABLE IF NOT EXISTS friend_requests (
    request_id INT AUTO_INCREMENT PRIMARY KEY,
    from_user_id INT NOT NULL,
    to_user_id INT NOT NULL,
    status ENUM('pending', 'accepted', 'rejected') DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (from_user_id) REFERENCES users(user_id),
    FOREIGN KEY (to_user_id) REFERENCES users(user_id)
);

-- Friendships table
CREATE TABLE IF NOT EXISTS friendships (
    friendship_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id_1 INT NOT NULL,
    user_id_2 INT NOT NULL,
    status ENUM('accepted', 'blocked') DEFAULT 'accepted',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id_1) REFERENCES users(user_id),
    FOREIGN KEY (user_id_2) REFERENCES users(user_id),
    UNIQUE KEY unique_friendship (user_id_1, user_id_2)
);

-- Watch parties table
CREATE TABLE IF NOT EXISTS watch_parties (
    party_id INT AUTO_INCREMENT PRIMARY KEY,
    host_id INT NOT NULL,
    movie_id INT NOT NULL,
    scheduled_time DATETIME NOT NULL,
    status ENUM('pending', 'active', 'completed', 'cancelled') DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (host_id) REFERENCES users(user_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
);

-- Watch party invites
CREATE TABLE IF NOT EXISTS watch_party_invites (
    invite_id INT AUTO_INCREMENT PRIMARY KEY,
    party_id INT NOT NULL,
    user_id INT NOT NULL,
    status ENUM('pending', 'accepted', 'declined') DEFAULT 'pending',
    invited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (party_id) REFERENCES watch_parties(party_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Movie lists
CREATE TABLE IF NOT EXISTS movie_lists (
    list_id INT AUTO_INCREMENT PRIMARY KEY,
    creator_id INT NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (creator_id) REFERENCES users(user_id)
);

-- List movies
CREATE TABLE IF NOT EXISTS list_movies (
    list_id INT NOT NULL,
    movie_id INT NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (list_id, movie_id),
    FOREIGN KEY (list_id) REFERENCES movie_lists(list_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
);
"""


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_social_routes(app, db_manager):
    """Initialize social routes with Flask app and database manager"""
    # Update global db reference to use provided db_manager
    global db
    db = db_manager
    
    # Create social tables if they don't exist
    try:
        db_manager.ensure_social_tables()
        logger.info("Social tables verified/created successfully")
    except Exception as e:
        logger.warning(f"Could not create social tables: {e}")
    
    # Register blueprint
    app.register_blueprint(social_bp)
    
    logger.info("Social routes initialized successfully")
    
    return social_bp
