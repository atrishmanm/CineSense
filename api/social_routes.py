"""
Social Features API Routes
Friend recommendations, watch parties, collaborative lists
"""

from flask import Blueprint, request, jsonify, session
from database.db_manager import db
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

social_bp = Blueprint('social', __name__, url_prefix='/api/social')

MAX_INVITEES_PER_PARTY = 100


def _json_body():
    """Parse a JSON request body, returning {} when the body is absent or invalid."""
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _are_friends(user_a, user_b):
    """Return True when an accepted friendship links the two users."""
    rows = db.query("""
        SELECT 1 AS ok FROM friendships
        WHERE status = 'accepted'
          AND ((user_id_1 = %s AND user_id_2 = %s)
            OR (user_id_1 = %s AND user_id_2 = %s))
        LIMIT 1
    """, (user_a, user_b, user_b, user_a))
    return bool(rows)


def _list_access(list_id, user_id):
    """Return (list_row, allowed) for a collaborative list.

    allowed is True when the user is the creator, an explicit member, or the
    list is public (view access).
    """
    list_row = db.query(
        "SELECT * FROM collaborative_lists WHERE list_id = %s",
        (list_id,),
        fetch_all=False
    )
    if not list_row:
        return None, False
    if list_row.get('creator_id') == user_id or list_row.get('is_public'):
        return list_row, True
    member = db.query(
        "SELECT 1 AS ok FROM collaborative_list_members WHERE list_id = %s AND user_id = %s",
        (list_id, user_id),
        fetch_all=False
    )
    return list_row, bool(member)


def _can_edit_list(list_id, user_id):
    """True when the user may add/remove items on the list (creator or member)."""
    list_row = db.query(
        "SELECT * FROM collaborative_lists WHERE list_id = %s",
        (list_id,),
        fetch_all=False
    )
    if not list_row:
        return None  # unknown list
    if list_row.get('creator_id') == user_id:
        return True
    member = db.query(
        "SELECT 1 AS ok FROM collaborative_list_members WHERE list_id = %s AND user_id = %s",
        (list_id, user_id),
        fetch_all=False
    )
    return bool(member)


def _parse_scheduled_time(value):
    """Parse a client supplied scheduled_time into MySQL DATETIME format."""
    if not value or not isinstance(value, str):
        return None
    raw = value.strip().replace('T', ' ')
    if '.' in raw:
        raw = raw.split('.')[0]
    if raw.endswith('Z'):
        raw = raw[:-1].strip()
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
        try:
            return datetime.strptime(raw, fmt).strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue
    return None

@social_bp.route('/ping', methods=['GET'])
def ping_social():
    return jsonify({'status': 'ok', 'message': 'Social API is alive'}), 200


# ============================================================================
# FRIEND MANAGEMENT
# ============================================================================

@social_bp.route('/friends/add', methods=['POST'])
def add_friend():
    """Send friend request"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = _json_body()
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

        # Check if a request already exists in either direction
        pending = db.query("""
            SELECT * FROM friend_requests
            WHERE (from_user_id = %s AND to_user_id = %s AND status = 'pending')
               OR (from_user_id = %s AND to_user_id = %s AND status = 'pending')
        """, (user_id, friend_id, friend_id, user_id))

        if pending:
            if pending[0]['from_user_id'] == friend_id:
                return jsonify({'error': 'This user already sent you a request. Check Requests tab.'}), 400
            return jsonify({'error': 'Friend request already sent'}), 400
        
        # Create friend request with duplicate key handling
        db.execute("""
            INSERT INTO friend_requests (from_user_id, to_user_id, status, created_at)
            VALUES (%s, %s, 'pending', NOW())
            ON DUPLICATE KEY UPDATE status = 'pending', created_at = NOW()
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
        data = _json_body()
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


@social_bp.route('/friends/remove', methods=['POST'])
def remove_friend():
    """Remove a friend relationship"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = _json_body()
    friend_id = data.get('friend_id')
    user_id = session['user_id']
    
    if not friend_id:
        return jsonify({'error': 'friend_id required'}), 400
        
    try:
        # Remove from friendships
        db.execute("""
            DELETE FROM friendships 
            WHERE (user_id_1 = %s AND user_id_2 = %s)
               OR (user_id_1 = %s AND user_id_2 = %s)
        """, (user_id, friend_id, friend_id, user_id))
        
        # Also remove any requests to allow clean re-requesting later
        db.execute("""
            DELETE FROM friend_requests
            WHERE (from_user_id = %s AND to_user_id = %s)
               OR (from_user_id = %s AND to_user_id = %s)
        """, (user_id, friend_id, friend_id, user_id))
        
        return jsonify({'message': 'Friend removed successfully'}), 200
    except Exception as e:
        logger.error(f"Error removing friend: {e}")
        return jsonify({'error': 'Failed to remove friend'}), 500


@social_bp.route('/friends/list', methods=['GET'])
def get_friends():
    """Get user's friends list"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        user_id = session['user_id']
        
        friends = db.query("""
            SELECT 
                u.user_id as friend_id,
                u.user_id as id,
                u.username,
                u.email,
                MAX(f.created_at) as created_at
            FROM (
                SELECT user_id_2 as friend_id, created_at FROM friendships WHERE user_id_1 = %s AND status = 'accepted'
                UNION ALL
                SELECT user_id_1 as friend_id, created_at FROM friendships WHERE user_id_2 = %s AND status = 'accepted'
            ) f
            JOIN users u ON f.friend_id = u.user_id
            GROUP BY u.user_id, u.username, u.email
            ORDER BY created_at DESC
        """, (user_id, user_id))
        
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
        try:
            limit = int(request.args.get('limit', 20))
        except (TypeError, ValueError):
            return jsonify({'error': 'limit must be an integer'}), 400
        limit = max(1, min(limit, 100))
        
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
        
        # Get movies friends have chosen (liked)
        recommendations = db.query("""
            SELECT 
                m.*,
                u.username as recommended_by,
                COUNT(ui.interaction_id) as friend_vote_count
            FROM movies m
            JOIN user_interactions ui ON m.movie_id = ui.chosen_movie_id
            JOIN users u ON ui.user_id = u.user_id
            WHERE ui.user_id IN (%s)
              AND m.movie_id NOT IN (
                  SELECT chosen_movie_id FROM user_interactions WHERE user_id = %s
                  UNION
                  SELECT rejected_movie_id FROM user_interactions WHERE user_id = %s
              )
            GROUP BY m.movie_id, u.username
            ORDER BY friend_vote_count DESC, m.popularity DESC
            LIMIT %s
        """ % (','.join(['%s'] * len(friend_ids)), '%s', '%s', '%s'), 
        (*friend_ids, user_id, user_id, limit))
        
        return jsonify({
            'recommendations': recommendations,
            'friend_count': len(friend_ids)
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting friend recommendations: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500


@social_bp.route('/friends/recommend', methods=['POST'])
def recommend_movie_to_friend():
    """Explicitly recommend a movie to a friend"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = _json_body()
        user_id = session['user_id']
        friend_id = data.get('friend_id')
        movie_id = data.get('movie_id')
        
        if not all([friend_id, movie_id]):
            return jsonify({'error': 'friend_id and movie_id required'}), 400

        # The friend must exist and be an accepted friend of the sender
        friend_user = db.get_user_by_id(friend_id)
        if not friend_user:
            return jsonify({'error': 'Friend not found'}), 404
        if not _are_friends(user_id, friend_id):
            return jsonify({'error': 'You can only recommend movies to accepted friends'}), 403

        movie = db.get_movie_by_id(movie_id)
        if not movie:
            return jsonify({'error': 'Movie not found'}), 404

        # Return a real recommendation payload to the caller. We deliberately do
        # NOT write to user_interactions: the previous implementation inserted a
        # fake zero-sum interaction (movie_1 == movie_2 == chosen == rejected) that
        # polluted ELO/interaction data and never reached the friend.
        return jsonify({
            'message': 'Movie recommended to friend!',
            'recommendation': {
                'from_user_id': user_id,
                'to_user_id': friend_id,
                'movie_id': movie_id,
                'movie': {
                    'movie_id': movie.get('movie_id'),
                    'title': movie.get('title'),
                    'poster_path': movie.get('poster_path'),
                    'release_year': movie.get('release_year'),
                    'tmdb_rating': movie.get('tmdb_rating'),
                },
                'created_at': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error recommending movie: {e}")
        return jsonify({'error': 'Failed to recommend movie'}), 500


# ============================================================================
# WATCH PARTIES
# ============================================================================

@social_bp.route('/watchparty/create', methods=['POST'])
def create_watch_party():
    """Create synchronized watch session"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = _json_body()
        user_id = session['user_id']
        
        movie_id = data.get('movie_id')
        scheduled_time = _parse_scheduled_time(data.get('scheduled_time'))
        invitees = data.get('invitees', [])  # List of user IDs
        
        if not movie_id or not scheduled_time:
            return jsonify({'error': 'movie_id and a valid scheduled_time (YYYY-MM-DD HH:MM) required'}), 400

        if not isinstance(invitees, list):
            return jsonify({'error': 'invitees must be a list of user IDs'}), 400
        if len(invitees) > MAX_INVITEES_PER_PARTY:
            return jsonify({'error': f'At most {MAX_INVITEES_PER_PARTY} invitees allowed'}), 400

        if not db.get_movie_by_id(movie_id):
            return jsonify({'error': 'Movie not found'}), 404

        # Validate + normalise invitee ids: each must be an existing user and a friend
        clean_invitees = []
        for raw_id in invitees:
            try:
                invitee_id = int(raw_id)
            except (TypeError, ValueError):
                return jsonify({'error': f'Invalid invitee id: {raw_id!r}'}), 400
            if invitee_id == user_id:
                continue
            if not db.get_user_by_id(invitee_id):
                return jsonify({'error': f'User {invitee_id} does not exist'}), 404
            if not _are_friends(user_id, invitee_id):
                return jsonify({'error': f'User {invitee_id} is not your friend'}), 403
            if invitee_id not in clean_invitees:
                clean_invitees.append(invitee_id)

        # Create watch party
        party_id = db.execute("""
            INSERT INTO watch_parties (host_id, movie_id, scheduled_time, status, created_at)
            VALUES (%s, %s, %s, 'scheduled', NOW())
        """, (user_id, movie_id, scheduled_time), return_lastrowid=True)
        
        # Invite friends
        for friend_id in clean_invitees:
            db.execute("""
                INSERT INTO watch_party_invites (party_id, user_id, status, invited_at)
                VALUES (%s, %s, 'invited', NOW())
                ON DUPLICATE KEY UPDATE status = 'invited', invited_at = NOW()
            """, (party_id, friend_id))
        
        return jsonify({
            'party_id': party_id,
            'message': 'Watch party created!',
            'invitees_count': len(clean_invitees)
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
        user_id = session['user_id']

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

        # Only the host or an invited participant may read the party
        if party[0]['host_id'] != user_id:
            invite = db.query(
                "SELECT invite_id FROM watch_party_invites WHERE party_id = %s AND user_id = %s",
                (party_id, user_id),
                fetch_all=False
            )
            if not invite:
                return jsonify({'error': 'Not authorized to view this watch party'}), 403

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

        party = db.query(
            "SELECT party_id FROM watch_parties WHERE party_id = %s",
            (party_id,),
            fetch_all=False
        )
        if not party:
            return jsonify({'error': 'Watch party not found'}), 404

        invite = db.query(
            "SELECT invite_id, status FROM watch_party_invites WHERE party_id = %s AND user_id = %s",
            (party_id, user_id),
            fetch_all=False
        )
        if not invite:
            return jsonify({'error': 'You are not invited to this watch party'}), 404

        # Update invitation status
        db.execute("""
            UPDATE watch_party_invites
            SET status = 'accepted'
            WHERE party_id = %s AND user_id = %s AND status <> 'accepted'
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
        data = _json_body()
        user_id = session['user_id']
        
        list_name = (data.get('name') or '').strip()
        description = data.get('description', '')
        is_public = bool(data.get('is_public', False))
        
        if not list_name:
            return jsonify({'error': 'List name required'}), 400
        if len(list_name) > 255:
            return jsonify({'error': 'List name must be at most 255 characters'}), 400
        
        list_id = db.execute("""
            INSERT INTO collaborative_lists (creator_id, name, description, is_public, created_at)
            VALUES (%s, %s, %s, %s, NOW())
        """, (user_id, list_name, description, is_public), return_lastrowid=True)

        # Register the creator as an explicit owner member
        try:
            db.execute("""
                INSERT INTO collaborative_list_members (list_id, user_id, role, joined_at)
                VALUES (%s, %s, 'owner', NOW())
                ON DUPLICATE KEY UPDATE role = 'owner'
            """, (list_id, user_id))
        except Exception as exc:
            logger.warning(f"Could not register list owner membership: {exc}")
        
        return jsonify({
            'list_id': list_id,
            'message': 'List created successfully'
        }), 201
    
    except Exception as e:
        logger.error(f"Error creating list: {e}")
        return jsonify({'error': 'Failed to create list'}), 500


@social_bp.route('/lists/<int:list_id>', methods=['GET'])
def get_collaborative_list(list_id):
    """Get a collaborative list with its items and members"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        user_id = session['user_id']
        list_row, allowed = _list_access(list_id, user_id)
        if not list_row:
            return jsonify({'error': 'List not found'}), 404
        if not allowed:
            return jsonify({'error': 'Not authorized to view this list'}), 403

        items = db.query("""
            SELECT
                cli.item_id,
                cli.list_id,
                cli.movie_id,
                cli.added_by,
                cli.notes,
                cli.position,
                cli.added_at,
                m.title,
                m.poster_path,
                m.release_year,
                m.tmdb_rating,
                u.username as added_by_username
            FROM collaborative_list_items cli
            JOIN movies m ON cli.movie_id = m.movie_id
            JOIN users u ON cli.added_by = u.user_id
            WHERE cli.list_id = %s
            ORDER BY cli.position IS NULL, cli.position ASC, cli.added_at ASC
        """, (list_id,))

        members = db.query("""
            SELECT clm.*, u.username
            FROM collaborative_list_members clm
            JOIN users u ON clm.user_id = u.user_id
            WHERE clm.list_id = %s
            ORDER BY u.username ASC
        """, (list_id,))

        return jsonify({
            'list': list_row,
            'items': items,
            'members': members,
            'count': len(items)
        }), 200

    except Exception as e:
        logger.error(f"Error fetching list: {e}")
        return jsonify({'error': 'Failed to fetch list'}), 500


@social_bp.route('/lists/<int:list_id>/add', methods=['POST'])
def add_movie_to_list(list_id):
    """Add movie to collaborative list"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        user_id = session['user_id']
        data = _json_body()
        movie_id = data.get('movie_id')
        notes = data.get('notes')
        position = data.get('position', 0)

        if not movie_id:
            return jsonify({'error': 'movie_id required'}), 400

        if notes is not None:
            notes = str(notes)
            if len(notes) > 500:
                return jsonify({'error': 'notes must be at most 500 characters'}), 400
        if position is None:
            position = 0
        try:
            position = int(position)
        except (TypeError, ValueError):
            return jsonify({'error': 'position must be an integer'}), 400
        if position < 0:
            return jsonify({'error': 'position must be non-negative'}), 400

        # Ownership / membership check (creator or collaborative_list_members)
        can_edit = _can_edit_list(list_id, user_id)
        if can_edit is None:
            return jsonify({'error': 'List not found'}), 404
        if not can_edit:
            return jsonify({'error': 'Not authorized to modify this list'}), 403

        if not db.get_movie_by_id(movie_id):
            return jsonify({'error': 'Movie not found'}), 404
        
        db.execute("""
            INSERT INTO collaborative_list_items (list_id, movie_id, added_by, notes, position, added_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE
                added_by = VALUES(added_by),
                notes = VALUES(notes),
                position = VALUES(position),
                added_at = NOW()
        """, (list_id, movie_id, user_id, notes, position))
        
        return jsonify({'message': 'Movie added to list'}), 200
    
    except Exception as e:
        logger.error(f"Error adding movie to list: {e}")
        return jsonify({'error': 'Failed to add movie'}), 500


# Social feature tables (friend_requests, friendships, watch_parties,
# watch_party_invites, collaborative_lists, collaborative_list_items,
# collaborative_list_members) are defined in database/schema.sql and are
# created by db_manager.ensure_social_tables(). Do not keep a duplicate DDL
# copy here -- it drifts out of sync with the schema.


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
