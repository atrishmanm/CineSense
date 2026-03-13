"""
CineSense Relational Schema Diagram Generator
Generates a visual relational schema showing tables, columns, PKs, FKs, and relationships.
Requirements: pip install matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def draw_relational_schema():
    fig, ax = plt.subplots(1, 1, figsize=(28, 22))
    ax.set_xlim(-1, 27)
    ax.set_ylim(-1, 21)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('#FAFAFA')

    # ── Colors ──
    header_color = '#2C3E50'
    header_text = 'white'
    pk_bg = '#D5F5E3'
    fk_bg = '#FADBD8'
    pk_fk_bg = '#FCF3CF'
    normal_bg = 'white'
    border_color = '#2C3E50'

    def draw_table(x, y, name, columns, width=4.5, row_height=0.4):
        """
        Draw a relational table box.
        columns: list of (col_name, col_type, key_type)
            key_type: 'PK', 'FK', 'PK,FK', or ''
        """
        total_height = (len(columns) + 1) * row_height

        # Header
        header = mpatches.FancyBboxPatch(
            (x, y - row_height), width, row_height,
            boxstyle="square,pad=0", facecolor=header_color,
            edgecolor=border_color, linewidth=2
        )
        ax.add_patch(header)
        ax.text(x + width/2, y - row_height/2, name,
                ha='center', va='center', fontsize=10,
                fontweight='bold', color=header_text, family='monospace')

        # Columns
        for i, (col_name, col_type, key_type) in enumerate(columns):
            cy = y - (i + 2) * row_height
            if key_type == 'PK':
                bg = pk_bg
            elif key_type == 'FK':
                bg = fk_bg
            elif key_type == 'PK,FK':
                bg = pk_fk_bg
            else:
                bg = normal_bg

            cell = mpatches.FancyBboxPatch(
                (x, cy), width, row_height,
                boxstyle="square,pad=0", facecolor=bg,
                edgecolor=border_color, linewidth=1
            )
            ax.add_patch(cell)

            prefix = ''
            if 'PK' in key_type:
                prefix += 'PK '
            if 'FK' in key_type:
                prefix += 'FK '

            display_name = f"{prefix}{col_name}" if prefix else col_name
            ax.text(x + 0.15, cy + row_height/2, display_name,
                    ha='left', va='center', fontsize=7,
                    family='monospace',
                    fontweight='bold' if 'PK' in key_type else 'normal')
            ax.text(x + width - 0.15, cy + row_height/2, col_type,
                    ha='right', va='center', fontsize=6.5,
                    family='monospace', color='#666')

        bottom_y = y - (len(columns) + 1) * row_height
        return {
            'x': x, 'y': y, 'w': width,
            'top': y, 'bottom': bottom_y,
            'left': x, 'right': x + width,
            'center_x': x + width/2,
            'center_y': (y + bottom_y) / 2
        }

    def draw_fk_line(t1, side1, t2, side2, color='#E74C3C', style='-'):
        """Draw a foreign key relationship line between tables."""
        if side1 == 'right':
            p1 = (t1['right'], t1['center_y'])
        elif side1 == 'left':
            p1 = (t1['left'], t1['center_y'])
        elif side1 == 'bottom':
            p1 = (t1['center_x'], t1['bottom'])
        else:
            p1 = (t1['center_x'], t1['top'])

        if side2 == 'right':
            p2 = (t2['right'], t2['center_y'])
        elif side2 == 'left':
            p2 = (t2['left'], t2['center_y'])
        elif side2 == 'bottom':
            p2 = (t2['center_x'], t2['bottom'])
        else:
            p2 = (t2['center_x'], t2['top'])

        ax.annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=1.5, connectionstyle='arc3,rad=0.1'))

    # ════════════════════════════════════════════════════════════
    # TABLE DEFINITIONS
    # ════════════════════════════════════════════════════════════

    # Row 1: users, user_interactions, movies
    t_users = draw_table(0, 20, 'users', [
        ('user_id', 'INT AUTO_INCR', 'PK'),
        ('username', 'VARCHAR(50)', ''),
        ('email', 'VARCHAR(100)', ''),
        ('password_hash', 'VARCHAR(255)', ''),
        ('created_at', 'TIMESTAMP', ''),
        ('last_active', 'TIMESTAMP', ''),
        ('interaction_count', 'INT', ''),
    ])

    t_interactions = draw_table(6, 20, 'user_interactions', [
        ('interaction_id', 'INT AUTO_INCR', 'PK'),
        ('user_id', 'INT', 'FK'),
        ('movie_1_id', 'INT', 'FK'),
        ('movie_2_id', 'INT', 'FK'),
        ('chosen_movie_id', 'INT', 'FK'),
        ('rejected_movie_id', 'INT', 'FK'),
        ('timestamp', 'TIMESTAMP', ''),
        ('session_id', 'VARCHAR(100)', ''),
    ])

    t_movies = draw_table(12, 20, 'movies', [
        ('movie_id', 'INT', 'PK'),
        ('tmdb_id', 'INT UNIQUE', ''),
        ('title', 'VARCHAR(255)', ''),
        ('original_title', 'VARCHAR(255)', ''),
        ('overview', 'TEXT', ''),
        ('release_year', 'INT', ''),
        ('runtime', 'INT', ''),
        ('poster_path', 'VARCHAR(255)', ''),
        ('backdrop_path', 'VARCHAR(255)', ''),
        ('tmdb_rating', 'DECIMAL(3,1)', ''),
        ('vote_count', 'INT', ''),
        ('popularity', 'DECIMAL(10,3)', ''),
        ('watch_link', 'VARCHAR(500)', ''),
        ('elo_score', 'INT DEFAULT 1500', ''),
        ('comparison_count', 'INT DEFAULT 0', ''),
        ('created_at', 'TIMESTAMP', ''),
    ])

    # Row 2: junction tables and entity tables
    t_movie_genres = draw_table(0, 10, 'movie_genres', [
        ('movie_id', 'INT', 'PK,FK'),
        ('genre_id', 'INT', 'PK,FK'),
    ])

    t_genres = draw_table(0, 8, 'genres', [
        ('genre_id', 'INT AUTO_INCR', 'PK'),
        ('genre_name', 'VARCHAR(50)', ''),
        ('tmdb_genre_id', 'INT UNIQUE', ''),
    ])

    t_movie_directors = draw_table(6, 10, 'movie_directors', [
        ('movie_id', 'INT', 'PK,FK'),
        ('director_id', 'INT', 'PK,FK'),
    ])

    t_directors = draw_table(6, 8, 'directors', [
        ('director_id', 'INT AUTO_INCR', 'PK'),
        ('director_name', 'VARCHAR(100)', ''),
        ('tmdb_person_id', 'INT UNIQUE', ''),
        ('popularity', 'DECIMAL(10,3)', ''),
    ])

    t_movie_actors = draw_table(12, 10, 'movie_actors', [
        ('movie_id', 'INT', 'PK,FK'),
        ('actor_id', 'INT', 'PK,FK'),
        ('cast_order', 'INT DEFAULT 0', ''),
        ('character_name', 'VARCHAR(255)', ''),
    ])

    t_actors = draw_table(12, 8, 'actors', [
        ('actor_id', 'INT AUTO_INCR', 'PK'),
        ('actor_name', 'VARCHAR(100)', ''),
        ('tmdb_person_id', 'INT UNIQUE', ''),
        ('popularity', 'DECIMAL(10,3)', ''),
    ])

    # Row 3: embedding tables
    t_user_embed = draw_table(19, 20, 'user_embeddings', [
        ('user_id', 'INT', 'PK,FK'),
        ('feature_index', 'INT', 'PK'),
        ('feature_value', 'DECIMAL(10,6)', ''),
        ('last_updated', 'TIMESTAMP', ''),
    ])

    t_movie_embed = draw_table(19, 14, 'movie_embeddings', [
        ('movie_id', 'INT', 'PK,FK'),
        ('feature_index', 'INT', 'PK'),
        ('feature_value', 'DECIMAL(10,6)', ''),
    ])

    # ════════════════════════════════════════════════════════════
    # FOREIGN KEY RELATIONSHIPS
    # ════════════════════════════════════════════════════════════
    draw_fk_line(t_interactions, 'left', t_users, 'right', '#E74C3C')
    draw_fk_line(t_interactions, 'right', t_movies, 'left', '#3498DB')
    draw_fk_line(t_movie_genres, 'bottom', t_genres, 'top', '#27AE60')
    draw_fk_line(t_movie_directors, 'bottom', t_directors, 'top', '#27AE60')
    draw_fk_line(t_movie_actors, 'bottom', t_actors, 'top', '#27AE60')
    draw_fk_line(t_user_embed, 'left', t_users, 'right', '#8E44AD')
    draw_fk_line(t_movie_embed, 'left', t_movies, 'right', '#8E44AD')

    # ════════════════════════════════════════════════════════════
    # TITLE & LEGEND
    # ════════════════════════════════════════════════════════════
    ax.set_title('CineSense — Relational Schema Diagram\n11 Tables | 3NF Normalized | MySQL 8.0',
                 fontsize=16, fontweight='bold', pad=20)

    legend_elements = [
        mpatches.Patch(facecolor=pk_bg, edgecolor=border_color, label='Primary Key (PK)'),
        mpatches.Patch(facecolor=fk_bg, edgecolor=border_color, label='Foreign Key (FK)'),
        mpatches.Patch(facecolor=pk_fk_bg, edgecolor=border_color, label='PK + FK (Composite)'),
        mpatches.Patch(facecolor=normal_bg, edgecolor=border_color, label='Regular Column'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig('relational_schema.png', dpi=200, bbox_inches='tight',
                facecolor='#FAFAFA', edgecolor='none')
    plt.savefig('relational_schema.pdf', bbox_inches='tight',
                facecolor='#FAFAFA', edgecolor='none')
    print("Relational Schema Diagram saved: relational_schema.png, relational_schema.pdf")
    plt.show()


if __name__ == '__main__':
    draw_relational_schema()
