"""
CineSense ER Diagram Generator
Generates a Chen-notation Entity-Relationship diagram using matplotlib.
Requirements: pip install matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def draw_er_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(24, 18))
    ax.set_xlim(-2, 22)
    ax.set_ylim(-2, 18)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ── Colors ──
    entity_color = '#D6EAF8'
    entity_border = '#2980B9'
    rel_color = '#FADBD8'
    rel_border = '#E74C3C'
    attr_color = '#FEF9E7'
    attr_border = '#F39C12'
    pk_color = '#D5F5E3'
    pk_border = '#27AE60'
    weak_color = '#F5EEF8'
    weak_border = '#8E44AD'

    def draw_entity(x, y, name, is_weak=False):
        w, h = 2.2, 0.9
        color = weak_color if is_weak else entity_color
        border = weak_border if is_weak else entity_border
        lw = 3 if is_weak else 2
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor=border,
                                        linewidth=lw)
        ax.add_patch(rect)
        if is_weak:
            inner = mpatches.FancyBboxPatch((x - w/2 + 0.08, y - h/2 + 0.08),
                                             w - 0.16, h - 0.16,
                                             boxstyle="round,pad=0.05",
                                             facecolor=color, edgecolor=border,
                                             linewidth=1.5)
            ax.add_patch(inner)
        ax.text(x, y, name, ha='center', va='center',
                fontsize=10, fontweight='bold', family='monospace')
        return (x, y)

    def draw_relationship(x, y, name):
        size = 1.2
        diamond = plt.Polygon([(x, y + size/2), (x + size/2, y),
                                (x, y - size/2), (x - size/2, y)],
                               facecolor=rel_color, edgecolor=rel_border,
                               linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y, name, ha='center', va='center',
                fontsize=7.5, fontweight='bold', family='monospace')
        return (x, y)

    def draw_attribute(x, y, name, is_pk=False, is_derived=False):
        w, h = 1.5, 0.5
        color = pk_color if is_pk else attr_color
        border = pk_border if is_pk else attr_border
        ellipse = mpatches.Ellipse((x, y), w, h, facecolor=color,
                                    edgecolor=border, linewidth=1.5)
        ax.add_patch(ellipse)
        ax.text(x, y, name, ha='center', va='center',
                fontsize=7, family='monospace',
                style='italic' if is_derived else 'normal',
                fontweight='bold' if is_pk else 'normal')
        if is_pk:
            ax.plot([x - len(name)*0.12, x + len(name)*0.12],
                    [y - 0.08, y - 0.08], color=pk_border, linewidth=1.5)
        return (x, y)

    def connect(p1, p2, label='', offset=0):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1.2, zorder=0)
        if label:
            mx = (p1[0] + p2[0]) / 2 + offset
            my = (p1[1] + p2[1]) / 2 + 0.2
            ax.text(mx, my, label, ha='center', va='center',
                    fontsize=8, fontweight='bold', color='#333')

    # ════════════════════════════════════════════════════════════
    # ENTITIES
    # ════════════════════════════════════════════════════════════
    user_pos = draw_entity(3, 9, 'USER')
    movie_pos = draw_entity(10, 9, 'MOVIE')
    genre_pos = draw_entity(17, 14, 'GENRE')
    director_pos = draw_entity(17, 9, 'DIRECTOR')
    actor_pos = draw_entity(17, 4, 'ACTOR')
    u_embed_pos = draw_entity(3, 3, 'USER_EMBEDDING', is_weak=True)
    m_embed_pos = draw_entity(10, 3, 'MOVIE_EMBEDDING', is_weak=True)

    # ════════════════════════════════════════════════════════════
    # RELATIONSHIPS
    # ════════════════════════════════════════════════════════════
    compares = draw_relationship(6.5, 12, 'COMPARES')
    has_genre = draw_relationship(13.5, 12, 'HAS_GENRE')
    directed_by = draw_relationship(13.5, 9, 'DIRECTED_BY')
    acted_in = draw_relationship(13.5, 6, 'ACTED_IN')
    has_u_embed = draw_relationship(3, 6, 'HAS_EMBED')
    has_m_embed = draw_relationship(10, 6, 'HAS_FEATURES')

    # ════════════════════════════════════════════════════════════
    # CONNECTIONS
    # ════════════════════════════════════════════════════════════
    connect(user_pos, compares, 'N')
    connect(movie_pos, compares, 'M')
    connect(movie_pos, has_genre, 'M')
    connect(genre_pos, has_genre, 'N')
    connect(movie_pos, directed_by, 'M')
    connect(director_pos, directed_by, 'N')
    connect(movie_pos, acted_in, 'M')
    connect(actor_pos, acted_in, 'N')
    connect(user_pos, has_u_embed, '1')
    connect(u_embed_pos, has_u_embed, 'N')
    connect(movie_pos, has_m_embed, '1')
    connect(m_embed_pos, has_m_embed, 'N')

    # ════════════════════════════════════════════════════════════
    # ATTRIBUTES — USER
    # ════════════════════════════════════════════════════════════
    draw_attribute(0.5, 11, 'user_id', is_pk=True)
    connect(user_pos, (0.5, 11))
    draw_attribute(2, 11.5, 'username')
    connect(user_pos, (2, 11.5))
    draw_attribute(3.5, 11.5, 'email')
    connect(user_pos, (3.5, 11.5))
    draw_attribute(5, 11, 'password')
    connect(user_pos, (5, 11))
    draw_attribute(1, 7.5, 'created_at')
    connect(user_pos, (1, 7.5))
    draw_attribute(4.5, 7.5, 'interaction_count')
    connect(user_pos, (4.5, 7.5))

    # ════════════════════════════════════════════════════════════
    # ATTRIBUTES — MOVIE
    # ════════════════════════════════════════════════════════════
    draw_attribute(7.5, 11, 'movie_id', is_pk=True)
    connect(movie_pos, (7.5, 11))
    draw_attribute(8.5, 10.5, 'title')
    connect(movie_pos, (8.5, 10.5))
    draw_attribute(10, 11.5, 'tmdb_id')
    connect(movie_pos, (10, 11.5))
    draw_attribute(11.5, 10.5, 'overview')
    connect(movie_pos, (11.5, 10.5))
    draw_attribute(12.5, 11, 'release_year')
    connect(movie_pos, (12.5, 11))
    draw_attribute(8.5, 7.5, 'tmdb_rating')
    connect(movie_pos, (8.5, 7.5))
    draw_attribute(10, 7.2, 'elo_score')
    connect(movie_pos, (10, 7.2))
    draw_attribute(11.5, 7.5, 'popularity')
    connect(movie_pos, (11.5, 7.5))

    # ════════════════════════════════════════════════════════════
    # ATTRIBUTES — GENRE
    # ════════════════════════════════════════════════════════════
    draw_attribute(15.5, 15.5, 'genre_id', is_pk=True)
    connect(genre_pos, (15.5, 15.5))
    draw_attribute(17, 15.8, 'genre_name')
    connect(genre_pos, (17, 15.8))
    draw_attribute(18.5, 15.5, 'tmdb_genre_id')
    connect(genre_pos, (18.5, 15.5))

    # ════════════════════════════════════════════════════════════
    # ATTRIBUTES — DIRECTOR
    # ════════════════════════════════════════════════════════════
    draw_attribute(19.5, 10.5, 'director_id', is_pk=True)
    connect(director_pos, (19.5, 10.5))
    draw_attribute(19.5, 9, 'director_name')
    connect(director_pos, (19.5, 9))
    draw_attribute(19.5, 7.5, 'popularity')
    connect(director_pos, (19.5, 7.5))

    # ════════════════════════════════════════════════════════════
    # ATTRIBUTES — ACTOR
    # ════════════════════════════════════════════════════════════
    draw_attribute(19.5, 5.5, 'actor_id', is_pk=True)
    connect(actor_pos, (19.5, 5.5))
    draw_attribute(19.5, 4, 'actor_name')
    connect(actor_pos, (19.5, 4))
    draw_attribute(19.5, 2.5, 'popularity')
    connect(actor_pos, (19.5, 2.5))

    # ════════════════════════════════════════════════════════════
    # ATTRIBUTES — ACTED_IN (Relationship attributes)
    # ════════════════════════════════════════════════════════════
    draw_attribute(15, 5.2, 'cast_order')
    connect(acted_in, (15, 5.2))
    draw_attribute(12, 5.2, 'character')
    connect(acted_in, (12, 5.2))

    # ════════════════════════════════════════════════════════════
    # ATTRIBUTES — COMPARES (Relationship attributes)
    # ════════════════════════════════════════════════════════════
    draw_attribute(5.5, 14, 'chosen_id')
    connect(compares, (5.5, 14))
    draw_attribute(7.5, 14, 'rejected_id')
    connect(compares, (7.5, 14))
    draw_attribute(6.5, 14.5, 'timestamp')
    connect(compares, (6.5, 14.5))
    draw_attribute(4, 13.5, 'session_id')
    connect(compares, (4, 13.5))

    # ════════════════════════════════════════════════════════════
    # ATTRIBUTES — WEAK ENTITIES
    # ════════════════════════════════════════════════════════════
    draw_attribute(1, 2.5, 'feat_index', is_pk=True)
    connect(u_embed_pos, (1, 2.5))
    draw_attribute(1, 1.5, 'feat_value')
    connect(u_embed_pos, (1, 1.5))
    draw_attribute(5, 2.5, 'last_updated')
    connect(u_embed_pos, (5, 2.5))

    draw_attribute(8, 2, 'feat_index', is_pk=True)
    connect(m_embed_pos, (8, 2))
    draw_attribute(12, 2, 'feat_value')
    connect(m_embed_pos, (12, 2))

    # ════════════════════════════════════════════════════════════
    # TITLE & LEGEND
    # ════════════════════════════════════════════════════════════
    ax.set_title('CineSense — Entity-Relationship Diagram\n(Chen Notation)',
                 fontsize=16, fontweight='bold', pad=20)

    legend_elements = [
        mpatches.Patch(facecolor=entity_color, edgecolor=entity_border,
                       linewidth=2, label='Strong Entity'),
        mpatches.Patch(facecolor=weak_color, edgecolor=weak_border,
                       linewidth=2, label='Weak Entity'),
        mpatches.Patch(facecolor=rel_color, edgecolor=rel_border,
                       linewidth=2, label='Relationship'),
        mpatches.Patch(facecolor=pk_color, edgecolor=pk_border,
                       linewidth=2, label='Primary Key Attribute'),
        mpatches.Patch(facecolor=attr_color, edgecolor=attr_border,
                       linewidth=2, label='Attribute'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig('er_diagram.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('er_diagram.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("ER Diagram saved: er_diagram.png, er_diagram.pdf")
    plt.show()


if __name__ == '__main__':
    draw_er_diagram()
