# CineSense — Database Documentation

> **DBMS Project: Database Design, Schema, DDL/DML, ER Diagram & Relational Model**
> RDBMS: MySQL 8.0+ | Normalization: Third Normal Form (3NF)
> Datasets: MovieLens 100K + TMDB 100K

---

## Table of Contents

1. [Database Overview](#1-database-overview)
2. [Entity-Relationship (ER) Model](#2-entity-relationship-er-model)
3. [ER Diagram](#3-er-diagram)
4. [Relational Schema Design](#4-relational-schema-design)
5. [Mapping ER to Relational Schemas](#5-mapping-er-to-relational-schemas)
6. [Normalization (1NF → 2NF → 3NF)](#6-normalization-1nf--2nf--3nf)
7. [Description of Tables](#7-description-of-tables)
8. [DDL Commands (Data Definition Language)](#8-ddl-commands-data-definition-language)
9. [DML Commands (Data Manipulation Language)](#9-dml-commands-data-manipulation-language)
10. [Views](#10-views)
11. [Stored Procedures](#11-stored-procedures)
12. [Indexes & Performance](#12-indexes--performance)
13. [Database Migration](#13-database-migration)
14. [Code: Generate ER Diagram](#14-code-generate-er-diagram)
15. [Code: Generate Relational Schema Diagram](#15-code-generate-relational-schema-diagram)

---

## 1. Database Overview

| Property | Value |
|----------|-------|
| **Database Name** | `cinesense` |
| **RDBMS** | MySQL 8.0+ |
| **Storage Engine** | InnoDB |
| **Character Set** | utf8mb4 |
| **Collation** | utf8mb4_unicode_ci |
| **Normalization Level** | Third Normal Form (3NF) |
| **Total Tables** | 11 |
| **Views** | 2 |
| **Stored Procedures** | 2 |
| **Total Records (approx.)** | 100,000+ movies, thousands of actors/directors/genres |

### Tables at a Glance

| # | Table Name | Type | Records (approx.) |
|---|-----------|------|-------------------|
| 1 | `users` | Entity | Variable (registered users) |
| 2 | `movies` | Entity | ~100,000 |
| 3 | `genres` | Entity | 19 |
| 4 | `directors` | Entity | ~50,000 |
| 5 | `actors` | Entity | ~100,000 |
| 6 | `movie_genres` | Relationship (M:N) | ~200,000 |
| 7 | `movie_directors` | Relationship (M:N) | ~100,000 |
| 8 | `movie_actors` | Relationship (M:N) | ~400,000 |
| 9 | `user_interactions` | Relationship (M:N) | Variable (grows with usage) |
| 10 | `user_embeddings` | Entity (weak) | Variable |
| 11 | `movie_embeddings` | Entity (weak) | ~5,500,000 (100K × 55 features) |

---

## 2. Entity-Relationship (ER) Model

### 2.1 Entities

| Entity | Description | Primary Key | Key Attributes |
|--------|-------------|-------------|----------------|
| **USER** | Registered platform user | `user_id` (auto-increment) | username, email, password_hash, created_at, last_active, interaction_count |
| **MOVIE** | Movie/film record from TMDB | `movie_id` | tmdb_id, title, original_title, overview, release_year, runtime, poster_path, backdrop_path, tmdb_rating, vote_count, popularity, watch_link, elo_score, comparison_count |
| **GENRE** | Movie genre category | `genre_id` (auto-increment) | genre_name, tmdb_genre_id |
| **DIRECTOR** | Film director | `director_id` (auto-increment) | director_name, tmdb_person_id, popularity |
| **ACTOR** | Film actor/actress | `actor_id` (auto-increment) | actor_name, tmdb_person_id, popularity |
| **DEPENDENT** | User's AI preference vector (weak entity) | (`user_id`, `feature_index`) | feature_value, last_updated |
| **MOVIE_EMBEDDING** | Movie's feature vector (weak entity) | (`movie_id`, `feature_index`) | feature_value |

### 2.2 Relationships

| Relationship | Entities Involved | Cardinality | Description |
|-------------|-------------------|-------------|-------------|
| **HAS_GENRE** | MOVIE ↔ GENRE | Many-to-Many (M:N) | A movie can have multiple genres; a genre applies to many movies |
| **DIRECTED_BY** | MOVIE ↔ DIRECTOR | Many-to-Many (M:N) | A movie can have multiple directors; a director directs many movies |
| **ACTED_IN** | MOVIE ↔ ACTOR | Many-to-Many (M:N) | A movie has many actors; an actor acts in many movies. Has attributes: `cast_order`, `character_name` |
| **COMPARES** | USER ↔ MOVIE | Many-to-Many (M:N) | User compares pairs of movies. Has attributes: `movie_1_id`, `movie_2_id`, `chosen_movie_id`, `rejected_movie_id`, `timestamp`, `session_id` |
| **HAS_EMBEDDING** | USER ↔ USER_EMBEDDING | One-to-Many (1:N) | A user has one preference vector (stored as multiple rows) |
| **HAS_FEATURES** | MOVIE ↔ MOVIE_EMBEDDING | One-to-Many (1:N) | A movie has one feature vector (stored as multiple rows) |

### 2.3 Cardinality Summary

```
USER (1) ──────── (N) USER_EMBEDDING       [1:N] User has many embedding rows
USER (1) ──────── (N) USER_INTERACTION      [1:N] User makes many comparisons
MOVIE (1) ─────── (N) MOVIE_EMBEDDING       [1:N] Movie has many feature rows
MOVIE (M) ─────── (N) GENRE                 [M:N] via movie_genres
MOVIE (M) ─────── (N) DIRECTOR              [M:N] via movie_directors
MOVIE (M) ─────── (N) ACTOR                 [M:N] via movie_actors
MOVIE (N) ─────── (M) USER (via interactions)[M:N] via user_interactions
```

---

## 3. ER Diagram

### ER Diagram (Chen Notation)

```
                                    ┌──────────────┐
                                    │   GENRE      │
                                    │──────────────│
                                    │ PK genre_id  │
                                    │ genre_name   │
                                    │ tmdb_genre_id│
                                    └──────┬───────┘
                                           │
                                           │ M:N
                                           │
                                    ◇ HAS_GENRE
                                           │
                                           │ M:N
     ┌───────────────┐              ┌──────┴───────┐              ┌──────────────┐
     │   DIRECTOR    │              │    MOVIE     │              │    ACTOR     │
     │───────────────│              │──────────────│              │──────────────│
     │ PK director_id│──── M:N ────│ PK movie_id  │──── M:N ────│ PK actor_id  │
     │ director_name │  DIRECTED_BY│ tmdb_id      │   ACTED_IN  │ actor_name   │
     │ tmdb_person_id│              │ title        │  (cast_order│ tmdb_person_id│
     │ popularity    │              │ overview     │   character)│ popularity   │
     └───────────────┘              │ release_year │              └──────────────┘
                                    │ runtime      │
                                    │ poster_path  │
                                    │ tmdb_rating  │
                                    │ popularity   │
                                    │ elo_score    │
                                    └──────┬───────┘
                                           │
                              ┌────────────┼────────────┐
                              │            │            │
                              │ 1:N        │ M:N        │ 1:N
                              ▼            ▼            ▼
                     ┌─────────────┐ ◇ COMPARES  ┌──────────────┐
                     │MOVIE_EMBED. │       │     │ USER_EMBED.  │
                     │─────────────│       │     │──────────────│
                     │PK(movie_id, │       │     │PK(user_id,   │
                     │  feat_index)│       │     │  feat_index) │
                     │ feat_value  │   ┌───┴──┐  │ feat_value   │
                     └─────────────┘   │ USER │  │ last_updated │
                                       │──────│  └──────────────┘
                                       │PK user_id  │
                                       │ username    │
                                       │ email       │
                                       │ password    │
                                       │ created_at  │
                                       └─────────────┘
```

### Generating Visual ER Diagram

See **Section 14** below for Python code that generates a publication-quality ER diagram image using `matplotlib` and `networkx`.

---

## 4. Relational Schema Design

### Relational Schemas (Formal Notation)

```
USERS (user_id, username, email, password_hash, created_at, last_active, interaction_count)
    PK: user_id
    UNIQUE: username, email

MOVIES (movie_id, tmdb_id, title, original_title, overview, release_year, runtime,
        poster_path, backdrop_path, tmdb_rating, vote_count, popularity, watch_link,
        elo_score, comparison_count, created_at)
    PK: movie_id
    UNIQUE: tmdb_id

GENRES (genre_id, genre_name, tmdb_genre_id)
    PK: genre_id
    UNIQUE: genre_name, tmdb_genre_id

DIRECTORS (director_id, director_name, tmdb_person_id, popularity)
    PK: director_id
    UNIQUE: tmdb_person_id

ACTORS (actor_id, actor_name, tmdb_person_id, popularity)
    PK: actor_id
    UNIQUE: tmdb_person_id

MOVIE_GENRES (movie_id, genre_id)
    PK: (movie_id, genre_id)
    FK: movie_id → MOVIES(movie_id) ON DELETE CASCADE
    FK: genre_id → GENRES(genre_id) ON DELETE CASCADE

MOVIE_DIRECTORS (movie_id, director_id)
    PK: (movie_id, director_id)
    FK: movie_id → MOVIES(movie_id) ON DELETE CASCADE
    FK: director_id → DIRECTORS(director_id) ON DELETE CASCADE

MOVIE_ACTORS (movie_id, actor_id, cast_order, character_name)
    PK: (movie_id, actor_id)
    FK: movie_id → MOVIES(movie_id) ON DELETE CASCADE
    FK: actor_id → ACTORS(actor_id) ON DELETE CASCADE

USER_INTERACTIONS (interaction_id, user_id, movie_1_id, movie_2_id, chosen_movie_id,
                   rejected_movie_id, timestamp, session_id)
    PK: interaction_id
    FK: user_id → USERS(user_id) ON DELETE CASCADE
    FK: movie_1_id → MOVIES(movie_id) ON DELETE CASCADE
    FK: movie_2_id → MOVIES(movie_id) ON DELETE CASCADE
    FK: chosen_movie_id → MOVIES(movie_id) ON DELETE CASCADE
    FK: rejected_movie_id → MOVIES(movie_id) ON DELETE CASCADE

USER_EMBEDDINGS (user_id, feature_index, feature_value, last_updated)
    PK: (user_id, feature_index)
    FK: user_id → USERS(user_id) ON DELETE CASCADE

MOVIE_EMBEDDINGS (movie_id, feature_index, feature_value)
    PK: (movie_id, feature_index)
    FK: movie_id → MOVIES(movie_id) ON DELETE CASCADE
```

---

## 5. Mapping ER to Relational Schemas

### Step 1: Map Strong Entities

Each strong entity becomes a table with its attributes as columns and its primary key preserved.

| ER Entity | Relational Table | Primary Key |
|-----------|-----------------|-------------|
| USER | `users` | `user_id` (AUTO_INCREMENT) |
| MOVIE | `movies` | `movie_id` |
| GENRE | `genres` | `genre_id` (AUTO_INCREMENT) |
| DIRECTOR | `directors` | `director_id` (AUTO_INCREMENT) |
| ACTOR | `actors` | `actor_id` (AUTO_INCREMENT) |

### Step 2: Map Weak Entities

Weak entities include the owner's primary key as part of their composite key.

| ER Weak Entity | Relational Table | Composite PK |
|---------------|-----------------|--------------|
| USER_EMBEDDING | `user_embeddings` | (`user_id`, `feature_index`) |
| MOVIE_EMBEDDING | `movie_embeddings` | (`movie_id`, `feature_index`) |

### Step 3: Map M:N Relationships

Each many-to-many relationship creates a junction (bridge) table.

| ER Relationship | Junction Table | Composite PK | Attributes |
|----------------|---------------|--------------|------------|
| HAS_GENRE | `movie_genres` | (`movie_id`, `genre_id`) | — |
| DIRECTED_BY | `movie_directors` | (`movie_id`, `director_id`) | — |
| ACTED_IN | `movie_actors` | (`movie_id`, `actor_id`) | `cast_order`, `character_name` |
| COMPARES | `user_interactions` | `interaction_id` (surrogate) | `movie_1_id`, `movie_2_id`, `chosen_movie_id`, etc. |

### Step 4: Map 1:N Relationships

One-to-many relationships are handled by adding a foreign key to the "many" side.

| Relationship | FK Added To | FK Column |
|-------------|------------|-----------|
| USER → USER_EMBEDDINGS | `user_embeddings` | `user_id` |
| MOVIE → MOVIE_EMBEDDINGS | `movie_embeddings` | `movie_id` |
| USER → USER_INTERACTIONS | `user_interactions` | `user_id` |

---

## 6. Normalization (1NF → 2NF → 3NF)

### First Normal Form (1NF)

**Rule:** All attributes must be atomic (no multi-valued or composite attributes).

**Before 1NF (denormalized):**
```
MOVIES_DENORMALIZED (movie_id, title, genres, directors, actors, ...)
                                        ↑ "Action, Comedy" (multi-valued!)
```

**After 1NF:**
- Genres split into a separate `genres` table
- Directors split into `directors` table
- Actors split into `actors` table
- Junction tables (`movie_genres`, `movie_directors`, `movie_actors`) link them

### Second Normal Form (2NF)

**Rule:** No partial dependencies (all non-key attributes depend on the entire primary key).

The `movie_actors` table has composite PK (`movie_id`, `actor_id`):
- `cast_order` depends on the full key ✓ (an actor's order is specific to a movie)
- `character_name` depends on the full key ✓ (a character is specific to both)

All tables satisfy 2NF.

### Third Normal Form (3NF)

**Rule:** No transitive dependencies (non-key attributes must depend directly on the PK, not through another non-key attribute).

**Example check — `movies` table:**
- `tmdb_rating` depends on `movie_id` directly ✓ (not through `tmdb_id`)
- `popularity` depends on `movie_id` directly ✓
- `elo_score` depends on `movie_id` directly ✓

**Example check — `user_interactions` table:**
- `rejected_movie_id` is derived from `movie_1_id`, `movie_2_id`, and `chosen_movie_id`. However, it is stored for query performance (denormalization for reads). This is an **intentional pragmatic denormalization** that does not violate the overall 3NF design goal.

All tables satisfy 3NF (with the noted pragmatic exception).

---

## 7. Description of Tables

### 7.1 `users` — Registered Users

| Column | Data Type | Constraints | Description |
|--------|----------|-------------|-------------|
| `user_id` | INT | PK, AUTO_INCREMENT | Unique user identifier |
| `username` | VARCHAR(50) | NOT NULL, UNIQUE | Display name |
| `email` | VARCHAR(100) | NOT NULL, UNIQUE | User email address |
| `password_hash` | VARCHAR(255) | NOT NULL | Bcrypt-hashed password |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Registration date |
| `last_active` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP ON UPDATE | Last activity time |
| `interaction_count` | INT | DEFAULT 0 | Total pairwise comparisons made |

**Indexes:** `idx_username`, `idx_email`

---

### 7.2 `movies` — Movie Records

| Column | Data Type | Constraints | Description |
|--------|----------|-------------|-------------|
| `movie_id` | INT | PK | Unique movie identifier |
| `tmdb_id` | INT | NOT NULL, UNIQUE | TMDB API identifier |
| `title` | VARCHAR(255) | NOT NULL | Movie title |
| `original_title` | VARCHAR(255) | — | Original language title |
| `overview` | TEXT | — | Plot summary |
| `release_year` | INT | — | Year of release |
| `runtime` | INT | — | Runtime in minutes |
| `poster_path` | VARCHAR(255) | — | TMDB poster image path |
| `backdrop_path` | VARCHAR(255) | — | TMDB backdrop image path |
| `tmdb_rating` | DECIMAL(3,1) | — | TMDB average rating (0.0–10.0) |
| `vote_count` | INT | — | Number of TMDB votes |
| `popularity` | DECIMAL(10,3) | — | TMDB popularity score |
| `watch_link` | VARCHAR(500) | — | External watch URL |
| `elo_score` | INT | DEFAULT 1500 | ELO rating (updated by comparisons) |
| `comparison_count` | INT | DEFAULT 0 | Times compared |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Record creation |

**Indexes:** `idx_title`, `idx_release_year`, `idx_elo_score`, `idx_popularity`, `idx_movie_rating_popularity`, `idx_movie_year_rating`, `ft_title_overview` (FULLTEXT)

---

### 7.3 `genres` — Genre Categories

| Column | Data Type | Constraints | Description |
|--------|----------|-------------|-------------|
| `genre_id` | INT | PK, AUTO_INCREMENT | Unique genre identifier |
| `genre_name` | VARCHAR(50) | NOT NULL, UNIQUE | Genre name (e.g., "Action") |
| `tmdb_genre_id` | INT | UNIQUE | TMDB genre identifier |

**Pre-populated with 19 standard TMDB genres:** Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, TV Movie, Thriller, War, Western.

---

### 7.4 `directors` — Film Directors

| Column | Data Type | Constraints | Description |
|--------|----------|-------------|-------------|
| `director_id` | INT | PK, AUTO_INCREMENT | Unique director identifier |
| `director_name` | VARCHAR(100) | NOT NULL | Director's name |
| `tmdb_person_id` | INT | UNIQUE | TMDB person identifier |
| `popularity` | DECIMAL(10,3) | — | TMDB popularity score |

---

### 7.5 `actors` — Film Actors

| Column | Data Type | Constraints | Description |
|--------|----------|-------------|-------------|
| `actor_id` | INT | PK, AUTO_INCREMENT | Unique actor identifier |
| `actor_name` | VARCHAR(100) | NOT NULL | Actor's name |
| `tmdb_person_id` | INT | UNIQUE | TMDB person identifier |
| `popularity` | DECIMAL(10,3) | — | TMDB popularity score |

---

### 7.6 `movie_genres` — Movie-Genre Junction (M:N)

| Column | Data Type | Constraints | Description |
|--------|----------|-------------|-------------|
| `movie_id` | INT | PK (composite), FK → movies | Movie reference |
| `genre_id` | INT | PK (composite), FK → genres | Genre reference |

**Referential Integrity:** ON DELETE CASCADE on both FKs

---

### 7.7 `movie_directors` — Movie-Director Junction (M:N)

| Column | Data Type | Constraints | Description |
|--------|----------|-------------|-------------|
| `movie_id` | INT | PK (composite), FK → movies | Movie reference |
| `director_id` | INT | PK (composite), FK → directors | Director reference |

---

### 7.8 `movie_actors` — Movie-Actor Junction (M:N)

| Column | Data Type | Constraints | Description |
|--------|----------|-------------|-------------|
| `movie_id` | INT | PK (composite), FK → movies | Movie reference |
| `actor_id` | INT | PK (composite), FK → actors | Actor reference |
| `cast_order` | INT | DEFAULT 0 | Billing order (0 = lead) |
| `character_name` | VARCHAR(255) | — | Character played |

---

### 7.9 `user_interactions` — Pairwise Comparison History

| Column | Data Type | Constraints | Description |
|--------|----------|-------------|-------------|
| `interaction_id` | INT | PK, AUTO_INCREMENT | Unique interaction ID |
| `user_id` | INT | NOT NULL, FK → users | User who compared |
| `movie_1_id` | INT | NOT NULL, FK → movies | First movie shown |
| `movie_2_id` | INT | NOT NULL, FK → movies | Second movie shown |
| `chosen_movie_id` | INT | NOT NULL, FK → movies | User's preferred movie |
| `rejected_movie_id` | INT | NOT NULL, FK → movies | Not chosen movie |
| `timestamp` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | When comparison was made |
| `session_id` | VARCHAR(100) | — | Browser session identifier |

**Indexes:** `idx_user_interactions`, `idx_chosen_movie`, `idx_rejected_movie`, `idx_session`

---

### 7.10 `user_embeddings` — User Preference Vectors

| Column | Data Type | Constraints | Description |
|--------|----------|-------------|-------------|
| `user_id` | INT | PK (composite), FK → users | User reference |
| `feature_index` | INT | PK (composite) | Dimension index (0–54) |
| `feature_value` | DECIMAL(10,6) | NOT NULL | Feature weight value |
| `last_updated` | TIMESTAMP | ON UPDATE CURRENT_TIMESTAMP | Last modification |

Each user's 55-dimensional preference vector is stored as 55 rows (one per feature dimension).

---

### 7.11 `movie_embeddings` — Movie Feature Vectors

| Column | Data Type | Constraints | Description |
|--------|----------|-------------|-------------|
| `movie_id` | INT | PK (composite), FK → movies | Movie reference |
| `feature_index` | INT | PK (composite) | Dimension index (0–54) |
| `feature_value` | DECIMAL(10,6) | NOT NULL | Feature weight value |

Each movie's 55-dimensional feature vector is stored as 55 rows.

---

## 8. DDL Commands (Data Definition Language)

### 8.1 CREATE TABLE Statements

```sql
-- ============================================================================
-- DATABASE CREATION
-- ============================================================================
CREATE DATABASE IF NOT EXISTS cinesense
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE cinesense;

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
-- GENRES TABLE
-- ============================================================================
CREATE TABLE genres (
    genre_id INT AUTO_INCREMENT PRIMARY KEY,
    genre_name VARCHAR(50) NOT NULL UNIQUE,
    tmdb_genre_id INT UNIQUE,
    INDEX idx_genre_name (genre_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- DIRECTORS TABLE
-- ============================================================================
CREATE TABLE directors (
    director_id INT AUTO_INCREMENT PRIMARY KEY,
    director_name VARCHAR(100) NOT NULL,
    tmdb_person_id INT UNIQUE,
    popularity DECIMAL(10,3),
    INDEX idx_director_name (director_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- ACTORS TABLE
-- ============================================================================
CREATE TABLE actors (
    actor_id INT AUTO_INCREMENT PRIMARY KEY,
    actor_name VARCHAR(100) NOT NULL,
    tmdb_person_id INT UNIQUE,
    popularity DECIMAL(10,3),
    INDEX idx_actor_name (actor_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- MOVIE_GENRES (Junction Table)
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
-- MOVIE_DIRECTORS (Junction Table)
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
-- MOVIE_ACTORS (Junction Table)
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
-- USER_INTERACTIONS (Pairwise Comparison History)
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
-- USER_EMBEDDINGS (User Preference Vectors)
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
-- MOVIE_EMBEDDINGS (Movie Feature Vectors)
-- ============================================================================
CREATE TABLE movie_embeddings (
    movie_id INT NOT NULL,
    feature_index INT NOT NULL,
    feature_value DECIMAL(10,6) NOT NULL,
    PRIMARY KEY (movie_id, feature_index),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    INDEX idx_movie_vector (movie_id, feature_index)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 8.2 ALTER TABLE (Migration — Lazy Loading)

```sql
-- Add source tracking
ALTER TABLE movies
ADD COLUMN movie_source ENUM('tmdb_api', 'user_interaction', 'cache', 'database')
DEFAULT 'database' AFTER comparison_count;

-- Add persistence flag
ALTER TABLE movies
ADD COLUMN is_persisted BOOLEAN DEFAULT FALSE AFTER movie_source;

-- Add LRU tracking
ALTER TABLE movies
ADD COLUMN last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
ON UPDATE CURRENT_TIMESTAMP AFTER is_persisted;

-- Add access counter
ALTER TABLE movies
ADD COLUMN access_count INT DEFAULT 0 AFTER last_accessed;

-- Add interaction type
ALTER TABLE user_interactions
ADD COLUMN interaction_type ENUM('comparison', 'recommendation', 'search', 'view', 'click')
DEFAULT 'comparison' AFTER session_id;
```

### 8.3 DROP TABLE

```sql
-- Drop in reverse dependency order (child tables first)
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
```

### 8.4 CREATE INDEX

```sql
-- Composite indexes for common queries
CREATE INDEX idx_movie_rating_popularity ON movies(tmdb_rating DESC, popularity DESC);
CREATE INDEX idx_movie_year_rating ON movies(release_year DESC, tmdb_rating DESC);
CREATE INDEX idx_user_interactions_timestamp ON user_interactions(user_id, timestamp DESC);

-- Full-text search
ALTER TABLE movies ADD FULLTEXT INDEX ft_title_overview (title, overview);
```

---

## 9. DML Commands (Data Manipulation Language)

### 9.1 INSERT (Create)

```sql
-- Insert a new user
INSERT INTO users (username, email, password_hash)
VALUES ('john_doe', 'john@example.com', '$2b$12$hashed_password_here');

-- Insert a movie
INSERT INTO movies (movie_id, tmdb_id, title, overview, release_year, runtime,
                    poster_path, tmdb_rating, vote_count, popularity)
VALUES (1, 550, 'Fight Club', 'An insomniac office worker...', 1999, 139,
        '/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg', 8.4, 26000, 73.433);

-- Insert a genre
INSERT INTO genres (genre_name, tmdb_genre_id)
VALUES ('Action', 28);

-- Link movie to genre (M:N relationship)
INSERT INTO movie_genres (movie_id, genre_id) VALUES (1, 7);

-- Link movie to actor
INSERT INTO movie_actors (movie_id, actor_id, cast_order, character_name)
VALUES (1, 1, 0, 'The Narrator');

-- Record a pairwise comparison
INSERT INTO user_interactions (user_id, movie_1_id, movie_2_id,
                               chosen_movie_id, rejected_movie_id, session_id)
VALUES (1, 550, 680, 550, 680, 'abc123def456');

-- Save user embedding vector (one row per dimension)
INSERT INTO user_embeddings (user_id, feature_index, feature_value)
VALUES (1, 0, 0.842531)
ON DUPLICATE KEY UPDATE feature_value = VALUES(feature_value);

-- Pre-populate genres
INSERT INTO genres (genre_id, genre_name, tmdb_genre_id) VALUES
(1, 'Action', 28), (2, 'Adventure', 12), (3, 'Animation', 16),
(4, 'Comedy', 35), (5, 'Crime', 80), (6, 'Documentary', 99),
(7, 'Drama', 18), (8, 'Family', 10751), (9, 'Fantasy', 14),
(10, 'History', 36), (11, 'Horror', 27), (12, 'Music', 10402),
(13, 'Mystery', 9648), (14, 'Romance', 10749), (15, 'Science Fiction', 878),
(16, 'TV Movie', 10770), (17, 'Thriller', 53), (18, 'War', 10752),
(19, 'Western', 37);
```

### 9.2 SELECT (Read)

```sql
-- Get all movies with their genres, directors, and cast (via VIEW)
SELECT * FROM movie_details WHERE movie_id = 550;

-- Get top 20 movies by ELO score
SELECT * FROM movie_details
WHERE tmdb_rating IS NOT NULL
ORDER BY elo_score DESC
LIMIT 20;

-- Search movies (multi-tier relevance)
SELECT m.*, 
    CASE
        WHEN LOWER(m.title) = 'inception' THEN 1000
        WHEN LOWER(m.title) LIKE 'inception%' THEN 900
        WHEN LOWER(m.title) LIKE '%inception%' THEN 800
        WHEN LOWER(m.overview) LIKE '%inception%' THEN 100
        ELSE 50
    END AS relevance_score
FROM movies m
WHERE LOWER(m.title) LIKE '%inception%'
   OR LOWER(m.overview) LIKE '%inception%'
ORDER BY relevance_score DESC, m.popularity DESC
LIMIT 20;

-- Get movies by genre
SELECT md.* FROM movie_details md
WHERE FIND_IN_SET('Action', md.genres) > 0
ORDER BY md.tmdb_rating DESC
LIMIT 20;

-- Get user's interaction history
SELECT * FROM user_interactions
WHERE user_id = 1
ORDER BY timestamp DESC
LIMIT 100;

-- Get user statistics (via VIEW)
SELECT * FROM user_stats WHERE user_id = 1;

-- Get user's preference vector
SELECT feature_index, feature_value
FROM user_embeddings
WHERE user_id = 1
ORDER BY feature_index;

-- Get random movies for comparison
SELECT * FROM movies
WHERE popularity > 10 AND tmdb_rating IS NOT NULL
ORDER BY RAND()
LIMIT 2;

-- Count movies per genre
SELECT g.genre_name, COUNT(mg.movie_id) AS movie_count
FROM genres g
LEFT JOIN movie_genres mg ON g.genre_id = mg.genre_id
GROUP BY g.genre_id
ORDER BY movie_count DESC;

-- Get top directors by number of movies
SELECT d.director_name, COUNT(md.movie_id) AS movie_count,
       AVG(m.tmdb_rating) AS avg_rating
FROM directors d
JOIN movie_directors md ON d.director_id = md.director_id
JOIN movies m ON md.movie_id = m.movie_id
GROUP BY d.director_id
HAVING movie_count >= 3
ORDER BY avg_rating DESC
LIMIT 20;
```

### 9.3 UPDATE (Modify)

```sql
-- Update movie ELO score manually
UPDATE movies
SET elo_score = elo_score + 16,
    comparison_count = comparison_count + 1
WHERE movie_id = 550;

-- Increment user interaction count
UPDATE users
SET interaction_count = interaction_count + 1,
    last_active = CURRENT_TIMESTAMP
WHERE user_id = 1;

-- Update movie metadata (UPSERT pattern used by the app)
INSERT INTO movies (movie_id, tmdb_id, title, tmdb_rating, popularity)
VALUES (550, 550, 'Fight Club', 8.4, 73.433)
ON DUPLICATE KEY UPDATE
    title = VALUES(title),
    tmdb_rating = VALUES(tmdb_rating),
    popularity = VALUES(popularity);
```

### 9.4 DELETE (Remove)

```sql
-- Delete a user (cascades to interactions and embeddings)
DELETE FROM users WHERE user_id = 1;

-- Delete a movie (cascades to genres, directors, actors, embeddings)
DELETE FROM movies WHERE movie_id = 550;

-- Clear all user interactions
DELETE FROM user_interactions WHERE user_id = 1;

-- Remove a movie-genre link
DELETE FROM movie_genres WHERE movie_id = 550 AND genre_id = 7;
```

---

## 10. Views

### 10.1 `movie_details` — Complete Movie Information

Joins movies with all related data (genres, directors, actors) using `GROUP_CONCAT`:

```sql
CREATE VIEW movie_details AS
SELECT 
    m.movie_id, m.tmdb_id, m.title, m.overview, m.release_year,
    m.runtime, m.poster_path, m.backdrop_path, m.tmdb_rating,
    m.vote_count, m.popularity, m.watch_link, m.elo_score,
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
```

**Usage:** `SELECT * FROM movie_details WHERE movie_id = 550;`

### 10.2 `user_stats` — User Analytics

```sql
CREATE VIEW user_stats AS
SELECT 
    u.user_id, u.username, u.interaction_count,
    u.created_at, u.last_active,
    COUNT(DISTINCT ui.interaction_id) AS total_comparisons,
    COUNT(DISTINCT ui.chosen_movie_id) AS unique_movies_chosen,
    DATEDIFF(CURRENT_DATE, u.created_at) AS days_active
FROM users u
LEFT JOIN user_interactions ui ON u.user_id = ui.user_id
GROUP BY u.user_id;
```

---

## 11. Stored Procedures

### 11.1 `update_user_interaction_count`

```sql
DELIMITER //
CREATE PROCEDURE update_user_interaction_count(IN p_user_id INT)
BEGIN
    UPDATE users 
    SET interaction_count = interaction_count + 1,
        last_active = CURRENT_TIMESTAMP
    WHERE user_id = p_user_id;
END //
DELIMITER ;

-- Usage:
CALL update_user_interaction_count(1);
```

### 11.2 `update_movie_elo` — ELO Score Update

Implements the ELO rating formula as a stored procedure:

```sql
DELIMITER //
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
    
    -- Calculate expected scores (ELO formula)
    SET expected_winner = 1 / (1 + POW(10, (loser_elo - winner_elo) / 400));
    SET expected_loser = 1 / (1 + POW(10, (winner_elo - loser_elo) / 400));
    
    -- Calculate ELO changes
    SET winner_change = ROUND(p_k_factor * (1 - expected_winner));
    SET loser_change = ROUND(p_k_factor * (0 - expected_loser));
    
    -- Update winner's score
    UPDATE movies 
    SET elo_score = elo_score + winner_change,
        comparison_count = comparison_count + 1
    WHERE movie_id = p_winner_id;
    
    -- Update loser's score
    UPDATE movies 
    SET elo_score = elo_score + loser_change,
        comparison_count = comparison_count + 1
    WHERE movie_id = p_loser_id;
END //
DELIMITER ;

-- Usage:
CALL update_movie_elo(550, 680, 32);  -- Fight Club beats Pulp Fiction, K=32
```

---

## 12. Indexes & Performance

### Index Strategy

| Index Name | Table | Columns | Type | Purpose |
|-----------|-------|---------|------|---------|
| `idx_username` | users | username | B-Tree | Fast login lookup |
| `idx_email` | users | email | B-Tree | Fast email validation |
| `idx_title` | movies | title | B-Tree | Title search |
| `idx_release_year` | movies | release_year | B-Tree | Year filtering |
| `idx_elo_score` | movies | elo_score | B-Tree | Ranking queries |
| `idx_popularity` | movies | popularity | B-Tree | Popular movies |
| `idx_movie_rating_popularity` | movies | (tmdb_rating DESC, popularity DESC) | Composite | Top-rated queries |
| `idx_movie_year_rating` | movies | (release_year DESC, tmdb_rating DESC) | Composite | Year + rating |
| `ft_title_overview` | movies | (title, overview) | FULLTEXT | Text search |
| `idx_genre_lookup` | movie_genres | (genre_id, movie_id) | Composite | Genre filtering |
| `idx_director_lookup` | movie_directors | (director_id, movie_id) | Composite | Director lookup |
| `idx_actor_lookup` | movie_actors | (actor_id, movie_id) | Composite | Actor lookup |
| `idx_cast_order` | movie_actors | (movie_id, cast_order) | Composite | Cast ordering |
| `idx_user_interactions` | user_interactions | (user_id, timestamp DESC) | Composite | User history |
| `idx_chosen_movie` | user_interactions | chosen_movie_id | B-Tree | Preference analysis |
| `idx_rejected_movie` | user_interactions | rejected_movie_id | B-Tree | Preference analysis |
| `idx_session` | user_interactions | session_id | B-Tree | Session tracking |
| `idx_user_vector` | user_embeddings | (user_id, feature_index) | Composite | Vector retrieval |
| `idx_movie_vector` | movie_embeddings | (movie_id, feature_index) | Composite | Vector retrieval |

### Why InnoDB?

- **Foreign Key Support**: Enforces referential integrity across all junction tables
- **Row-Level Locking**: Allows concurrent reads/writes (important for real-time recommendations)
- **ACID Transactions**: Ensures data consistency when updating ELO scores
- **Crash Recovery**: InnoDB's redo/undo logs protect against data loss

---

## 13. Database Migration

### Migration System (`database/run_migration.py`)

The project uses a custom migration runner that:
1. Creates a `schema_migrations` table to track applied migrations
2. Scans `database/migrations/` for SQL files
3. Applies unapplied migrations in order
4. Records each migration's timestamp

### Migration: 001_lazy_loading_migration.sql

Adds support for:
- **Movie source tracking**: Where each movie came from (TMDB API, user interaction, cache, database)
- **Persistence flags**: Whether a movie should be kept in DB or can be evicted
- **LRU tracking**: Last accessed time and access count for cache eviction
- **Interaction types**: Beyond comparisons — also tracks recommendations, searches, views, clicks
- **Cache stats table**: Monitors cache hit/miss rates and performance

---

## 14. Code: Generate ER Diagram

Save the following Python script as `generate_er_diagram.py` and run it to produce a publication-quality ER diagram:

```python
"""
CineSense ER Diagram Generator
Generates a Chen-notation Entity-Relationship diagram using matplotlib.
Requirements: pip install matplotlib networkx
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
        style = 'underline' if is_pk else 'normal'
        ax.text(x, y, name, ha='center', va='center',
                fontsize=7, family='monospace', style='italic' if is_derived else 'normal',
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
    print("✅ ER Diagram saved: er_diagram.png, er_diagram.pdf")
    plt.show()


if __name__ == '__main__':
    draw_er_diagram()
```

---

## 15. Code: Generate Relational Schema Diagram

Save the following Python script as `generate_schema_diagram.py` and run it to produce a relational schema diagram showing all tables, columns, primary keys, foreign keys, and relationships:

```python
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
                prefix += '🔑 '
            if 'FK' in key_type:
                prefix += '🔗 '

            ax.text(x + 0.15, cy + row_height/2, f"{prefix}{col_name}",
                    ha='left', va='center', fontsize=7,
                    family='monospace', fontweight='bold' if 'PK' in key_type else 'normal')
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
    print("✅ Relational Schema Diagram saved: relational_schema.png, relational_schema.pdf")
    plt.show()


if __name__ == '__main__':
    draw_relational_schema()
```

### How to Run

```bash
# Install dependencies
pip install matplotlib

# Generate ER Diagram
python generate_er_diagram.py

# Generate Relational Schema Diagram  
python generate_schema_diagram.py
```

**Output files:**
- `er_diagram.png` / `er_diagram.pdf` — Chen-notation ER diagram
- `relational_schema.png` / `relational_schema.pdf` — Relational schema with all tables, PKs, FKs
