import sqlite3
import json
from datetime import datetime

import config


def get_connection():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            weibo_id TEXT UNIQUE NOT NULL,
            user_name TEXT,
            user_avatar TEXT,
            text TEXT,
            images TEXT,
            created_at TEXT,
            created_ts INTEGER DEFAULT 0,
            scraped_at TEXT
        )
    ''')
    # Add created_ts column if upgrading from older schema
    try:
        conn.execute('ALTER TABLE posts ADD COLUMN created_ts INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.execute('CREATE INDEX IF NOT EXISTS idx_created_ts ON posts(created_ts DESC)')
    conn.commit()
    conn.close()


def insert_post(post_data):
    """Insert a post, ignore if weibo_id already exists. Returns True if inserted."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            '''INSERT OR IGNORE INTO posts
               (weibo_id, user_name, user_avatar, text, images, created_at, created_ts, scraped_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                post_data['weibo_id'],
                post_data['user_name'],
                post_data['user_avatar'],
                post_data['text'],
                json.dumps(post_data.get('images', []), ensure_ascii=False),
                post_data['created_at'],
                post_data.get('created_ts', 0),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            )
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_posts(page=1, per_page=20):
    conn = get_connection()
    offset = (page - 1) * per_page
    rows = conn.execute(
        'SELECT * FROM posts ORDER BY created_ts DESC, id DESC LIMIT ? OFFSET ?',
        (per_page, offset)
    ).fetchall()
    conn.close()
    posts = []
    for row in rows:
        post = dict(row)
        post['images'] = json.loads(post['images']) if post['images'] else []
        posts.append(post)
    return posts


def get_post_count():
    conn = get_connection()
    count = conn.execute('SELECT COUNT(*) FROM posts').fetchone()[0]
    conn.close()
    return count


def get_last_scrape_time():
    conn = get_connection()
    row = conn.execute('SELECT MAX(scraped_at) FROM posts').fetchone()
    conn.close()
    return row[0] if row and row[0] else None
