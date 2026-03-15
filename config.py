import os
from dotenv import load_dotenv

load_dotenv()

WEIBO_COOKIE = os.getenv('WEIBO_COOKIE', '')
WEIBO_GROUP_ID = os.getenv('WEIBO_GROUP_ID', '')
SCRAPE_INTERVAL = int(os.getenv('SCRAPE_INTERVAL', '3600'))
PORT = int(os.getenv('PORT', '5000'))
DB_PATH = os.path.join(os.path.dirname(__file__), 'weibo.db')
