import math
import logging

from flask import Flask, render_template, request, redirect, url_for, flash
from apscheduler.schedulers.background import BackgroundScheduler

import config
import database
import scraper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'weibo-lu-secret'

PER_PAGE = 20


def scrape_job():
    logger.info('Scheduled scrape starting...')
    try:
        count = scraper.scrape_latest()
        logger.info('Scheduled scrape finished, %d new posts', count)
    except Exception:
        logger.exception('Scheduled scrape failed')


@app.route('/')
def index():
    page = request.args.get('page', 1, type=int)
    page = max(1, page)
    total = database.get_post_count()
    total_pages = max(1, math.ceil(total / PER_PAGE))
    page = min(page, total_pages)
    posts = database.get_posts(page=page, per_page=PER_PAGE)
    last_scrape = database.get_last_scrape_time()
    return render_template(
        'index.html',
        posts=posts,
        page=page,
        total_pages=total_pages,
        total=total,
        last_scrape=last_scrape,
    )


@app.route('/scrape')
def manual_scrape():
    count = scraper.scrape_latest()
    flash(f'抓取完成，新增 {count} 条微博')
    return redirect(url_for('index'))


if __name__ == '__main__':
    database.init_db()

    scheduler = BackgroundScheduler()
    scheduler.add_job(scrape_job, 'interval', seconds=config.SCRAPE_INTERVAL,
                      id='weibo_scraper', replace_existing=True)
    scheduler.start()
    logger.info('Scheduler started, interval = %d seconds', config.SCRAPE_INTERVAL)

    try:
        app.run(host='0.0.0.0', port=config.PORT, debug=False)
    finally:
        scheduler.shutdown()
