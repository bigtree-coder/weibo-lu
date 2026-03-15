import time
import random
import logging

import requests

import config
import database

logger = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': ('Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) '
                   'AppleWebKit/605.1.15 (KHTML, like Gecko) '
                   'Mobile/15E148 Safari/604.1'),
    'X-Requested-With': 'XMLHttpRequest',
    'Referer': 'https://m.weibo.cn/',
    'Accept': 'application/json, text/plain, */*',
}

# Group feed endpoint (not /feed/friends)
FEED_URL = 'https://m.weibo.cn/feed/group'


def fetch_feed(cookie, group_id, page=1):
    """Fetch one page of the friend group feed. Returns list of statuses."""
    headers = {**HEADERS, 'Cookie': cookie}
    params = {'gid': group_id, 'page': page}

    resp = requests.get(FEED_URL, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get('ok') != 1:
        logger.warning('API returned not ok: %s', data.get('msg', 'unknown'))
        return []

    # Response uses cards array; card_type 9 contains mblog (post)
    statuses = []
    cards = data.get('data', {}).get('cards', [])
    for card in cards:
        if card.get('card_type') == 9:
            mblog = card.get('mblog')
            if mblog:
                statuses.append(mblog)

    # Fallback: some endpoints return data.statuses directly
    if not statuses:
        statuses = data.get('data', {}).get('statuses', [])

    return statuses


def parse_status(status):
    """Extract relevant fields from a weibo status dict."""
    # Collect images
    images = []
    pics = status.get('pics', [])
    for pic in pics:
        large = pic.get('large', {})
        url = large.get('url') or pic.get('url', '')
        if url:
            images.append(url)

    # Build text: include retweet content if present
    text = status.get('text', '')
    retweet = status.get('retweeted_status')
    if retweet:
        rt_user = retweet.get('user', {}).get('screen_name', '?')
        rt_text = retweet.get('text', '')
        text = f'{text}<br/><div class="retweet">转发 @{rt_user}: {rt_text}</div>'
        # Also grab retweet images if the original post has none
        if not images:
            for pic in retweet.get('pics', []):
                large = pic.get('large', {})
                url = large.get('url') or pic.get('url', '')
                if url:
                    images.append(url)

    user = status.get('user', {})
    return {
        'weibo_id': str(status.get('id', '')),
        'user_name': user.get('screen_name', ''),
        'user_avatar': user.get('profile_image_url', ''),
        'text': text,
        'images': images,
        'created_at': status.get('created_at', ''),
    }


def scrape_latest():
    """Scrape latest posts from the configured friend group. Returns count of new posts."""
    cookie = config.WEIBO_COOKIE
    group_id = config.WEIBO_GROUP_ID

    if not cookie or not group_id:
        logger.error('WEIBO_COOKIE or WEIBO_GROUP_ID not configured')
        return 0

    new_count = 0

    # Fetch up to 3 pages
    for page_num in range(1, 4):
        try:
            statuses = fetch_feed(cookie, group_id, page=page_num)
        except Exception:
            logger.exception('Failed to fetch feed (page %d)', page_num)
            break

        if not statuses:
            break

        has_new = False
        for status in statuses:
            post_data = parse_status(status)
            if post_data['weibo_id']:
                inserted = database.insert_post(post_data)
                if inserted:
                    new_count += 1
                    has_new = True

        # Stop paging if no new posts found (all duplicates)
        if not has_new:
            break

        time.sleep(random.uniform(2, 5))

    logger.info('Scrape done, %d new posts saved', new_count)
    return new_count
