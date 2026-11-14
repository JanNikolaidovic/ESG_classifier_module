"""Polite downloading: one request at a time, with pauses, retries and robots.txt checks."""

import logging
import time
import urllib.robotparser
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

USER_AGENT = "esg-screening-research/0.1 (python-requests; builds a research dataset from public IFC disclosures)"
RETRY_STATUS = {429, 500, 502, 503, 504}  # "too many requests" and temporary server errors


class PoliteFetcher:
    """Downloads web pages slowly and carefully, so we never burden the website.

    - waits at least `delay` seconds between two requests
    - checks each website's robots.txt and refuses pages it disallows
    - on temporary errors, waits longer and longer before retrying (5 s, 10 s, 20 s, ...)
    """

    def __init__(self, delay=1.0, max_retries=4, timeout=60):
        self.delay = delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last_request = 0.0
        self._robots = {}

    def get(self, url, params=None):
        """Return the text of `url`; raise an error if it still fails after all retries."""
        if not self._allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        for attempt in range(self.max_retries + 1):
            self._wait()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code not in RETRY_STATUS:
                    response.raise_for_status()
                    return response.text
                wait = float(response.headers.get("Retry-After", 5 * 2 ** attempt))
                reason = f"HTTP {response.status_code}"
            except (requests.ConnectionError, requests.Timeout) as error:
                wait, reason = 5 * 2 ** attempt, type(error).__name__
            if attempt < self.max_retries:
                log.warning("%s for %s, retrying in %.0f s", reason, url, wait)
                time.sleep(wait)
        raise RuntimeError(f"giving up on {url} after {self.max_retries + 1} attempts ({reason})")

    def _wait(self):
        pause = self._last_request + self.delay - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        self._last_request = time.monotonic()

    def _allowed(self, url):
        host = urlparse(url)
        if host.netloc not in self._robots:
            robots = urllib.robotparser.RobotFileParser()
            try:
                self._wait()
                text = self.session.get(f"{host.scheme}://{host.netloc}/robots.txt", timeout=self.timeout)
                robots.parse(text.text.splitlines() if text.ok else [])
            except requests.RequestException:
                robots.parse([])  # robots.txt unreachable: no rules to follow
            self._robots[host.netloc] = robots
        return self._robots[host.netloc].can_fetch(USER_AGENT, url)
