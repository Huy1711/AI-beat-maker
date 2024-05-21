import logging
import time
from threading import Thread

import aiohttp
import requests

from .cookie import SunoCookie

logger = logging.getLogger("beat-maker-api")
logging.basicConfig(level=logging.INFO)


### SUNO URLS
BASE_URL = "https://studio-api.suno.ai"
CLERK_BASE_URL = "https://clerk.suno.com"
GENERATE_MUSIC_URL = f"{BASE_URL}/api/generate/v2/"


COMMON_HEADERS = {
    "Content-Type": "text/plain;charset=UTF-8",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://suno.com",
    "Origin": "https://suno.com",
}


class SunoClient:
    def __init__(self, cookie: str, session_id: str) -> None:
        self.suno_cookie = SunoCookie()
        self.suno_cookie.set_session_id(session_id)
        self.suno_cookie.load_cookie(cookie)
        self.session_id = session_id
        self.renew_token_url = f"{CLERK_BASE_URL}/v1/client/sessions/{session_id}/tokens?_clerk_js_version=4.72.0-snapshot.vc141245"
        self.start_keep_alive()

    def _keep_session_alive(self) -> None:
        """Renew the authentication token periodically to keep the session alive."""
        while True:
            headers = {"cookie": self.suno_cookie.get_cookie()}
            headers.update(COMMON_HEADERS)

            response = requests.post(url=self.renew_token_url, headers=headers)

            resp_headers = dict(response.headers)
            set_cookie = resp_headers.get("Set-Cookie")
            self.suno_cookie.load_cookie(set_cookie)

            new_token = response.json().get("jwt")
            self.suno_cookie.set_token(new_token)
            time.sleep(5)

    def start_keep_alive(self):
        t = Thread(target=self._keep_session_alive)
        t.start()

    async def generate(self, data):
        headers = {"Authorization": f"Bearer {self.suno_cookie.get_token()}"}
        headers.update(COMMON_HEADERS)
        logger.info("Generating Song...")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=GENERATE_MUSIC_URL, json=data, headers=headers
            ) as resp:
                response = await resp.json()
        logger.info("Generated Song Successfully")
        return response

    async def get_feed(self, ids):
        headers = {"Authorization": f"Bearer {self.suno_cookie.get_token()}"}
        api_url = f"{BASE_URL}/api/feed/?ids={ids}"
        logger.info("Getting Song...")
        async with aiohttp.ClientSession() as session:
            async with session.get(url=api_url, headers=headers) as resp:
                response = await resp.json()
        logger.info("Get Song Successfully")
        return response

    # async def generate_custom(self, prompt, title, tags, make_instrumental=False):
    #     self.client.headers["Authorization"] = f"Bearer {self.token}"
    #     logger.info("Generating Song...")
    #     data = {
    #         "make_instrumental": make_instrumental,
    #         "mv": "chirp-v3-0",
    #         "title": title,
    #         "tag": tags,
    #         "prompt": prompt,
    #     }
    #     response = self.client.post(GENERATE_MUSIC_URL, json=data)
    #     response_json = response.json()
    #     logger.info("Generated Song Successfully")
    #     return response_json
