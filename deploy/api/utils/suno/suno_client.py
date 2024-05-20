import logging
import time

import requests

logger = logging.getLogger("beat-maker-api")
logging.basicConfig(level=logging.INFO)

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"

### SUNO URLS
BASE_URL = "https://studio-api.suno.ai"
CLERK_BASE_URL = "https://clerk.suno.com"
GENERATE_MUSIC_URL = f"{BASE_URL}/api/generate/v2/"

COMMON_HEADERS = {
    "Content-Type": "text/plain;charset=UTF-8",
    "User-Agent": USER_AGENT,
    "Referer": "https://suno.com",
    "Origin": "https://suno.com",
}


class SunoClient:
    def __init__(self, cookie: str, session_id: str) -> None:
        self.client = requests.Session()
        self.client.headers.update(COMMON_HEADERS)
        self.client.headers["Cookie"] = cookie
        self.session_id = session_id
        self.token = None
        self._keep_session_alive()

    def _keep_session_alive(self) -> None:
        """Renew the authentication token periodically to keep the session alive."""
        renew_url = f"{CLERK_BASE_URL}/v1/client/sessions/{self.session_id}/tokens?_clerk_js_version=4.72.4"
        response = self.client.post(renew_url)
        new_token = response.json().get("jwt")
        self.token = new_token
        self.client.headers["Authorization"] = f"Bearer {self.token}"
        time.sleep(5)

    async def generate(self, prompt, make_instrumental=False):
        self.client.headers["Authorization"] = f"Bearer {self.token}"
        logger.info("Generating Song...")
        data = {
            "make_instrumental": make_instrumental,
            "mv": "chirp-v3-0",
            "prompt": "",
            "gpt_description_prompt": prompt,
        }
        response = self.client.post(GENERATE_MUSIC_URL, json=data)
        response_json = response.json()
        logger.info("Generated Song Successfully")
        return response_json

    async def generate_custom(self, prompt, title, tags, make_instrumental=False):
        self.client.headers["Authorization"] = f"Bearer {self.token}"
        logger.info("Generating Song...")
        data = {
            "make_instrumental": make_instrumental,
            "mv": "chirp-v3-0",
            "title": title,
            "tag": tags,
            "prompt": prompt,
        }
        response = self.client.post(GENERATE_MUSIC_URL, json=data)
        response_json = response.json()
        logger.info("Generated Song Successfully")
        return response_json
