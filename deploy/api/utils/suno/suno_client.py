import logging
import time
import typing as tp
from threading import Thread

import aiohttp
import requests

from .cookie import SunoCookie

logger = logging.getLogger()
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
    """
    This class provide some helper functions to interact
    with Suno API (suno.com) for the music generation feature
    """

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
            time.sleep(10)

    def start_keep_alive(self):
        logger.info("Created SunoClient! Start keeping alive session")
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

        if response.get("metadata"):
            error_message = response["metadata"]["error_message"]
        else:
            raise Exception(f"Error: {response}")
        if error_message is not None:
            raise Exception(f"Error in prompt: {error_message}")

        logger.info("Generated Song Successfully")
        return response

    async def get_song_by_ids(self, ids: tp.List[str]) -> tp.List[str]:
        """Get songs that have been saved to Suno website using song ids"""
        headers = {"Authorization": f"Bearer {self.suno_cookie.get_token()}"}
        api_url = f'{BASE_URL}/api/feed/?ids={",".join(ids)}'

        async with aiohttp.ClientSession() as session:
            async with session.get(url=api_url, headers=headers) as resp:
                response = await resp.json()
        return response

    async def _wait_gen_song_complete(self, song_ids, timeout_secs=100):
        """Wait for song generating process to complete."""
        start_time = time.time()
        song_results = []
        logger.info("Getting Song...")
        while (time.time() - start_time) < timeout_secs:
            song_results = await self.get_song_by_ids(song_ids)
            is_comlete = all(
                song["status"] in ["complete", "streaming"] for song in song_results
            )
            if is_comlete:
                logger.info("Get Song Successfully")
                return song_results
            time.sleep(3)
        return song_results

    async def generate_and_get_song(self, data, is_custom=False):
        if is_custom:
            gen_response = await self.generate_custom(data)
        else:
            gen_response = await self.generate(data)

        song_ids = []
        for song_metadata in gen_response["clips"]:
            song_ids.append(song_metadata["id"])

        responses = await self._wait_gen_song_complete(song_ids)
        if len(responses) == 0:
            raise Exception("Please try again!")

        return responses
