import requests

SUNO_INIT_ENDPOINT = "/suno-connect"  # POST
SUNO_GEN_ENDPOINT = "/generate"  # POST


def init_suno_connection(url: str, session_id: str, cookie: str):
    response = requests.post(
        url + SUNO_INIT_ENDPOINT,
        json={
            "session_id": session_id,
            "cookie": cookie,
        },
    )
    print(response)
    return response


def run_music_generate(url: str, prompt: str, make_instrumental: bool, mv="chirp-v3-0"):
    response = requests.post(
        url + SUNO_GEN_ENDPOINT,
        json={
            "gpt_description_prompt": prompt,
            "make_instrumental": make_instrumental,
            "mv": mv,
        },
    )
    return response.json()
