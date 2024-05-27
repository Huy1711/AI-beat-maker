import requests

SEARCH_ENDPOINT = "/search"  # POST
GET_SONG_ENDPOINT = "/songs"  # GET


def search_audio(url: str, file):
    response = requests.post(
        url + SEARCH_ENDPOINT,
        files={
            "file": file,
        },
    )
    return response.json()


def get_song_in_db(url: str, song_id: str):
    response = requests.get(
        f"{url}{GET_SONG_ENDPOINT}/{song_id}",
    )
    return response.content
