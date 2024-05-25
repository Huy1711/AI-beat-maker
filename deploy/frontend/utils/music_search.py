import requests

SEARCH_ENDPOINT = "/search"  # POST
GET_SONG_ENDPOINT = "/song"  # GET


def init_suno_connection(session_id: str, cookie: str):
    response = requests.post(
        settings.TRANSCRIBE_ENDPOINT, files={"audio_file": st.session_state.audio}
    )
