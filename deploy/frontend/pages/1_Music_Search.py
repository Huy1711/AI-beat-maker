import sys

sys.path.append("../")

import streamlit as st
from config import settings
from utils.music_search import get_song, search_audio

st.title("🎶 Find similar song")
uploaded_file = st.file_uploader("Upload a Wav file", type=("wav", "mp3"))

st.session_state.audio = uploaded_file
st.session_state.audio_play_time = 0
st.audio(st.session_state.audio, start_time=st.session_state.audio_play_time)

query_ready = st.button(label="Query", key="query_button")

if uploaded_file and query_ready:
    search_responses = search_audio(url=settings.backend_url, file=uploaded_file)
    print("search_responses", search_responses)
    st.write(f"Found {len(search_responses)} similar songs")
    for response in search_responses:
        duplicate_song = get_song(settings.backend_url, response["file_id"])
        st.write(response["file_id"])
        st.audio(
            duplicate_song,
            start_time=response["start"],
            end_time=response["start"] + response["duration"],
        )
