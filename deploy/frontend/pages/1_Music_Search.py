import datetime
import sys
from io import BytesIO

sys.path.append("../")

import streamlit as st
from config import settings
from utils.audio import show_audio_snippets
from utils.music_search import get_song, search_audio

st.title("🎶 Find similar song")
uploaded_file = st.file_uploader("Upload a Wav file", type=("wav", "mp3"))

st.session_state.audio = uploaded_file
st.session_state.audio_play_time = 0
st.audio(st.session_state.audio, start_time=st.session_state.audio_play_time)

query_ready = st.button(label="Query", key="query_button")

if uploaded_file and query_ready:
    search_responses = search_audio(url=settings.backend_url, file=uploaded_file)
    st.write(f"Found {len(search_responses)} similar snippets")

    col1, col2 = st.columns(2)
    for response in search_responses:
        with col1:
            start_time = response["start"]
            end_time = response["start"] + response["duration"]
            start_time_format = str(datetime.timedelta(seconds=round(start_time)))
            end_time_format = str(datetime.timedelta(seconds=round(end_time)))
            st.write(f"Your song ({start_time_format} - {end_time_format})")
            show_audio_snippets(
                file=BytesIO(uploaded_file.getvalue()),
                start_time=start_time,
                end_time=end_time,
            )

        with col2:
            start_time = response["enrolled_start"]
            end_time = response["enrolled_start"] + response["duration"]
            start_time_format = str(datetime.timedelta(seconds=round(start_time)))
            end_time_format = str(datetime.timedelta(seconds=round(end_time)))
            st.write(
                f'Detected song {response["file_id"]} ({start_time_format} - {end_time_format})'
            )
            similar_audio_content = get_song(settings.backend_url, response["file_id"])
            show_audio_snippets(
                file=BytesIO(similar_audio_content),
                start_time=start_time,
                end_time=end_time,
            )
