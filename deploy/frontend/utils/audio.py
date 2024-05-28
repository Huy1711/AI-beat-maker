import numpy as np
import soundfile as sf
import streamlit as st

SAMPLE_RATE = 8000


def show_audio_snippets(file, start_time, end_time, sample_rate=SAMPLE_RATE):
    """Cut audio into snippet and show to the interface."""
    duration = end_time - start_time
    query_audio = sf.read(
        file=file,
        start=int(start_time * SAMPLE_RATE),
        frames=int(duration * SAMPLE_RATE),
    )
    st.audio(
        np.expand_dims(query_audio[0], axis=0),
        sample_rate=sample_rate,
        start_time=start_time,
        end_time=end_time,
    )
