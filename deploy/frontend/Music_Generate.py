import os

import streamlit as st
from config import settings
from utils.music_gen import init_suno_connection, run_music_generate

with st.sidebar:
    suno_session_id = st.text_input(
        "Suno Session ID", key="suno_session_id", type="default"
    )
    suno_cookie = st.text_input("Suno Cookie", key="suno_cookie", type="password")
    st.button(
        label="Submit",
        on_click=init_suno_connection,
        args=(settings.backend_url, suno_session_id, suno_cookie),
        key="suno_submit_button",
    )

st.title("🎵 Music Generator")
st.caption("🚀 A Streamlit music making site powered by Suno")
on = st.toggle(label="Instrument only")
make_instrumental = True if on else False


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Please provide your text prompt"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not suno_session_id or not suno_cookie:
        st.info(
            "Please add your Suno Session ID and Cookie at the side bar to continue."
        )
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = run_music_generate(
        url=settings.backend_url, prompt=prompt, make_instrumental=make_instrumental
    )
    # client = OpenAI(api_key=openai_api_key)
    # response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    # msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
