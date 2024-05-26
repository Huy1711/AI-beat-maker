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
make_instrumental = st.toggle(label="Instrument only")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "is_notification": True,
            "content": "Please provide your text prompt",
        }
    ]

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        if msg["is_notification"]:
            st.chat_message("assistant").write(msg["content"])
        else:
            with st.chat_message("assistant"):
                for i, gen_song in enumerate(msg["content"]):
                    st.write(f"Song #{i+1}")
                    st.write("Tags:", gen_song["metadata"]["tags"])
                    st.write("Lyrics:", gen_song["metadata"]["prompt"])
                    st.audio(gen_song["audio_url"])

if prompt := st.chat_input():
    if not suno_session_id or not suno_cookie:
        st.info(
            "Please add your Suno Session ID and Cookie at the side bar to continue."
        )
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    responses = run_music_generate(
        url=settings.backend_url, prompt=prompt, make_instrumental=make_instrumental
    )
    with st.chat_message("assistant"):
        for i, gen_song in enumerate(responses):
            st.write(f"Song #{i+1}")
            st.write("Tags:", gen_song["metadata"]["tags"])
            st.write("Lyrics:", gen_song["metadata"]["prompt"])
            st.audio(gen_song["audio_url"])

    st.session_state.messages.append(
        {"role": "assistant", "is_notification": False, "content": responses}
    )
