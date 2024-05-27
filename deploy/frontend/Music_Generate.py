import streamlit as st
from config import settings
from utils.music_gen import get_suno_songs, init_suno_connection, run_music_generate

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
st.caption("🚀 An AI music making site powered by Suno.com")
make_instrumental = st.toggle(label="Instrument only")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "is_notification": True,
            "content": "Provide me a description of the song  \n (e.g. a rap song about chickens)",
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
                col1, col2 = st.columns(2)
                with col1:
                    st.write(
                        "***Lyrics***:  \n",
                        msg["content"][0]["metadata"]["prompt"].replace("\n", "  \n"),
                    )
                with col2:
                    for i, gen_song in enumerate(msg["content"]):
                        if gen_song["status"] == "streaming":
                            updated_responses = get_suno_songs(
                                url=settings.backend_url, ids=[gen_song["id"]]
                            )
                            gen_song = updated_responses.json()[0]
                        st.write(f'**Song** ***#{i+1}***: {gen_song["title"]}')
                        st.audio(gen_song["audio_url"])
                        st.write(f'Tags: *{gen_song["metadata"]["tags"]}*')

if prompt := st.chat_input():
    if not suno_session_id or not suno_cookie:
        st.info(
            "Please add your Suno Session ID and Cookie at the side bar to continue."
        )
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Generating music..."):
        responses = run_music_generate(
            url=settings.backend_url, prompt=prompt, make_instrumental=make_instrumental
        )
    responses_json = responses.json()
    if responses.status_code != 200:
        st.write("Out of tokens! Please visit suno.com or using another account")
        st.write(responses_json)
        st.stop()
    with st.chat_message("assistant"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(
                "***Lyrics***:  \n",
                responses_json[0]["metadata"]["prompt"].replace("\n", "  \n"),
            )
        with col2:
            for i, gen_song in enumerate(responses_json):
                st.write(f'**Song** ***#{i+1}***: {gen_song["title"]}')
                st.audio(gen_song["audio_url"])
                st.write(f'Tags: *{gen_song["metadata"]["tags"]}*')

    st.session_state.messages.append(
        {"role": "assistant", "is_notification": False, "content": responses_json}
    )
