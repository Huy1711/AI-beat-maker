import streamlit as st

# with st.sidebar:
#     anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")
#     "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
#     "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("📝 Your music is similar with...")
uploaded_file = st.file_uploader("Upload a Wav file", type=("wav"))

if uploaded_file:
    audio = uploaded_file.read().decode()
    # prompt = f"""{anthropic.HUMAN_PROMPT} Here's an article:\n\n<article>
    # {article}\n\n</article>\n\n{question}{anthropic.AI_PROMPT}"""

    # client = anthropic.Client(api_key=anthropic_api_key)
    # response = client.completions.create(
    #     prompt=prompt,
    #     stop_sequences=[anthropic.HUMAN_PROMPT],
    #     model="claude-v1",  # "claude-2" for Claude 2 model
    #     max_tokens_to_sample=100,
    # )
    # st.write("### Answer")
    # st.write(response.completion)
