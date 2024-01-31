import random
import time
import requests
import openai
import streamlit as st
import pandas as pd
from tqdm.notebook import tqdm
from utils.utils import search_docs
import re
from streamlit_js_eval import streamlit_js_eval

"""
An embeddings file that is loaded in this script had been created in code similar to:

df = pd.read_excel("")
info_column = ""

tokenizer = tiktoken.get_encoding("cl100k_base")
df['n_tokens'] = df[info_column].apply(lambda x: len(tokenizer.encode(x)))

ada_embeddings = []
for _, row in df.iterrows():
    if row["n_tokens"] < 7800:
        response = openai.Embedding.create(
            input=row[info_column],
            engine="text_embedding_ada_002_API_testing"
        )
        ada_embeddings.append(response['data'][0]['embedding'])
    else:
        print(row["Content ID"])
        ada_embeddings.append([0] * 1536)

df["Embeddings"] = ada_embeddings

"""

def refresh():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")

def empty_input():
    st.session_state["prompt"] = st.session_state["prompt_area"]
    print(st.session_state["prompt"], st.session_state["prompt_area"])
    st.session_state["prompt_area"] = ""


token = ""
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = ""
openai.api_type = ""
openai.api_base = ""
openai.api_version = "2023-05-15"

rfp_df = pd.read_pickle("")
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

st.set_page_config(layout="wide")
st.title("RFP GenAI MVP")
# st.selectbox("Select RFP Database version", ("Version 2", "Version 1"))

hide_submit_text = """
<style>
.e1y5xkzn1 {
visibility: hidden;
}
</style>
"""
st.markdown(hide_submit_text, unsafe_allow_html=True)

prompt_engineering_texts = ["If you do not know, please say I don't know. Do not guess.",
                            "Please include the relevant context IDs in your answer. Do not guess context IDs.",
                           ]

prompt_engineering_text = f"[Note: {' '.join(prompt_engineering_texts)}]"

with st.sidebar:
    st.header("Options")
    hp_temperature = st.slider("OpenAI Temperature", 0.0, 2.0, 0.0)
    st.markdown("<sup>**Note**: Too high of a temperature may cause the model/endpoint to hang</sup>", unsafe_allow_html=True)

    st.header("Search RFP database")
    container = st.empty()
    if query := st.text_input("Enter search query"):
        container = st.empty()

        search_words = re.findall(r'\w+', query)
        filtered_df = rfp_df[rfp_df.apply(lambda row: all(word.lower() in str(row).lower() for word in search_words), axis=1)]
        # filtered_df = rfp_df[rfp_df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)]
        if not filtered_df.empty:
            for index, row in filtered_df.iterrows():
                expander_text = f"Content ID: {row['Content ID']}"
                if row['Comments'] != '':
                    expander_text += f" ({row['Comments']})"

                emb_text = row["Embedding Text"]
                with st.expander(expander_text):
                    for par in emb_text.split("\n"):
                        st.markdown(par)
        else:
            container.write("No results found.")
    else:
        for index, row in rfp_df.iterrows():
            expander_text = f"Content ID: {row['Content ID']}"
            if row['Comments'] != '':
                expander_text += f" ({row['Comments']})"

            emb_text = row["Embedding Text"]
            with st.expander(expander_text):
                for par in emb_text.split("\n"):
                    st.markdown(par)

col1, col2 = st.columns(2)

if 'further_prompts' not in st.session_state:
    messages = []
    st.session_state['further_prompts'] = messages

further_prompts = []
context_messages = []

if 'ran' not in st.session_state:
    st.session_state["ran"] = False

if 'prompt' not in st.session_state:
    st.session_state["prompt"] = ''

if 'content_ids' not in st.session_state:
    st.session_state["content_ids"] = []

if 'incs' not in st.session_state:
    st.session_state["incs"] = []

if 'prompt_similarity_res' not in st.session_state:
    st.session_state["prompt_similarity_res"] = pd.DataFrame()

with col1:
    st.subheader("Prompt")
    initial_area = st.empty()
    st.text_area("Enter the initial RFP prompt", key="prompt_area")
    submit_col, reset_col = st.columns(2)
    with submit_col:
        submit_button = st.button("Submit", use_container_width=True, on_click=empty_input)

    with reset_col:
        reset_button = st.button("Reset", use_container_width=True, on_click=refresh)

    if submit_button:
        if not st.session_state["ran"]:
            prompt_similarity_res = search_docs(rfp_df, st.session_state["prompt"], top_n=28)

            toks = 0
            texts = []
            incs = []
            for _, row in prompt_similarity_res.iterrows():
                if toks + row["n_tokens"] > 6000:
                    continue
                else:
                    toks += row["n_tokens"]

                texts.append(row["Embedding Text"])
                incs.append(row["Content ID"])

            texts.append("""You are given information on how... [redacted]""")



            context_messages += [{"role": "system", "content": text} for text in texts]

            # context_messages.append(
            #     {"role": "user", "content": f"Give me an appropriate response to this question: {prompt}"})

            st.session_state["further_prompts"].append({"role": "user", "content": f"Give me an appropriate response to this question: {st.session_state['prompt']}" + prompt_engineering_text})
            st.session_state["ran"] = True
            st.session_state["incs"] += incs
            st.session_state["prompt_similarity_res"] = prompt_similarity_res

        else:
            st.session_state["further_prompts"].append({"role": "user", "content": f"{st.session_state['prompt']}" + prompt_engineering_text})

        full_response = ""
        for response in openai.ChatCompletion.create(
                engine="35_turbo",
                model=st.session_state["openai_model"],
                temperature=hp_temperature,
                messages=context_messages + st.session_state["further_prompts"],
                stream=True
        ):
            resp = response.choices[0].delta.get("content", "")
            full_response += resp
        st.session_state["further_prompts"].append({"role": "assistant", "content": f"{full_response}"})

    with initial_area.container():
        for mes in st.session_state["further_prompts"]:
            if mes["role"] == "user":
                st.info(mes["content"].replace(prompt_engineering_text, ''))
            else:
                st.success(mes["content"])

with col2:
    try:
        st.subheader("Sources")
        gpt_sources_tab, vector_sim_tab = st.tabs(["GPT Response Sources", "Provided Context Sources"])
        with gpt_sources_tab:
            try:

                content_id = re.findall(r'Content ID: (\d+)', full_response)

                content_ids = re.findall(r'Content IDs: (\d+(?:, \d+)*)', full_response)
                if len(content_ids) > 0:
                    content_ids = content_ids[0].split(", ")

                all_content_ids = content_id + content_ids
                print(all_content_ids)
                st.session_state["content_ids"] = list(set(st.session_state["content_ids"] + all_content_ids))
                st.session_state["content_ids"].sort()
                print(st.session_state["content_ids"])
            except NameError:
                pass

            not_found = []
            for id in st.session_state["content_ids"]:
                try:
                    row = rfp_df[rfp_df["Content ID"] == str(id)].iloc[0]
                except IndexError:
                    not_found.append(id)
                    continue

                expander_text = f"Content ID: {row['Content ID']}"
                if row['Comments'] != '':
                    expander_text += f" ({row['Comments']})"

                emb_text = row["Embedding Text"]
                with st.expander(expander_text):
                    for par in emb_text.split("\n"):
                        st.markdown(par)

            if len(not_found) > 0:
                st.warning(f"**Not found**: {', '.join(not_found)}")

        with vector_sim_tab:
            inc_content = st.session_state["prompt_similarity_res"][st.session_state["prompt_similarity_res"]["Content ID"].isin(st.session_state["incs"])]

            for _, row in inc_content.iterrows():
                expander_text = f"[Similarity: {round(row['similarities'] * 100, 2)}]\nContent ID: {row['Content ID']}"
                if row['Comments'] != '':
                    expander_text += f" ({row['Comments']})"

                emb_text = row["Embedding Text"]
                with st.expander(expander_text):
                    for par in emb_text.split("\n"):
                        st.markdown(par)
    except (NameError, KeyError):
        pass