import streamlit as st
import anthropic
import os
from llmware.library import Library
from llmware.configs import LLMWareConfig
import datetime
import shutil
import zipfile

with st.sidebar:
    #anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")
    #"[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("📝 Upload Files")
uploaded_files = st.file_uploader("Upload documents", type=("pdf","txt"), accept_multiple_files=True)
#question = st.text_input(
#    "Ask something about the article",
#    placeholder="Can you give me a short summary?",
#    disabled=not uploaded_file,
#)

if uploaded_files:
    
    #file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
    #st.write(file_details)
    home_dir = os.path.expanduser('~')
    foldername = home_dir + "/pdf_files/"
    to_day = datetime.date.today().strftime("%Y-%m-%d")
    foldername+=to_day
    for uploaded_file in uploaded_files:
    #if uploaded_file.type=="zip":
    #if uploaded_file.name.endswith('.zip'):
    #   if not os.path.exists(foldername):
    #      os.mkdir(foldername)
    #   with zipfile.ZipFile(uploaded_file.name, 'r') as zip_ref:
    #      zip_ref.extractall(foldername)
    #   st.success(f"Extracted files from {uploaded_file.name}") 
    #else:
       if not os.path.exists(foldername):
          os.mkdir(foldername)
       with open(os.path.join(foldername,uploaded_file.name),"wb") as f: 
          f.write(uploaded_file.getbuffer())
    st.success("Saved File")

#if uploaded_file and question and not anthropic_api_key:
#    st.info("Please add your Anthropic API key to continue.")

#if uploaded_file and question and anthropic_api_key:
#    article = uploaded_file.read().decode()
#    prompt = f"""{anthropic.HUMAN_PROMPT} Here's an article:\n\n<article>
#    {article}\n\n</article>\n\n{question}{anthropic.AI_PROMPT}"""

#    client = anthropic.Client(api_key=anthropic_api_key)
#    response = client.completions.create(
#        prompt=prompt,
#        stop_sequences=[anthropic.HUMAN_PROMPT],
#        model="claude-v1",  # "claude-2" for Claude 2 model
#        max_tokens_to_sample=100,
#    )
#    st.write("### Answer")
#    st.write(response.completion)

def create_embeddings():
  # Your function logic here
    LLMWareConfig().set_active_db("sqlite")
    vector_db = "faiss"
    library_name = "my_library2"
    embedding_model_name = "mini-lm-sbert"
    library = Library().create_new_library(library_name)
    home_dir = os.path.expanduser('~')
    sample_files_path = home_dir + '/pdf_files/'
    #foldername = "/home/ankit/pdf_files/"
    to_day = datetime.date.today().strftime("%Y-%m-%d")
    #foldername+=to_day
    contracts_path = os.path.join(sample_files_path, to_day)
    library.add_files(input_folder_path=contracts_path)
    embeddings_stat = library.install_new_embedding(embedding_model_name=embedding_model_name, vector_db=vector_db)
    if embeddings_stat['embeddings_created'] > 0:
    #    embed_created = True
        foldername = home_dir + "/pdf_files/"
        to_day = datetime.date.today().strftime("%Y-%m-%d")
        foldername+=to_day
        source_path = foldername
        destination_path = home_dir + "/pdf_files/processed"
        files = os.listdir(source_path)
        for f in files:
           full_filename = source_path+"/"+f
           shutil.move(full_filename, destination_path)
    #    shutil.copytree(source_path, destination_path)
    return library.get_embedding_status()

clicked = st.button("Create embeddings in VectorDB")

if clicked:
  embedding_status = create_embeddings()
  st.write("Embeddings Created!")
