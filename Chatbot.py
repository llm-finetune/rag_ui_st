from openai import OpenAI
import streamlit as st
import os
from llmware.library import Library
from llmware.retrieval import Query
from llmware.setup import Setup
from llmware.status import Status
from llmware.prompts import Prompt
from llmware.configs import LLMWareConfig

def semantic_rag (library_name, embedding_model_name, llm_model_name, prompt):
    # Step 1 - Create library which is the main 'organizing construct' in llmware
    finalResponse = " "
    #print ("\nupdate: Step 1 - Creating library: {}".format(library_name))

    #library = Library().create_new_library(library_name)
    library = Library().load_library(library_name)
    home_dir = os.path.expanduser('~')
    sample_files_path = home_dir + '/pdf_files'

    contracts_path = os.path.join(sample_files_path, "processed")

    #print("update: Step 3 - Parsing and Text Indexing Files")
    #library.add_files(input_folder_path=contracts_path)

    #print("\nupdate: Step 4 - Generating Embeddings in {} db - with Model- {}".format(vector_db, embedding_model))
    #library.install_new_embedding(embedding_model_name=embedding_model_name, vector_db=vector_db)

    print("\nupdate: Loading model for LLM inference - ", llm_model_name)
    prompter = Prompt().load_model(llm_model_name)

    #query = "what is the executive's base annual salary"
    query = prompt

    results = Query(library).semantic_query(query, result_count=50, embedding_distance_threshold=1.0)

    for i, contract in enumerate(os.listdir(contracts_path)):

        qr = []

        if contract != ".DS_Store":

            print("\nContract Name: ", i, contract)

            #   we will look through the list of semantic query results, and pull the top results for each file
            for j, entries in enumerate(results):

                library_fn = entries["file_source"]
                if os.sep in library_fn:
                    # handles difference in windows file formats vs. mac / linux
                    library_fn = library_fn.split(os.sep)[-1]

                if library_fn == contract:
                    print("Top Retrieval: ", j, entries["distance"], entries["text"])
                    qr.append(entries)

            #   we will add the query results to the prompt
            source = prompter.add_source_query_results(query_results=qr)

            #   run the prompt
            response = prompter.prompt_with_source(query, prompt_name="default_with_context", temperature=0.7)

            #   note: prompt_with_resource returns a list of dictionary responses
            #   -- depending upon the size of the source context, it may call the llm several times
            #   -- each dict entry represents 1 call to the LLM

            for resp in response:
                if "llm_response" in resp:
                    print("\nupdate: llm answer - ", resp["llm_response"])
                    resp1 = (resp["llm_response"]) + '\n' + contract + '\n'
            finalResponse += resp1

            # start fresh for next document
            prompter.clear_source_materials()
    return finalResponse

with st.sidebar:
    #openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    #"[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    #"[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    #if not openai_api_key:
    #    st.info("Please add your OpenAI API key to continue.")
    #    st.stop()

## Start Adding custom code here AK
    LLMWareConfig().set_active_db("sqlite")
    #LLMWareConfig().set_vector_db("faiss")
    vector_db = "faiss"
    lib_name = "my_library2"
    #LLMWareConfig().set_config("debug_mode", 2)
    #library = setup_library("example5_library")
    embedding_model = "mini-lm-sbert"
    llm_model_name = "llmware/dragon-yi-6b-gguf"
    library_name=lib_name
    embedding_model_name=embedding_model
    output = semantic_rag(library_name, embedding_model_name, llm_model_name, prompt)
    #sample_folders = ["Agreements", "Invoices", "UN-Resolutions-500", "SmallLibrary", "FinDocs", "AgreementsLarge"]
    #library_name = "example1_library"
    #selected_folder = sample_folders[0]
    #output = parsing_documents_into_library(library_name, selected_folder, prompt)

## End Adding custom code here AK

    #client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    print("Prompt > ")
    print(prompt)
    
    st.chat_message("user").write(prompt)
    st.chat_message("assistant").write(output)
    #response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    #msg = response.choices[0].message.content
    #st.session_state.messages.append({"role": "assistant", "content": msg})
    #st.chat_message("assistant").write(msg)
