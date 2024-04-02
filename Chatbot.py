from openai import OpenAI
import streamlit as st
import os
from llmware.library import Library
from llmware.retrieval import Query
from llmware.setup import Setup
from llmware.status import Status
from llmware.prompts import Prompt
from llmware.configs import LLMWareConfig

def parsing_documents_into_library(library_name, sample_folder, prompt):

    print(f"\nExample - Parsing Files into Library")

    #   create new library
    print (f"\nStep 1 - creating library {library_name}")
    library = Library().create_new_library(library_name)

    #   load the llmware sample files
    #   -- note: if you have used this example previously, UN-Resolutions-500 is new path
    #   -- to pull updated sample files, set: 'over_write=True'

    ##sample_files_path = Setup().load_sample_files(over_write=False)
    ##print (f"Step 2 - loading the llmware sample files and saving at: {sample_files_path}")
    sample_files_path = '/home/ankit/llmware_data/sample_files'
    #   note: to replace with your own documents, just point to a local folder path that has the documents
    ingestion_folder_path = os.path.join(sample_files_path, sample_folder)

    print (f"Step 3 - parsing and indexing files from {ingestion_folder_path}")

    #   add files is the key ingestion method - parses, text chunks and indexes all files in folder
    #       --will automatically route to correct parser based on file extension
    #       --supported file extensions:  .pdf, .pptx, .docx, .xlsx, .csv, .md, .txt, .json, .wav, and .zip, .jpg, .png

    parsing_output = library.add_files(ingestion_folder_path)

    print (f"Step 4 - completed parsing - {parsing_output}")

    #   check the updated library card
    updated_library_card = library.get_library_card()
    doc_count = updated_library_card["documents"]
    block_count = updated_library_card["blocks"]
    print(f"Step 5 - updated library card - documents - {doc_count} - blocks - {block_count} - {updated_library_card}")

    #   check the main folder structure created for the library - check /images to find extracted images
    library_path = library.library_main_path
    print(f"Step 6 - library artifacts - including extracted images - saved at folder path - {library_path}")

    #   use .add_files as many times as needed to build up your library, and/or create different libraries for
    #   different knowledge bases

    #   now, your library is ready to go and you can start to use the library for running queries
    #   if you are using the "Agreements" library, then a good easy 'hello world' query is "base salary"
    #   if you are using one of the other sample folders (or your own), then you should adjust the query

    #   queries are always created the same way, e.g., instantiate a Query object, and pass a library object
    #   --within the Query class, there are a variety of useful methods to run different types of queries

    ##test_query = "base salary"
    test_query = prompt

    print(f"\nStep 7 - running a test query - {test_query}\n")

    query_results = Query(library).text_query(test_query, result_count=10)

    for i, result in enumerate(query_results):

        #   note: each result is a dictionary with a wide range of useful keys
        #   -- we would encourage you to take some time to review each of the keys and the type of metadata available

        #   here are a few useful attributes
        text = result["text"]
        file_source = result["file_source"]
        page_number = result["page_num"]
        doc_id = result["doc_ID"]
        block_id = result["block_ID"]
        matches = result["matches"]

        #   -- if print to console is too verbose, then pick just a few keys for print
        print("query results: ", i, result)

    return query_results[0]['text']

def setup_library(library_name):

    """ Note: this setup_library method is provided to enable a self-contained example to create a test library """

    #   Step 1 - Create library which is the main 'organizing construct' in llmware
    print ("\nupdate: Creating library: {}".format(library_name))

    library = Library().create_new_library(library_name)

    #   check the embedding status 'before' installing the embedding
    embedding_record = library.get_embedding_status()
    print("embedding record - before embedding ", embedding_record)

    #   Step 2 - Pull down the sample files from S3 through the .load_sample_files() command
    #   --note: if you need to refresh the sample files, set 'over_write=True'
    #print ("update: Downloading Sample Files")

    #sample_files_path = Setup().load_sample_files(over_write=False)
    sample_files_path = '/home/ankit/llmware_data/sample_files'
    #   Step 3 - point ".add_files" method to the folder of documents that was just created
    #   this method parses the documents, text chunks, and captures in database

    print("update: Parsing and Text Indexing Files")

    library.add_files(input_folder_path=os.path.join(sample_files_path, "Agreements"))

    return library


def install_vector_embeddings(library, embedding_model_name):

    """ This method is the core example of installing an embedding on a library.
        -- two inputs - (1) a pre-created library object and (2) the name of an embedding model """

    library_name = library.library_name
    vector_db = LLMWareConfig().get_vector_db()

    print(f"\nupdate: Starting the Embedding: "
          f"library - {library_name} - "
          f"vector_db - {vector_db} - "
          f"model - {embedding_model_name}")

    #   *** this is the one key line of code to create the embedding ***
    library.install_new_embedding(embedding_model_name=embedding_model_name, vector_db=vector_db,batch_size=100)

    #   note: for using llmware as part of a larger application, you can check the real-time status by polling Status()
    #   --both the EmbeddingHandler and Parsers write to Status() at intervals while processing
    update = Status().get_embedding_status(library_name, embedding_model_name)
    print("update: Embeddings Complete - Status() check at end of embedding - ", update)

    # Start using the new vector embeddings with Query
    #sample_query = "incentive compensation"
    sample_query = prompt
    print("\n\nupdate: Run a sample semantic/vector query: {}".format(sample_query))

    #   queries are constructed by creating a Query object, and passing a library as input
    query_results = Query(library).semantic_query(sample_query, result_count=20)

    for i, entries in enumerate(query_results):

        #   each query result is a dictionary with many useful keys

        text = entries["text"]
        document_source = entries["file_source"]
        page_num = entries["page_num"]
        vector_distance = entries["distance"]

        #   to see all of the dictionary keys returned, uncomment the line below
        #   print("update: query_results - all - ", i, entries)

        #  for display purposes only, we will only show the first 125 characters of the text
        if len(text) > 125:  text = text[0:125] + " ... "

        print("\nupdate: query results - {} - document - {} - page num - {} - distance - {} "
              .format( i, document_source, page_num, vector_distance))

        print("update: text sample - ", text)

    #   lets take a look at the library embedding status again at the end to confirm embeddings were created
    embedding_record = library.get_embedding_status()

    print("\nupdate:  embedding record - ", embedding_record)

    return 0

def semantic_rag (library_name, embedding_model_name, llm_model_name, prompt):
    # Step 1 - Create library which is the main 'organizing construct' in llmware
    finalResponse = " "
    print ("\nupdate: Step 1 - Creating library: {}".format(library_name))

    library = Library().create_new_library(library_name)

    sample_files_path = '/home/ankit/llmware_data/sample_files'

    contracts_path = os.path.join(sample_files_path, "Legal")

    print("update: Step 3 - Parsing and Text Indexing Files")
    library.add_files(input_folder_path=contracts_path)

    print("\nupdate: Step 4 - Generating Embeddings in {} db - with Model- {}".format(vector_db, embedding_model))
    library.install_new_embedding(embedding_model_name=embedding_model_name, vector_db=vector_db)

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
                    finalResponse += (resp["llm_response"])

            # start fresh for next document
            prompter.clear_source_materials()
    return finalResponse

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

## Start Adding custom code here AK
    LLMWareConfig().set_active_db("sqlite")
    #LLMWareConfig().set_vector_db("faiss")
    vector_db = "faiss"
    lib_name = "example5_library"
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

    client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    print("Prompt > ")
    print(prompt)
    
    st.chat_message("user").write(prompt)
    st.chat_message("assistant").write(output)
    #response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    #msg = response.choices[0].message.content
    #st.session_state.messages.append({"role": "assistant", "content": msg})
    #st.chat_message("assistant").write(msg)
