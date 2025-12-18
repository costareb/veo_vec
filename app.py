import streamlit as st
import pathlib
import urllib.parse
from datetime import date
import time
from weaviate.classes.init import AdditionalConfig, Timeout
import streamlit.components.v1 as components
import http.server
import threading
import functools

from urllib.parse import quote
from pathlib import Path
import os
import weaviate
from PIL import Image
import base64
from pathlib import Path
#from streamlit import caption, feedback
#from veo_database import _connect_client, _close_client, semantic_search, hybrid_search, show_collections
from app_utils import (populate_fun, initialize, under_construction, run_RAG, _connect_client,
                       _close_client, semantic_search, hybrid_search, show_collections, save_rag_feedback)
from fake_OCR import unpack_txts
from sanity import run_sanity_check_fun


# constants
wait = 5
STATIC_BASE = "http://localhost:8000"

def pdf_link(relative_path, label=None):
    # relative_path example: "clientX/2023/doc1.pdf"
    url = f"{STATIC_BASE}/{quote(relative_path)}"
    text = label or relative_path
    return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{text}</a>'


print("state:")
print(st.session_state)
print(st.session_state.items())

icon = Image.open(os.path.join("app_data", "observer.png"))
st.set_page_config(page_title="VeoVec Search", page_icon=icon, layout="wide")

if "main_state" not in st.session_state:
    st.session_state["main_state"] = "welcome"
    # being here means we start the current session
    try: # run a test if the connection is working
        client = _connect_client()
        _close_client(client)
    except weaviate.exceptions.WeaviateConnectionError:
        st.error("Connection to Weaviate Client failed.")
        st.write("Please check if docker desktop ist installed and running.")
        st.write("After installation run docker and refresh browser tab. Install from:")
        st.write("Windows - https://docs.docker.com/desktop/setup/install/windows-install/")
        st.write("MacOS   - https://docs.docker.com/desktop/setup/install/mac-install/")
        path = os.path.join("app_data", "docker.jpg")
        docker_image = Image.open(path)
        docker_image = docker_image.resize((600, 400))
        st.image(docker_image)

# to check:
if "pop_data" not in st.session_state:
    st.session_state["pop_data"] = None

# ---------------- Sidebar with tabs ----------------
with (st.sidebar):
    tabs = st.tabs(["🔎 Search", f"❓RAG", f"🌍 Filter", f"{chr(128218)} Collections", "🛠️ Preprocess PDFs", "💁‍♂️ Help"])  # three “panels” in the single sidebar

    # get the options for the selectors
    options = show_collections(client=None)
    with tabs[0]:
        st.header("Search Settings")
        collection_name = st.selectbox(label="Choose a collection", options=options, key="search_collection")
        mode = st.radio("Mode", ["Hybrid", "Semantic"], key="search_mode")                          
        limit = st.slider("Max results", 1, 50, 5, key="search_limit")                                  
        # enable/disable alpha based on selected mode (avoid session_state lookup issues)   
        alpha_disabled = (mode != "Hybrid")                                                         
        alpha = st.slider("Hybrid α (0=keyword, 1=vector)", 0.0, 1.0, 0.5, 0.05,
                          key="search_alpha", disabled=alpha_disabled)

        open_search = st.button("Open Search 🔎")

    with tabs[1]:
        st.header("Experimental - Retrieval Augmented Generation")
        st.caption("RAG is based on 2 models and one collection (i.e. the chunks you created when populating and their corresponding vector representations). The models are:")
        st.caption("1. nomic-embed-text for embedding the query prompt and searching the information.")
        st.caption("2. mistral-nemo for answer generation - https://mistral.ai/news/mistral-nemo ")
        rag_collection = st.selectbox("**Choose collection to use for information retrieval (IR):**", options=options, key = "rag_collection")
        rag_alpha = st.slider("Hybrid search parameter α for IR (0=keyword, 1=vector)", 0.0, 1.0, 0.5, 0.05,
                          key="rag_alpha")
        rag_limit = st.slider("Max number of text chunks to consider for IR", 5, 100, 5, key="rag_limit")
        rag_number_context_shown = st.slider("Display max this number of files that were relevant for IR", 0, 50, 5, key="rag_number_context_shown")
        rag_timeout = st.slider("Max time to respond (minutes)", 1, 121, 5, key="rag_timeout")
        open_rag = st.button("Open RAG")


    with tabs[2]:
        st.caption("If according to you and your workflow this makes sense we can put manual filter options here, "
                   "like filter for language, country, and whatever data we can get.")
        st.caption("However, for now this does not make a lot of sense as the only properties we have for each file are language and country, "
                   "which are the same over the whole database.")
        st.caption("I left it open for now as a useful implementation depends a lot on:")
        st.caption("1. your needs (seeing e.g. all spanish files does not make a lot of sense as they are too many, I imagine it barely useful to see a list of >4000 files here)")
        st.caption("2. the data that we can get (of course it would be nice to filter for e.g. dates, but can we get them? for all, or at least a multitude of document?)")


    with tabs[3]:
        st.header("Manage Collections here.")
        # initialize
        new_collection_name = st.text_input("**Initialize new** 🚀", placeholder="New collection name goes here", key="new_collection_name")
        initialize_new = st.button("Initialize", key="btn_init")
        # remove
        collection_to_remove = st.selectbox(label="**Remove a collection** 🪚", options=options, key="remove_collection")
        col_rem_1, col_rem_2 = st.columns(2, gap="small")
        with col_rem_1:
            remove_selected = st.button("Remove selected")
        with col_rem_2:
            remove_all = st.button("Remove all ")

        # populate
        populate_collection = st.selectbox(label="**Populate a collection** 🐝", options=options, key="populate_collection")
        col1_new_coll, col2_new_coll = st.columns(2, gap="small")
        with col1_new_coll:
            chunk_size = st.number_input("Chunk size (in words aka tokens - maximum 600-1000)",
                                         value=400,
                                         key="chunk_size")
        with col2_new_coll:
            overlap = st.number_input("Chunk Overlap (in words aka tokens)", value=100, key="chunk_overlap")
        populate_path = st.text_input("Path to OCR processed data folder",
                                      value = str(os.path.join(os.getcwd(), "pdf_data")),
                                      key="populate_path")

        batch_size = st.slider("Batchsize: How many chunks to insert in one step.", min_value=5, max_value=100, value=30, step=5, key="batch_size")

        populate = st.button("Populate")

        # get collection info and stats for one colleciton
        stats_collection = st.selectbox(label="**Show a collection** 🔬", options=options, key="stats_collection_name")
        show_stats_collection = st.button(label="Show me!", key="show_stats_collection")

        # optional maybe include later for sinlge files
        # pdf_upload = st.file_uploader("Upload .txt", type=["txt"], key="ingest_upload")
        # run_ingest = st.button("Ingest", key="btn_ingest")

    with tabs[4]:
        st.header("OCR")
        st.caption("In this section we will have access to the OCR engine for future data.\n")
        st.caption("For now this is just a dummy simulating the OCR process by copying the txt files into\n"
                   "the same folder where OCR would put them and where 'Populate' looks for them.\n")
        st.caption("In order to run this make sure the folder '2025-07-25_FAOLEX_Kolumbien_gesamt' is saved here:\n")
        pdf_path = str(os.path.join(os.getcwd(), "pdf_data"))
        st.write(f"**{pdf_path}**")
        st.caption("where 'app' is just the unzipped folder that I sent you, the main folder of this application.")

        # count pdfs in folder etc
        run_fake_ocr = st.button("1. Run fake-OCR 🔥", key="run_fake_ocr")
        st.caption(f"Then run the below sanity checks.")
        run_sanity_check = st.button("2. Run sanity check 🚑", key="run_sanity_check")


    with tabs[5]:
        st.header("Info")
        readme = st.button("Readme", key="readme")
        show_stats_total = st.button("Show stats", key="show_stats")


        default_pdf_root = pathlib.Path("pdf_data").resolve()                                       
        pdf_root_str = st.text_input("PDF root folder",
                                     value=str(default_pdf_root), key="set_pdf_root")              


# ------------------ update main states ----------------------------------
# for debugging
#st.write(f"Current State before update: {st.session_state["main_state"]}")

if open_search:
    st.session_state["main_state"] = "search"

# Trigger initialize screen
if initialize_new and new_collection_name.strip():
    st.session_state["main_state"] = "initialize"

# trigger the remove all aka reset screen
if remove_all:
    st.session_state["main_state"] = "reset"

# trigger remove selected
if remove_selected and collection_to_remove:
    st.session_state["main_state"] = "remove_collection"

# trigger populate collection:
if populate_path and populate and populate_collection and chunk_size and overlap and batch_size:
    st.session_state["main_state"] = "populate"

if show_stats_collection and stats_collection:
    st.session_state["main_state"] = "single_collection_stats"

if show_stats_total:
    st.session_state["main_state"] = "show_stats_total"

if run_fake_ocr:
    st.session_state["main_state"] = "run_fake_ocr"

if open_rag:
    st.session_state["main_state"] = "open_rag"

if readme:
    st.session_state["main_state"] = "readme"

if run_sanity_check:
    st.session_state["main_state"] = "sanity_check"



# ---------------- handle different main states ----------------
# for debugging
#st.write(f"Current State after update: {st.session_state["main_state"]}")

main_state = st.session_state.get("main_state")
# Handle initialize screen
if not main_state in ("welcome", "readme"):
    st.title(f"VeoVec — Database Manager")

if main_state == "welcome":
    path = os.path.join("app_data", "welcome_screen.png")
    welcome_image = Image.open(path)
    st.image(welcome_image)

elif  main_state == "initialize":
    db_name = st.session_state["new_collection_name"].strip().replace(" ", "_")
    client = _connect_client()
    try:
        # Use the proper existence check
        exists = client.collections.exists(db_name)

        if exists:
            st.warning(f"Collection '{db_name}' already exists. Overwrite?")
            c1, c2 = st.columns(2, gap="small")

            with c1:
                if st.button("✅ Proceed", key=f"overwrite_{db_name}"):
                    client.collections.delete(db_name)
                    initialize(client, db_name)
                    st.success(f"Recreated collection '{db_name}'.")
                    st.session_state["main_state"] = None
                    if st.button("OK."):
                        st.rerun()

            with c2:
                if st.button("❌ Abort", key=f"abort_{db_name}"):
                    st.info("Operation aborted.")
                    st.session_state["main_state"] = None
                    if st.button("OK."):
                        st.rerun()

        else:
            # Fresh create
            initialize(client, db_name)
            st.success(f"Created new collection '{db_name}'.")
            st.session_state["main_state"] = None
            if st.button("OK."):
                st.rerun()

    finally:
        _close_client(client)

# handle the reset screen
elif main_state == "reset":
    client = _connect_client()
    try:
        st.warning(f"Do you really want to reset and remove ALL collections?")
        c1, c2 = st.columns(2, gap="small")
        with c1:
            if st.button("✅ Proceed", key=f"remove_all"):
                client.collections.delete_all()
                st.success(f"All collections deleted successfully.")
                st.session_state["main_state"] = None
                if st.button("OK."):
                    st.rerun()
        with c2:
            if st.button("❌ Abort", key=f"abort_reset"):
                st.info("Operation aborted.")
                st.session_state["main_state"] = None
                if st.button("OK."):
                    st.rerun()
    finally:
        _close_client(client)

elif main_state == "remove_collection":
    client = _connect_client()
    collection_name = st.session_state["remove_collection"]
    try:
        st.warning(f"Do you really want to remove collection '{collection_name}'?")
        c1, c2 = st.columns(2, gap="small")
        with c1:
            if st.button("✅ Proceed", key=f"remove_{collection_name}"):
                client.collections.delete(collection_name)
                st.success(f"Collection '{collection_name}' removed successfully.")
                st.session_state["main_state"] = None
                if st.button("OK."):
                    st.rerun()
        with c2:
            if st.button("❌ Abort", key=f"abort_remove_{collection_name}"):
                st.info("Operation aborted.")
                st.session_state["main_state"] = None
                if st.button("OK."):
                    st.rerun()
    finally:
        _close_client(client)

# handel populate screen
elif main_state == "populate":
    populate_path = st.session_state["populate_path"]
    populate_collection = st.session_state["populate_collection"]
    client = _connect_client()
    try:
        populate_fun(client=client,
                     data_path=populate_path,
                     db_name=populate_collection,
                     chunk_size=chunk_size,
                     overlap=overlap,
                     batch_size=batch_size)
        # the screen shown when this works or does not work is all handled inside the function
        # so is the state update, keeps code clearer than introducing further states.
    except (weaviate.exceptions.WeaviateInsertManyAllFailedError, weaviate.exceptions.WeaviateBatchError) as e:
        st.error(f"'{e}': Population collection '{populate_collection}' failed. Your computer seems to be too slow for the chosen"
                 f"chunk size and/or batchsize.")
        st.info(f"In case this happened right at the start of populating  '{populate_collection}'"
                 f"reduce batchsize and/or chunk size and retry. Note that batchsize influences only HOW the insertion is done not the final result."
                f" In case this happened during the process it is better to reduce only the batch size and retry. "
                f"It will then restart from the point where it stopped.")
        st.warning(f"Why is this happening? When populating each text chunk needs to be embedded to a vector. "
                   f"Machine learning models often and for multiple reasons process data in batches, i.e. not one chunk at a time but "
                   f"batch size number of chunks at a time. In this application weaviate sends n=batch_size number of chunks to the model in one go "
                   f"and then waits for an answer. It does not wait forever though, there are limits beyond which further waiting does not make sense (computation is simply too"
                   f"slow for finishing in a reasonable amount of time). I set these waiting limits quite high, however, it might happen that on a weaker machine the embedding process is still to slow, "
                   f"consequentially weaviate has to wait too long for the models answer, and throws an error."
                   f"You can try solving this by two approaches: 1) reduce the batch size, i.e. send the model smaller batches, so it finishes working on them faster, 2) reduce the chunk size, i.e. send the model "
                   f"the same amount of chunks but each chunk is smaller. If this does not solve the problem we have to scale down for local machines; meaning: choose faster, smaller and probably dumber models when running this locally.")
    finally:
        _close_client(client)

elif main_state == "single_collection_stats":
    client = _connect_client()
    try:
        db_name = st.session_state["stats_collection_name"]
        #dataframe = get_summary_stats_for(client, db_name)
        #st.dataframe(dataframe)
        under_construction(text=f"Here you will find summary statistics on the chosen collection '{db_name}'. "
                                f"Example: how many files, countries, text chunks (data entries), languages, etc. are in this chosen collection.")
        if st.button("OK."):
            st.session_state["main_state"] = None
            st.rerun()
    finally:
        _close_client(client)

elif main_state == "run_fake_ocr":
    client = _connect_client()
    try:
        txt_path = os.path.join("app_data", "fake_ocr_data.json")
        target_folder = os.path.join(os.getcwd(), "pdf_data")
        unpack_txts(mapping_file = txt_path, target_folder = target_folder)
        st.session_state["main_state"] = None # this mus stand before the button, otherwise the button triggers rerun
        if st.button("OK."):
            st.rerun()
    except FileNotFoundError:
        st.error(" There is  problem with finding '2025-07-25_FAOLEX_Kolumbien_gesamt' and all its subfolders and pdfs inside of 'pdf_data'. "
                 "Please check if the folder is really there, folder names must perfectly match. Then retry.")
        st.warning("Below you see what I DID find in pdf_data, maybe this helps.")
        found_in_target = [f for f in os.listdir(os.path.join(os.getcwd(), "pdf_data"))]
        if len(found_in_target) == 0:
            st.write("'pdf_data' seems to be empty.")
        else:
            st.write("Folders and files found in 'pdf_data':")
            st.write(found_in_target)

    finally:
        _close_client(client)



elif main_state == "open_rag":
    default_rag_prompt = "Use the retrieved context to answer the question below. Your answer MUST be written in clear English, even if the provided context is not in English."

    st.write(f"selected collection: {st.session_state['rag_collection']}")
    grouped_task = st.text_area("**Metaprompt** - How to generate an answer (answer the following question, summarize, act as an expert for, etc). Only seen by the answer generating model",
                                value=default_rag_prompt,
                                key="grouped_task")

    rag_query = st.text_area(f"**RAG prompt** - The prompt on the task itself, also used for retrieving the information.",
                         placeholder="Your RAG prompt goes here.",
                         height=150, key="rag_query")
    rag_go = st.button("Generate answer.", key = "rag_go")

    if rag_go:
        response = None
        client = _connect_client(timeout = Timeout(init=60,
                                                   query=rag_timeout*60,
                                                   insert=600))
        try:
            # get rag output and show
            with st.spinner("Generating answer... please wait ⏳"):
                response, file_data = run_RAG(client,
                                       limit=rag_limit,
                                       query=rag_query,
                                       collection=rag_collection,
                                       grouped_task=grouped_task,
                                              alpha=rag_alpha)
                st.success("Done!")

            st.write(response)
        except weaviate.exceptions.WeaviateQueryError:
            st.error("Timeout expired before the model returned a response. Set higher timeout or lower limit on the text chunks to consider.")
        finally:
            _close_client(client)


        if not response is None:
            st.write(response.generative.text)

            # show context files
            if rag_number_context_shown > 0:
                st.write("The following files have found to be relevant for generating this response:")

                pdf_root = pathlib.Path(pdf_root_str).resolve()  # e.g. "/app/pdf_data"

                for i, file_dict in enumerate(file_data[0:rag_number_context_shown], 1):
                    file_path = file_dict.get("filepath", "unknown file")

                    rel_posix = pathlib.Path(file_path).as_posix()
                    abs_path = pdf_root / rel_posix
                    link_ok = abs_path.exists()

                    exp_title = f"{i}. {pathlib.Path(file_path).name}"

                    with st.expander(exp_title):
                        if link_ok:
                            st.markdown(pdf_link(rel_posix, label=rel_posix), unsafe_allow_html=True)
                        else:
                            st.caption(f"⚠️ Missing file under PDF root: {abs_path}")

                        st.write("**Relevant chunk:**")
                        st.write(file_dict.get("text_chunk", ""))

            st.write(" ")
            st.write("**Please leave your feedback here:**")

            feedback_data = {
                "limit": rag_limit,
                "query": rag_query,
                "collection": rag_collection,
                "grouped_task": grouped_task,
                "alpha": rag_alpha,
                "date": date.today().isoformat(),
            }

            def save_rag_feedback_callback(rag_collection, feedback_data):
                # This is called after the feedback widget changes
                rag_feedback = st.session_state["rag_feedback"]
                save_rag_feedback(rag_feedback, rag_collection, feedback_data)

            st.feedback(
                options="faces",
                width="stretch",
                key="rag_feedback",  # stored in st.session_state["rag_feedback"]
                on_change=save_rag_feedback_callback,
                kwargs={
                    "rag_collection": rag_collection,
                    "feedback_data": feedback_data,
                },
            )

            if st.button("Continue without feedback."):
                st.rerun()

        else:
            st.error("run_RAG did not return a valid response.")
            if st.button("OK"):
                st.rerun()

#default mode search - after this branch no state update to None, we want to stay here.

elif main_state == "search":
    # Main query box stays in the body (feels nicer for long text)
    query = st.text_area(f"**Selected collection:** {st.session_state['search_collection']}",
                         placeholder="Your free-text query goes here.",
                         height=150)
    run_search = st.button("Search 🔎", key="btn_search")

    if collection_name is None:
        st.warning("Please choose the collection you want to search in ('Search Settings' to the left)."
                   "If there is no collection available initialize and populate one using the 'Collections' tab.")
        if st.button("Done."):
            st.rerun()

    # Use the sidebar's Search button & controls  # changed
    if run_search and query.strip():  # changed
        client = _connect_client()  # reuse your connector
        try:
            if mode == "Hybrid":  # changed
                results = hybrid_search(query=query, client=client, alpha=alpha, limit=limit,
                                        collection_name=collection_name)
            else:
                results = semantic_search(query=query, client=client, limit=limit,
                                          collection_name=collection_name)

            if not results:
                count = client.collections.get(collection_name).aggregate.over_all().total_count
                if count == 0:
                    st.warning(f"The collection seems to be empty, therefore no results have been found."
                               f"Please use the 'Collections' tab to populate your collection '{collection_name}', then retry.")
                else:
                    st.info("No results.")
            else:

                pdf_root = Path("/app/pdf_data").resolve()  # still useful for sanity checks

                for i, r in enumerate(results, 1):
                    file_path = r.get("filepath", "unknown file")

                    # filepath is now RELATIVE (e.g., "a/b/c.pdf")
                    rel_posix = Path(file_path).as_posix()
                    abs_path = pdf_root / rel_posix
                    link_ok = abs_path.exists()

                    exp_title = f"{i}. {Path(file_path).name}"

                    with st.expander(exp_title, expanded=False):
                        if link_ok:
                            st.markdown(pdf_link(rel_posix, label=rel_posix), unsafe_allow_html=True)
                        else:
                            st.caption(f"⚠️ Missing file under PDF root: {abs_path}")

                        if "score" in r:
                            st.write(f"**Similarity score:** {r['score']:.4f}")
                        if "distance" in r:
                            st.write(f"**Distance:** {r['distance']:.4f}")

                        st.write("**Relevant chunk:**")
                        st.write(r.get("text_chunk", ""))

        finally:
            _close_client(client)  # changed

elif main_state == "show_stats_total":
    under_construction("Here we will be able to see summary statistics on the total database, i.e. all collections.")
    st.session_state["main_state"] = None
    st.button("OK.")

elif main_state == "readme":
    path_txt = os.path.join("app_data", "readme.txt")
    path_html = os.path.join("app_data", "readme.html")

    # rename only if html does not exist yet
    if os.path.exists(path_txt) and not os.path.exists(path_html):
        os.rename(path_txt, path_html)

    # load the HTML file
    with open(path_html, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=2000, scrolling=True)
    #st.markdown(html_content, unsafe_allow_html=True)
    st.session_state["main_state"] = None
    st.button("Leave.")

elif main_state == "sanity_check":
    run_sanity_check_fun()
    st.session_state["main_state"] = None
    st.button("OK")


if __name__ == "__main__":
    today = date.today()
    formatted_date = f"{today.year}_{today.month}_{today.day}"
    print(formatted_date)