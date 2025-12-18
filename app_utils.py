

import hashlib
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
import uuid
from PIL import Image
from weaviate.classes.init import AdditionalConfig, Timeout
from pathlib import Path
import os, json
import pandas as pd
from datetime import datetime
import streamlit as st
from weaviate.classes.data import DataObject



################ helper functions ###############
def _connect_client(timeout=Timeout(init=10, query=300)):
    host = os.getenv("WEAVIATE_HOST", "localhost")
    ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")

    client = weaviate.connect_to_custom(
        http_host=host,
        http_port=8080,
        http_secure=False,
        grpc_host=host,
        grpc_port=50051,
        grpc_secure=False,
        additional_config=AdditionalConfig(
            timeout=timeout,
            # MOST IMPORTANT: tell client about modules
            headers={"X-Openai-Api-Key": "dummy",  # required even if unused
                     "X-Ollama-BaseUrl": ollama_url,
                     "X-Ollama-Model": "nomic-embed-text"}
        ),
    )
    return client

def _close_client(client):
    client.close()
    #print("Weaviate client closed successfully \u2713")
    #print("_" * 50)

def _get_collection(client, db_name):
    return client.collections.get(db_name)

def _get_txt_paths(path_):
    if not isinstance(path_, Path):
        path_ = Path(path_)
    return list(path_.rglob("*.txt"))  # paths are relative to the path_

def _compute_file_id(pdf_path: Path) -> str:
    h = hashlib.sha256()
    with pdf_path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):  # 1MB chunks
            h.update(block)
    return h.hexdigest()

def _make_chunk_uuid(file_id: str, chunk_index: int) -> str:
    """Deterministic UUID derived from file_id + chunk index"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_id}:{chunk_index}"))

def _prefetch_existing_file_ids(collection) -> set[str]:
    existing = set()
    # or specify cache size:
    for obj in collection.iterator(return_properties=["file_id"], cache_size=512):
        fid = obj.properties.get("file_id")
        if isinstance(fid, str):
            existing.add(fid)
    return existing


def add_new_collection_to_feedback(d: dict, collection_name: str, pop_log, chunk_size, overlap, batch_size=1):
    base = collection_name

    # Step 1: If the base key exists, shift it to a numbered slot
    if base in d:
        if len(d[base]["rag_feedback_data"]) > 0: # do this only if there is feedback data stored, otherwise useless to keep
            counter = 1
            # Find the first free numbered key
            while f"{base}_{counter}" in d:
                counter += 1

            # Shift existing base key upward
            d[f"{base}_{counter}"] = d.pop(base)

    # Step 2: Insert the new (always unnumbered) key with value None
    d[base] = {"pop_time": datetime.now().isoformat(),
               "pop_structure": (chunk_size, overlap, batch_size),
               "pop_log":pop_log,
               "rag_feedback_data":[]}

    return None
###########################################################################

############################### Populate a Collection #####################

# max words and overlaps are important hyperparameters may be crucial to tune
def _chunk_text(text: str, max_words: int = 600, overlap: int = 100):
    words = text.split()
    chunks = []
    start = 0
    step = max_words - overlap
    if step <= 0:
        raise ValueError("chunk_size must be larger than overlap")

    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        start += step
    return chunks

# nomic-embed-text supports up to roughly 8192 tokens input (depending on the quantization),
# but for best semantic performance, it’s ideal to stay below 500–1000 tokens per chunk.
def add_txt(
    txt_file_path: Path,
    collection,
    pdf_filepath_abs: Path,
    pdf_root: Path,
    file_id: str,
    chunk_size: int,
    overlap: int,
    batch_size: int = 1,
):
    """
    Add text chunks from a txt file to the Weaviate collection, batching inserts
    to avoid Ollama timeout issues.

    :param txt_file_path: Path to the .txt file
    :param collection: Weaviate collection object
    :param pdf_filepath_abs: Absolute path to the corresponding PDF (can be host or container)
    :param pdf_root: Root folder under which PDFs live (used to store relative paths)
    :param file_id: Unique hash of the PDF file
    :param chunk_size: Maximum words per chunk
    :param overlap: Word overlap between chunks
    :param batch_size: Number of chunks per insert_many call
    """

    txt_file_path = Path(txt_file_path)
    pdf_filepath_abs = Path(pdf_filepath_abs).resolve()
    pdf_root = Path(pdf_root).resolve()

    with open(txt_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    chunked_document = _chunk_text(raw_text, max_words=chunk_size, overlap=overlap)

    pdf_filename = pdf_filepath_abs.name

    # Store relative path under pdf_root, normalized to posix for URLs
    try:
        pdf_rel_path = pdf_filepath_abs.relative_to(pdf_root).as_posix()
    except ValueError:
        # If somehow outside root, fall back to filename only (but warn in logs)
        pdf_rel_path = pdf_filename
        print(f"[WARN] PDF {pdf_filepath_abs} not under root {pdf_root}. Storing filename only.")

    country = "Venezuela"
    language = "Spanish"

    objs = [
        DataObject(
            properties={
                "text_chunk": chunk,
                "filename": pdf_filename,
                "filepath": pdf_rel_path,
                "language": language,
                "country": country,
                "chunk_index": i,
                "file_id": file_id,
            },
            uuid=_make_chunk_uuid(file_id, i),
        )
        for i, chunk in enumerate(chunked_document)
    ]

    # Insert in batches
    for j in range(0, len(objs), batch_size):
        batch = objs[j : j + batch_size]
        collection.data.insert_many(batch)


def populate_fun(
    data_path: str = None,
    client=None,
    db_name: str = None,
    chunk_size: int = None,
    overlap: int = None,
    batch_size: int = 1,
):
    """
    :param data_path: str - the path where we find the txt files, ocr already applied.
                           This is also treated as the PDF root.
    :return: None
    """

    if data_path is None:
        st.error("No data_path provided.")
        return

    collection = client.collections.get(db_name)

    pdf_root = Path(data_path).resolve()   # IMPORTANT: root for relative paths
    print(f"\nPopulating database from folder called '{pdf_root}'")
    print(f"Path: {os.path.abspath(pdf_root)}")

    all_txt = _get_txt_paths(pdf_root)

    # Prefetch existing file_ids from DB
    existing_file_ids = _prefetch_existing_file_ids(collection)
    seen_this_run = set()

    new_files = 0
    duplicates_list = []
    missing_pdf_list = []
    already_seen_list = []

    progress_text = st.empty()
    progress_bar = st.progress(0)
    total = len(all_txt)

    for i, txt_file in enumerate(all_txt):
        progress_text.text(
            f"Populating... {i + 1}/{total} - {100*(i + 1)/total:.2f} % "
            f"- In test version total should be 4019"
        )
        progress_bar.progress((i + 1) / total)

        txt_file = Path(txt_file)

        # Find matching PDF and ensure it exists
        pdf_filepath_abs = txt_file.with_suffix(".pdf").resolve()
        if not pdf_filepath_abs.is_file():
            missing_pdf_list.append(txt_file)
            continue

        # get file id of pdf
        file_id = _compute_file_id(pdf_filepath_abs)

        if file_id in seen_this_run:
            duplicates_list.append(pdf_filepath_abs)
            continue

        if file_id in existing_file_ids:
            already_seen_list.append(pdf_filepath_abs)
            continue

        # mark as seen this run
        seen_this_run.add(file_id)

        add_txt(
            txt_file_path=txt_file,
            collection=collection,
            pdf_filepath_abs=pdf_filepath_abs,
            pdf_root=pdf_root,
            file_id=file_id,
            chunk_size=chunk_size,
            overlap=overlap,
            batch_size=batch_size,
        )
        new_files += 1

    # --- logging / feedback ---
    filename = "population_log.json"
    feedback_dir = Path(os.getcwd()) / "feedback_collector"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    filepath = feedback_dir / filename

    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    stats_dict = {
        "total_files_found": total,
        "files_already_seen": len(already_seen_list),
        "files_added": new_files,
        "duplicates_in_this_run": len(duplicates_list),
        "txt_files_missing_PDF": len(missing_pdf_list),
    }

    add_new_collection_to_feedback(
        d=data,
        collection_name=db_name,
        pop_log=stats_dict,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    data.setdefault(db_name, {})

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    progress_text.text("✅ Done!")
    st.success("Population complete.")

    problem = (
        len(already_seen_list) > 0
        or len(duplicates_list) > 0
        or len(missing_pdf_list) > 0
    )

    df_stats = pd.DataFrame.from_dict(stats_dict, orient="index", columns=["Count"])

    if problem:
        timestamp = datetime.now().isoformat().replace(":", "-")
        filename_txt_report = f"population_report_{db_name}_{timestamp}.txt"
        st.warning(
            f"Populated {st.session_state['populate_collection']} successfully but problems encountered. "
            f"Report saved to {filename_txt_report}."
        )
        st.dataframe(df_stats)

        with open(filename_txt_report, "w", encoding="utf-8") as f:
            f.write(
                f"\tTotal files found:                                          {total}\n"
                f"\tFiles already seen                                          {len(already_seen_list)}\n"
                f"\tFiles added:                                                {new_files}\n"
                f"\tFiles that existed more than once in this run (duplicates): {len(duplicates_list)}\n"
                f"\tTxt files with missing pdf:                                 {len(missing_pdf_list)}\n\n"
                "\tPDFs that are already in the DB:\n"
                f"{already_seen_list}\n\n"
                "\tDuplicates hence ignored (only the first instance of each file is added):\n"
                f"{duplicates_list}\n\n"
                "\tTxt without pdf, hence ignored:\n"
                f"{missing_pdf_list}\n"
            )
    else:
        st.dataframe(df_stats)

    st.session_state["main_state"] = None
    if st.button("OK."):
        st.rerun()


################################ Populate collection end #####################


###################### initialize start #########################
# change models here!!
def initialize(client, db_name):
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    client.collections.create(
        name=db_name,
        vector_config=Configure.Vectors.text2vec_ollama(
            api_endpoint=ollama_url,
            model="nomic-embed-text",
            source_properties=["text_chunk"],
            vectorize_collection_name=False
        ),
        generative_config=Configure.Generative.ollama(
            api_endpoint=ollama_url,
            model="mistral-nemo"
        ),
        properties=[
            Property(name="text_chunk", data_type=DataType.TEXT),
            Property(name="filename", data_type=DataType.TEXT),
            Property(name="filepath", data_type=DataType.TEXT),
            Property(name="language", data_type=DataType.TEXT),
            Property(name="country", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="file_id", data_type=DataType.TEXT),
        ],
    )

################################### initialize end ###################################


####################################### RAG #########################################

def run_RAG(client, query, limit, collection, grouped_task, alpha):
    """
    :param client:
    :param query:
    :param limit:
    :param collection: a string, the collection name
    :return:
    """
    coll = client.collections.get(collection)
    res = coll.generate.hybrid(
        query=query,
        limit=limit,
        return_properties=["text_chunk", "filename", "filepath", "country"],
        grouped_task=(grouped_task),
        alpha=alpha,
    )

    files_considered = []

    try:
        for obj in res.objects:
            props = obj.properties
            filename = props.get("filename")
            filepath = props.get("filepath")
            text_chunk = props.get("text_chunk")

            meta = getattr(obj, "metadata", None)
            score = None

            # TODO: find how to get score, for now remains None

            # Only add unique filenames, preserving relevance order
            if filename and not any(f["filename"] == filename for f in files_considered):
                files_considered.append({"filename":filename,
                                         "score": score,
                                         "filepath": filepath,
                                         "text_chunk":text_chunk})

    except AttributeError:
        # in case something does not work with unpacking the response
        st.error("attribute error in response")
        pass

    return res, files_considered

############################################ RAG end ################################

def under_construction(text):
    path = os.path.join("app_data", "under_construction.png")
    st.write(text)

    construction_image = Image.open(path)
    construction_image = construction_image.resize((600, 400))
    st.image(construction_image)

############################# search functions ##########################################

def semantic_search(query, client, limit, collection_name):
    # get collection
    collection = client.collections.get(collection_name)

    # get responses
    response = collection.query.near_text(
        query=query,
        limit=None,
        return_metadata=MetadataQuery(distance=True)
    )

    results = []
    already_shown = set()
    for o in response.objects:
        file = o.properties["filepath"]
        if file in already_shown:
            continue
        already_shown.add(file)
        results.append({
            "filepath": file,
            "distance": o.metadata.distance,
            "text_chunk": o.properties["text_chunk"],
        })
        if len(already_shown) >= limit:
            break
    return results


def hybrid_search(query, client, alpha, limit, collection_name):
    collection = client.collections.get(collection_name)

    response = collection.query.hybrid(
        query=query,
        alpha=alpha,
        limit=None,
        return_metadata=MetadataQuery(score=True, explain_score=True)
    )

    results = []
    already_shown = set()
    for o in response.objects:
        file = o.properties["filepath"]
        if file in already_shown:
            continue
        already_shown.add(file)
        results.append({
            "filepath": file,
            "score": o.metadata.score,
            "explain": getattr(o.metadata, "explain_score", None),
            "text_chunk": o.properties["text_chunk"],
        })
        if len(already_shown) >= limit:
            break
    return results

####################### search functions end #############################################

####################### rag feedback saver #############################################

def save_rag_feedback(rag_feedback, rag_collection, feedback_data):

    filename = f"population_log.json"
    folder_path = os.path.join(os.getcwd(), "feedback_collector")
    full_path = os.path.join(os.getcwd(), "feedback_collector", filename)

    # Ensure directory exists
    os.makedirs(os.path.dirname(folder_path), exist_ok=True)

    found_database = False
    if os.path.exists(full_path):
        found_database = True
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        st.warning("Did not find the corresponding collection in the population logs. This warning regards only the feedback and documentation.")
        data =  {rag_collection:{"pop_time": None,
                                 "pop_structure": (None, None),
                                 "pop_log": None,
                                 "rag_feedback_data": [rag_feedback, feedback_data]}}

    if found_database:
        data[rag_collection]["rag_feedback_data"].append((rag_feedback, feedback_data))

    # Save back to the same file
    with open(full_path, "w") as f:
        json.dump(data, f)

    st.info("Feedback saved. Thank you.")

    return True


####################### show collecitons stats - unfinished  #############################################


def show_collections(client):
    need_to_close = False
    if client is None:
        need_to_close = True
        client = _connect_client()
    try:
        existing = [c for c in client.collections.list_all()]
        print("Collections:")
        print(existing, "\n")
        return existing
    finally:
        if need_to_close:
            _close_client(client)
        else:
            pass



if __name__ == "__main__":
    pass


