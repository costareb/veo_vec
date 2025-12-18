import os
import json

from typing import List, Tuple
import json, time
from pathlib import Path
import streamlit as st
from datetime import date

############### Pack txt files, this function is not needed in the app but for me to pack and ship them. ############

def pack_txts(folder: str, out_file: str) -> None:
    """
    Walk `folder`, find all .txt files, and save a single JSON file containing
    a list of tuples: (content, relative_path).

    - `content` is the UTF-8 text of the file (errors replaced).
    - `relative_path` is the path of the .txt relative to `folder`.
    """
    root = Path(folder).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    records: List[Tuple[str, str]] = []

    for p in root.rglob("*.txt"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()  # stable, POSIX-style slashes
            text = p.read_text(encoding="utf-8", errors="replace")
            records.append((text, rel))

    # Save as JSON (compact but readable). For huge corpora, gzip it.
    out_path = Path(out_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"Packed {len(records)} .txt files from {root} -> {out_path}")


################################## Unpack/redistribute .txts #########################

def unpack_txts(mapping_file: str, target_folder: str, overwrite: bool = True) -> None:
    """
    Load the previously saved JSON `mapping_file` and recreate all .txt files
    under `target_folder` using the same relative paths.
    """

    st.write(f"Target folder: {target_folder}")
    src = Path(mapping_file).resolve()
    if not src.is_file():
        st.error("No file providing the txt data found. Fake OCR failed.")
        st.session_state["main_display"] = None
        time.sleep(4)
        st.rerun()

    target_root = Path(target_folder).resolve()

    try:
        records = json.loads(src.read_text(encoding="utf-8"))
    except Exception:
        st.error("The TXT mapping file is not valid JSON. Fake OCR failed.")
        st.session_state["main_display"] = None
        time.sleep(4)
        st.rerun()

    if not isinstance(records, list):
        st.error("Invalid mapping format (expected a list). Fake OCR failed.")
        st.session_state["main_display"] = None
        time.sleep(4)
        st.rerun()

    total = len(records)
    if total == 0:
        st.info(f"No entries found in mapping. Nothing to write into {target_root}.")
        return

    written = 0
    skipped = 0

    progress = st.progress(0.0, text="Starting...")
    status = st.empty()

    for i, item in enumerate(records, start=1):
        progress.progress(i / total, text=f"Reading... {i}/{total} - {100*i/total:.2f}%")

        # Expect tuples/lists of [content, relative_path]
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            st.error("There is something wrong with the txt providing file. Fake OCR failed.")
            st.session_state["main_display"] = None
            time.sleep(4)
            st.rerun()

        content, rel = item

        # Normalize/validate relative path
        rel_path = Path(str(rel)).as_posix().lstrip("/")
        dest = (target_root / rel_path).resolve()

        # security check:
        if target_root not in dest.parents:
            raise ValueError(f"Unsafe path detected: {dest}")

        if not dest.parent.exists():
            raise FileNotFoundError(f"Destination folder does not exist: {dest.parent}")

        if dest.exists() and not overwrite:
            skipped += 1
            continue

        dest.write_text(str(content), encoding="utf-8")
        written += 1

    status.success(f"Unpacked {written} files into {target_root} (skipped {skipped}).")
    st.session_state["main_display"] = None

def json_files_equal(file1: str, file2: str):
    path1, path2 = Path(file1), Path(file2)
    with path1.open("r", encoding="utf-8") as f1, path2.open("r", encoding="utf-8") as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)
    return data1 == data2


if __name__ == "__main__":
    #pack_txts(folder="pdf_data", out_file="fake_ocr_data_check.json")
    print(json_files_equal(file1="fake_ocr_data_check.json",
                           file2="app_data/fake_ocr_data.json" ))
