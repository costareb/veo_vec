import json
import streamlit as st
import os

def run_sanity_check_fun():
    check_file = os.path.join("app_data", "fake_ocr_data_check.json")
    try:
        with open(check_file, "r", encoding="utf-8") as f:
            data = json.load(f) # at intex 0 the text at index 1 the path

        progress_text = st.empty()
        progress_bar = st.progress(0)

        total = len(data)

        for i, (test_content, test_path) in enumerate(data):
            progress_text.text(
                f"Checking... {i + 1}/{total} - {100 * (i + 1) / total:.2f} % - In test version total should be 4019")
            progress_bar.progress((i + 1) / total)

            test_path = os.path.join("pdf_data", test_path)
            with open(test_path, "r", encoding="utf-8") as f:
                found_text = f.read()
                assert(found_text == test_content)

        st.success("Sanity check done. You are ready for initializing a collection and populating it.")
    except AssertionError or FileNotFoundError:
        st.error("Sanity check failed.")


if __name__ == "__main__":
    run_sanity_check_fun()