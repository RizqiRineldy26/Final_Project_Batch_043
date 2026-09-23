import sys


def check(name, fn):
    print(f"IMPORTING: {name}", flush=True)
    fn()
    print(f"OK: {name}", flush=True)


# Ordered to roughly match prediction_test.py so we can pinpoint the culprit.
check("streamlit", lambda: __import__("streamlit"))
check("numpy", lambda: __import__("numpy"))
check("pandas", lambda: __import__("pandas"))
check("tensorflow", lambda: __import__("tensorflow"))
check("nltk", lambda: __import__("nltk"))
check("supabase", lambda: __import__("supabase"))
check("langchain_google_genai", lambda: __import__("langchain_google_genai"))
check("langchain_community.document_loaders", lambda: __import__("langchain_community.document_loaders"))
check("chromadb", lambda: __import__("chromadb"))
check("langchain_chroma", lambda: __import__("langchain_chroma"))
check("langchain_text_splitters", lambda: __import__("langchain_text_splitters"))
check("langchain_core.messages", lambda: __import__("langchain_core.messages"))
check("langchain_core.prompts", lambda: __import__("langchain_core.prompts"))
check("langchain_core.output_parsers", lambda: __import__("langchain_core.output_parsers"))

print("ALL IMPORTS SUCCEEDED", flush=True)

import streamlit as st  # noqa: E402

st.title("Import diagnostic")
st.write("All imports succeeded — check the deploy logs above for the exact order.")
