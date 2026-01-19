import sys
import os
from PIL import Image
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Backend.rag_chain import ask_multimodal

st.set_page_config(page_title="Enterprise Multimodal RAG", layout="wide")
st.title("🏢 Enterprise Knowledge Assistant")

query = st.text_input(
    "Ask a question from company policy documents:",
    placeholder="e.g. Show workflow diagram for leave approval"
)

if st.button("Search"):
    with st.spinner("Searching documents and images..."):
        texts, image_paths = ask_multimodal(query)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📄 Text Information")
        if texts:
            for t in texts:
                st.write(t)
                st.divider()
        else:
            st.write("No relevant text found.")

    with col2:
        st.subheader("🖼️ Visual References (Images)")
        if image_paths:
            for img_path in image_paths:
                if os.path.exists(img_path):
                    st.image(img_path, caption="Found in Policy Document", width=250)
                else:
                    st.warning(f"Image not found at path: {img_path}")
        else:
            st.info("No relevant images found for this query.")
