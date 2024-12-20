from io import BytesIO
from qdrant_client import QdrantClient
import streamlit as st
import base64
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image


# Model
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")
model.eval()
collection_name = "fashion"


if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None

def embedded_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt")
    embedding = None
    with torch.no_grad():
        embedding = model(**inputs)
    return embedding.last_hidden_state[:, 0, :].squeeze().tolist()

@st.cache_resource
def get_client():
    return QdrantClient(
        url=st.secrets.get("qdrant_url"),
        api_key=st.secrets.get("qdrant_key")
    )

def get_records(query):
    client = get_client()

    results = client.query_points(
        collection_name=collection_name,
        query=query, # <--- Dense vector
        limit=21
    )

    return results.points


def find_similar(image_bytes):
    # Process Image to Vectors
    st.session_state.image_bytes = image_bytes


def get_bytes_from_base64(base64_string):
    return BytesIO(base64.b64decode(base64_string))

# File Uploaders
uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"]) 

# Results' Header
if uploaded_image:
    # Check Value in session_state
    if not st.session_state.image_bytes:
        st.session_state.image_bytes = uploaded_image.getvalue()

    # Display Uploaded or Selected Image
    displayed_image = st.image(
        image=st.session_state.image_bytes
    )
    # Process Image to Vectors
    vectors = embedded_image(st.session_state.image_bytes)

    # Similar Products
    st.header("Similar Products:")
    st.divider()

    # Separate in columns
    column = st.columns(3)
    records = get_records(vectors)

    for idx, record in enumerate(records):
        column_idx = idx % 3
        image_bytes = get_bytes_from_base64(record.payload['base64'])
        with column[column_idx]:
            st.image(
                image=image_bytes
            )
            st.button(
                label="Find Similar",
                key=record.id,
                on_click=find_similar,
                args=[record.payload['base64']]
            )    