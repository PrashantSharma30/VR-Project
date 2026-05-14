import streamlit as st
import torch
import faiss
import pickle
import numpy as np
import os
import zipfile

from huggingface_hub import (
    snapshot_download,
    hf_hub_download
)

from ultralytics import YOLO
from transformers import (
    CLIPModel,
    CLIPProcessor
)

from PIL import Image

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Visual Product Search Engine",
    layout="wide"
)

st.title("Visual Product Search Engine")

st.write(
    "Upload a full-body image and choose "
    "which clothing region to search."
)

# =====================================================
# HUGGING FACE DATASET
# =====================================================

HF_REPO = (
    "Sharma-30/fashion-retrieval-assets"
)

# =====================================================
# DOWNLOAD ASSETS
# =====================================================

@st.cache_resource
def setup_assets():

    # ---------------------------------------------
    # DOWNLOAD clip_finetuned + embeddings
    # ---------------------------------------------

    dataset_path = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns=[
            "clip_finetuned/*",
            "embeddings/*"
        ]
    )

    # ---------------------------------------------
    # DOWNLOAD gallery.zip
    # ---------------------------------------------

    gallery_zip = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename="gallery.zip"
    )

    # ---------------------------------------------
    # EXTRACT gallery.zip
    # ---------------------------------------------

    extract_path = os.path.join(
        dataset_path,
        "gallery"
    )

    if not os.path.exists(
        extract_path
    ):

        with zipfile.ZipFile(
            gallery_zip,
            "r"
        ) as zip_ref:

            zip_ref.extractall(
                extract_path
            )

    return (
        dataset_path,
        extract_path
    )

dataset_path, gallery_path = (
    setup_assets()
)

# =====================================================
# PATHS
# =====================================================

CLIP_PATH = os.path.join(
    dataset_path,
    "clip_finetuned"
)

EMB_PATH = os.path.join(
    dataset_path,
    "embeddings"
)

INDEX_PATH = os.path.join(
    EMB_PATH,
    "index_C.bin"
)

META_PATH = os.path.join(
    EMB_PATH,
    "meta_C.pkl"
)

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_all():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ---------------------------------------------
    # CLIP
    # ---------------------------------------------

    clip_model = CLIPModel.from_pretrained(
        CLIP_PATH
    ).to(device)

    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    clip_model.eval()

    # ---------------------------------------------
    # YOLO
    # ---------------------------------------------

    yolo_model = YOLO("yolov8n.pt")

    # ---------------------------------------------
    # FAISS
    # ---------------------------------------------

    index = faiss.read_index(
        INDEX_PATH
    )

    # ---------------------------------------------
    # METADATA
    # ---------------------------------------------

    with open(META_PATH, "rb") as f:

        metadata = pickle.load(f)

    # ---------------------------------------------
    # BUILD IMAGE PATH MAP
    # ---------------------------------------------

    all_images = {}

    for root, dirs, files in os.walk(gallery_path):

        for file in files:

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                all_images[file] = os.path.join(
                    root,
                    file
                )

    # ---------------------------------------------
    # FIX IMAGE PATHS
    # ---------------------------------------------

    for item in metadata:

        filename = item["image_name"]

        if filename in all_images:

            item["image_path"] = (
                all_images[filename]
            )

        else:

            item["image_path"] = None

    return (
        device,
        clip_model,
        clip_processor,
        yolo_model,
        index,
        metadata
    )

(
    device,
    clip_model,
    clip_processor,
    yolo_model,
    index,
    metadata
) = load_all()

# =====================================================
# IMAGE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Full Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# MAIN PIPELINE
# =====================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader(
        "Uploaded Image"
    )

    st.image(
        image,
        width=300
    )

    # ---------------------------------------------
    # SAVE TEMP IMAGE
    # ---------------------------------------------

    temp_path = "temp.jpg"

    image.save(temp_path)

    # ---------------------------------------------
    # YOLO DETECTION
    # ---------------------------------------------

    results = yolo_model(temp_path)

    boxes = results[0].boxes

    person_box = None

    for box in boxes:

        cls_id = int(box.cls[0])

        label = yolo_model.names[cls_id]

        if label == "person":

            person_box = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            break

    # =================================================
    # CROPPING
    # =================================================

    if person_box is not None:

        x1, y1, x2, y2 = map(
            int,
            person_box
        )

        person_crop = image.crop(
            (x1, y1, x2, y2)
        )

        w, h = person_crop.size

        upper_crop = person_crop.crop(
            (
                0,
                0,
                w,
                int(h * 0.65)
            )
        )

        lower_crop = person_crop.crop(
            (
                0,
                int(h * 0.45),
                w,
                h
            )
        )

        # =============================================
        # DISPLAY OPTIONS
        # =============================================

        st.subheader(
            "Choose Clothing Region"
        )

        cols = st.columns(3)

        with cols[0]:

            st.image(person_crop)

            full_btn = st.button(
                "Search Full Body"
            )

        with cols[1]:

            st.image(upper_crop)

            upper_btn = st.button(
                "Search Upper Body"
            )

        with cols[2]:

            st.image(lower_crop)

            lower_btn = st.button(
                "Search Lower Body"
            )

        # =============================================
        # USER SELECTION
        # =============================================

        selected_crop = None

        if full_btn:
            selected_crop = person_crop

        elif upper_btn:
            selected_crop = upper_crop

        elif lower_btn:
            selected_crop = lower_crop

        # =============================================
        # RETRIEVAL
        # =============================================

        if selected_crop is not None:

            st.subheader(
                "Retrieved Products"
            )

            inputs = clip_processor(
                images=selected_crop,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():

                outputs = (
                    clip_model.vision_model(
                        pixel_values=inputs[
                            "pixel_values"
                        ]
                    )
                )

                pooled_output = (
                    outputs.pooler_output
                )

                query_embedding = (
                    clip_model.visual_projection(
                        pooled_output
                    )
                )

            query_embedding = (
                query_embedding
                .cpu()
                .numpy()
            )

            query_embedding = (
                query_embedding /
                np.linalg.norm(
                    query_embedding,
                    axis=1,
                    keepdims=True
                )
            )

            TOP_K = 5

            distances, indices = (
                index.search(
                    query_embedding,
                    TOP_K
                )
            )

            # =========================================
            # DISPLAY RESULTS
            # =========================================

            result_cols = st.columns(
                TOP_K
            )

            for rank, idx in enumerate(
                indices[0]
            ):

                item = metadata[idx]

                with result_cols[rank]:

                    if item["image_path"] is not None:

                        try:

                            retrieved_img = Image.open(
                                item["image_path"]
                            ).convert("RGB")

                            st.image(
                                retrieved_img
                            )

                            st.write(
                                f"Rank {rank+1}"
                            )

                            st.write(
                                f"Score: {distances[0][rank]:.3f}"
                            )

                            st.caption(
                                item.get(
                                    "caption",
                                    ""
                                )
                            )

                        except:

                            st.write(
                                "Could not load image"
                            )

                    else:

                        st.write(
                            "Image not found"
                        )

    else:

        st.error(
            "No person detected in image."
        )
