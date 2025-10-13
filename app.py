# app.py
import os
import glob
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

st.set_page_config(page_title="Tree Species CNN", layout="wide")

# --- CONFIG ---
DEFAULT_REPO_PATH = "Tree_Species_Dataset"  # adjust if your dataset folder has a different name
MODEL_FILENAME = "improved_cnn_model.h5"
IMG_SIZE = (128, 128)  # can be increased if you have memory
BATCH_SIZE = 32
EPOCHS = 15  # default; user can change in UI

# --- HELPERS ---
def find_dataset_path():
    # If DEFAULT_REPO_PATH exists use it, else try to find directory with many subfolders
    if os.path.isdir(DEFAULT_REPO_PATH):
        return DEFAULT_REPO_PATH
    # fallback: look for directories in current working directory with >1 subfolders
    for entry in os.listdir("."):
        if os.path.isdir(entry):
            subdirs = [d for d in os.listdir(entry) if os.path.isdir(os.path.join(entry, d))]
            if len(subdirs) >= 2:
                # likely a dataset
                return entry
    return None

@st.cache_data
def gather_image_paths(dataset_path):
    image_paths = []
    labels = []
    for class_name in sorted(os.listdir(dataset_path)):
        class_folder = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_folder):
            for fname in os.listdir(class_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_paths.append(os.path.join(class_folder, fname))
                    labels.append(class_name)
    return image_paths, labels

def build_simple_cnn(input_shape=(128,128,3), num_classes=30):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

@st.cache_resource
def load_trained_model(path):
    if os.path.exists(path):
        return load_model(path)
    return None

def load_and_preprocess_image(img: Image.Image, img_size=IMG_SIZE):
    img = img.convert("RGB")
    img = img.resize(img_size)
    arr = img_to_array(img) / 255.0
    return arr

# --- UI ---
st.title("🌳 Tree Species Identification — Streamlit CNN")
st.markdown(
    """
This app provides basic dataset inspection, training, evaluation, and inference flows for the CNN model
used to classify tree species (adapted from your notebook).
"""
)

# Sidebar: dataset & model controls
with st.sidebar:
    st.header("Settings")
    dataset_path = find_dataset_path()
    st.write("Detected dataset folder:", dataset_path if dataset_path else "Not found")
    model_file = st.text_input("Model filename", value=MODEL_FILENAME)
    img_size = st.selectbox("Image size", options=["(64,64)","(128,128)","(224,224)"], index=1)
    if img_size == "(64,64)":
        IMG = (64,64)
    elif img_size == "(224,224)":
        IMG = (224,224)
    else:
        IMG = (128,128)
    batch_size = st.number_input("Batch size", min_value=8, max_value=128, value=BATCH_SIZE, step=8)
    epochs = st.number_input("Epochs", min_value=1, max_value=200, value=EPOCHS, step=1)
    st.markdown("---")
    st.write("Model controls:")
    load_model_btn = st.button("Load model from disk")
    train_model_btn = st.button("Train model")
    st.write("Tip: training in Streamlit runs in the same process; for long runs use a dedicated training script or smaller epochs.")
    st.markdown("---")
    st.write("Inference:")
    uploaded_file = st.file_uploader("Upload an image for prediction", type=["png","jpg","jpeg"])

# Main area
if dataset_path and os.path.isdir(dataset_path):
    st.subheader("Dataset overview")
    image_paths, labels = gather_image_paths(dataset_path)
    st.write(f"Found **{len(set(labels))}** classes and **{len(image_paths)}** images.")
    class_counts = {}
    for l in labels:
        class_counts[l] = class_counts.get(l, 0) + 1
    # show top 8 classes by count
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    st.write("Top classes by image count:")
    for cls, cnt in sorted_counts[:8]:
        st.write(f"- {cls}: {cnt} images")
    # show sample images
    st.write("Sample images:")
    cols = st.columns(6)
    sample_paths = image_paths[:6]
    for c, p in zip(cols, sample_paths):
        try:
            c.image(Image.open(p), caption=os.path.basename(os.path.dirname(p)), use_column_width=True)
        except Exception as e:
            c.write("Couldn't load image")
else:
    st.warning(
        "Dataset folder not found. Put your dataset in a folder named "
        "`Tree_Species_Dataset` or run the notebook's cloning step and place dataset in the working directory."
    )

# Load model on demand
model_loaded = None
if load_model_btn:
    try:
        model_loaded = load_trained_model(model_file)
        if model_loaded:
            st.success(f"Loaded model from `{model_file}`.")
        else:
            st.error(f"No model file found at `{model_file}`.")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Training flow
if train_model_btn:
    if not dataset_path:
        st.error("No dataset found — cannot train.")
    else:
        st.info("Preparing data generators...")
        # prepare data generators with basic split
        all_classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        num_classes = len(all_classes)
        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15,
                                           rotation_range=20, width_shift_range=0.1,
                                           height_shift_range=0.1, horizontal_flip=True, shear_range=0.1, zoom_range=0.1)
        train_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=IMG,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        val_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=IMG,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        st.write(f"Classes ({num_classes}): {all_classes}")
        st.write("Building model...")
        model = build_simple_cnn(input_shape=(IMG[0], IMG[1], 3), num_classes=num_classes)
        st.write(model.summary())

        # Callbacks
        checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)

        st.info("Starting training — this will run in the same process as Streamlit. Use smaller epochs for interactive runs.")
        with st.spinner("Training..."):
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=[checkpoint, early_stop],
                verbose=1
            )
        st.success("Training finished. Best model saved to disk (if checkpoint triggered).")
        st.write("Training history (last values):")
        st.write({k: float(v[-1]) if isinstance(v, (list,tuple)) else v for k,v in history.history.items()})

        # load best model
        if os.path.exists(model_file):
            model_loaded = load_model(model_file)
            st.success("Best model loaded for inference.")

# Inference on uploaded image
if uploaded_file:
    st.subheader("Inference")
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded image", use_column_width=True)
        # ensure model is loaded
        if model_loaded is None:
            # try loading default model file
            model_loaded = load_trained_model(model_file)
        if model_loaded is None:
            st.error("No trained model available. Train a model or provide a model file named in the sidebar.")
        else:
            arr = load_and_preprocess_image(img, img_size=IMG)
            x = np.expand_dims(arr, axis=0)
            preds = model_loaded.predict(x)[0]
            # Get class names from training generator if possible
            # Attempt to infer classes from dataset directory structure
            classes = None
            if os.path.isdir(dataset_path):
                classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
            if classes and len(classes)==len(preds):
                top_idx = np.argsort(preds)[::-1][:5]
                st.write("Top predictions:")
                for idx in top_idx:
                    st.write(f"- **{classes[idx]}** — {preds[idx]*100:.2f}%")
            else:
                # fallback: show numeric class probabilities
                top_idx = np.argsort(preds)[::-1][:5]
                st.write("Top probabilities (class index):")
                for idx in top_idx:
                    st.write(f"- class_{idx} — {preds[idx]*100:.2f}%")
    except Exception as e:
        st.error(f"Error processing uploaded image: {e}")

# Show saved model info
st.markdown("---")
st.subheader("Model file info")
if os.path.exists(model_file):
    mtime = os.path.getmtime(model_file)
    st.write(f"Model file `{model_file}` exists (size: {os.path.getsize(model_file)//1024} KB).")
else:
    st.write(f"No model file named `{model_file}` was found in the working directory.")

st.markdown("### Notes & tips")
st.markdown(
    """
- Training inside Streamlit is convenient for demos but not ideal for long runs. For heavier training, use a separate script or notebook.
- Adjust `IMG size`, `Batch size`, and `Epochs` in the sidebar depending on your GPU/CPU memory.
- If your dataset folder name differs, change the `DEFAULT_REPO_PATH` variable at the top of this file.
"""
)
