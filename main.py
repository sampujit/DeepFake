import os
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Dict, Tuple
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.set_page_config(page_title="Deepfake Detector", page_icon="🖼️", layout="centered")


@st.cache_resource
def load_model_from_local(
    model_path: str,
) -> Tuple[Optional[object], Optional[object], Optional[torch.device], Dict[int, str]]:
    """
    Load processor + model from a local folder (default: ./deepfake_vs_real_image_detection).
    Returns (model, processor, device, id2label).
    """
    if not model_path:
        st.error("Local model path not provided.")
        return None, None, None, {}

    # Resolve relative path to the app directory
    if not os.path.isabs(model_path):
        base = os.path.dirname(__file__)
        model_path = os.path.normpath(os.path.join(base, model_path))

    if not os.path.isdir(model_path):
        st.error(f"Model folder not found: {model_path}")
        return None, None, None, {}

    try:
        processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModelForImageClassification.from_pretrained(model_path)
    except Exception as e:
        st.error(f"Failed to load model/processor from local folder '{model_path}': {e}")
        return None, None, None, {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id2label_raw = getattr(model.config, "id2label", None) or {}
    id2label: Dict[int, str] = {}
    try:
        id2label = {int(k): v for k, v in id2label_raw.items()}
    except Exception:
        for k, v in id2label_raw.items():
            try:
                id2label[int(k)] = v
            except Exception:
                pass
        if not id2label and isinstance(id2label_raw, dict):
            id2label = {i: lbl for i, lbl in enumerate(id2label_raw.values())}

    return model, processor, device, id2label


def predict_image(
    model,
    processor,
    device: torch.device,
    id2label: Dict[int, str],
    image: Image.Image,
) -> Tuple[str, float, Dict[str, float]]:
    """Run the model on a PIL image and return prediction, confidence, and all scores."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape (1, num_labels)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    num_labels = probs.shape[0]
    if id2label and len(id2label) >= num_labels:
        labels = [id2label.get(i, str(i)) for i in range(num_labels)]
    else:
        labels = ["real", "fake"][:num_labels]

    top_idx = int(probs.argmax())
    pred_label = labels[top_idx]
    confidence = float(probs[top_idx])
    scores = {labels[i]: float(probs[i]) for i in range(num_labels)}

    return pred_label, confidence, scores


def main():
    st.title("🖼️ Deepfake vs Real — Image Detector (Local Model)")

    st.markdown(
        "This app loads an image-classification model from a local folder in the same directory."
    )

    # Default local model folder (user-specified)
    MODEL_DIR = os.getenv("MODEL_DIR") or os.getenv("LOCAL_MODEL_PATH") or "deepfake_vs_real_image_detection"

    st.write(f"Local model folder: **{MODEL_DIR}**")
    if st.button("Reload model"):
        try:
            # clear cached resource
            load_model_from_local.clear()
        except Exception:
            pass
        st.experimental_rerun()

    with st.spinner(f"Loading local model from '{MODEL_DIR}'..."):
        model, processor, device, id2label = load_model_from_local(MODEL_DIR)

    if model is None or processor is None:
        st.error("Model failed to load from local folder. Ensure the folder contains a Transformers image-classification model.")
        st.stop()

    st.success(f"Model loaded — running on {device.type.upper()}")

    uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if not uploaded_file:
        st.info("Upload an image to analyze.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Failed to open image: {e}")
        return

    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Analyze"):
        with st.spinner("Running inference..."):
            pred_label, confidence, scores = predict_image(model, processor, device, id2label, image)

        st.metric("Prediction", f"{pred_label} ({confidence * 100:.2f}%)")
        st.markdown("**Scores**")
        for lbl, sc in sorted(scores.items(), key=lambda x: -x[1]):
            st.write(f"- {lbl}: {sc*100:.2f}%")

        if pred_label.lower() in ("fake", "deepfake", "manipulated"):
            st.warning("Model indicates this image may be a deepfake.")
        else:
            st.info("Model indicates this image may be real.")

    st.markdown("---")
    st.caption(
        "Notes: Place a Transformers image-classification model (config.json, pytorch_model.bin / weights) "
        "inside the './deepfake_vs_real_image_detection' folder or set MODEL_DIR env var to another local folder."
    )


if __name__ == "__main__":
    main()