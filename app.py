import os
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Dict, Tuple
from transformers import (
    AutoModel,
    AutoImageProcessor,
    AutoModelForImageClassification,
)
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Deepfake Detector", page_icon="🖼️", layout="centered")


@st.cache_resource
def load_model_from_hub(
    model_id: str,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[Optional[object], Optional[object], Optional[torch.device], Dict[int, str]]:
    """
    Load processor + model directly from the Hugging Face Hub using transformers.from_pretrained.
    Tries to load an image-classification model (AutoModelForImageClassification). If that fails,
    falls back to AutoModel and returns a clear error because AutoModel does not produce logits.
    Returns (model, processor, device, id2label).
    """
    if not model_id:
        st.error("HF_MODEL_ID not set. Set HF_MODEL_ID environment variable to your model id.")
        return None, None, None, {}

    try:
        processor = AutoImageProcessor.from_pretrained(
            model_id, use_auth_token=token, cache_dir=cache_dir
        )
    except Exception as e:
        st.error(f"Failed to load processor from Hugging Face Hub ('{model_id}'): {e}")
        return None, None, None, {}

    # Prefer to load an image-classification model (provides logits)
    try:
        model = AutoModelForImageClassification.from_pretrained(
            model_id, use_auth_token=token, cache_dir=cache_dir
        )
    except Exception:
        # Fallback: try generic AutoModel (may not have logits / classification head)
        try:
            model = AutoModel.from_pretrained(model_id, use_auth_token=token, cache_dir=cache_dir)
            st.warning(
                "Loaded model with AutoModel (no classification head detected). "
                "This repo may not be an image-classification model. The app expects a "
                "transformers image-classification model (AutoModelForImageClassification)."
            )
        except Exception as e:
            st.error(f"Failed to load model from Hugging Face Hub ('{model_id}'): {e}")
            return None, None, None, {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # read id2label if present; otherwise leave empty mapping
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

    # If model does not output logits (e.g. AutoModel), abort with clear error
    if not hasattr(model, "forward") or not hasattr(model, "eval"):
        raise RuntimeError("Loaded model is not callable. Ensure it's a transformers model.")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # Most classification models expose `logits`
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError(
                "Model output does not contain logits. Make sure the Hugging Face repo is an image-classification model "
                "or use a model that inherits from AutoModelForImageClassification."
            )
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
    st.title("🖼️ Deepfake vs Real — Image Detector")

    st.markdown(
        "This app loads an image-classification model from the Hugging Face Hub. "
        "Set HF_MODEL_ID (and HF_TOKEN for private models) as environment variables."
    )

    MODEL_ID = os.getenv("HF_MODEL_ID") or "sampujit/DeepFake"
    HF_TOKEN = os.getenv("HF_TOKEN") or None
    CACHE_DIR = os.getenv("HF_HOME") or None

    st.write(f"Model: **{MODEL_ID}**")
    if st.button("Reload model"):
        try:
            load_model_from_hub.clear()
        except Exception:
            pass
        st.experimental_rerun()

    with st.spinner(f"Loading model '{MODEL_ID}' from Hugging Face Hub..."):
        model, processor, device, id2label = load_model_from_hub(MODEL_ID, token=HF_TOKEN, cache_dir=CACHE_DIR)

    if model is None or processor is None:
        st.error("Model failed to load. Ensure HF_MODEL_ID (and HF_TOKEN if needed) are set correctly.")
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
        try:
            with st.spinner("Running inference..."):
                pred_label, confidence, scores = predict_image(model, processor, device, id2label, image)
        except Exception as e:
            st.error(str(e))
            return

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
        "Notes: The app loads the model directly from the Hugging Face Hub via transformers.from_pretrained. "
        "If the model is private, set HF_TOKEN. Ensure the repository is an image-classification model."
    )


if __name__ == "__main__":
    main()