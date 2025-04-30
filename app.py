
import streamlit as st
from PIL import Image
import torch
import numpy as np
from model import QRC_UNet
from utils import preprocess_image, postprocess_mask

st.set_page_config(page_title="Lung CT Scan Segmentation", layout="wide")
st.title("🧬 Lung Nodule Segmentation (QRC-U-Net)")

@st.cache_resource
def load_model():
    model = QRC_UNet()
    model.load_state_dict(torch.load("qrc_unet_trained.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload a lung CT image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original CT Image", use_column_width=True)

    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output)
        binary_mask = postprocess_mask(pred_mask)

    st.subheader("🩻 Predicted Segmentation Mask")
    st.image(binary_mask, caption="Binary Segmentation Mask", use_column_width=True)

    overlay = np.array(image.resize((256, 256))).copy()
    overlay[:, :, 1] = np.maximum(overlay[:, :, 1], binary_mask)
    st.subheader("📊 Overlay Visualization")
    st.image(overlay, caption="Image + Segmentation Mask", use_column_width=True)

    confidence = pred_mask.mean().item()
    st.info(f"🧠 Confidence Score: {confidence:.2f}")
