import streamlit as st
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from qrc_model import MobileViT_QRC_U_Net

# --- TTA utilities ---
transform = transforms.ToTensor()

tta_transforms = [
    (lambda img: img,                           lambda m: m),
    (lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),  lambda m: np.fliplr(m)),
    (lambda img: img.rotate(90, expand=True),          lambda m: np.rot90(m, k=3)),
    (lambda img: img.rotate(180, expand=True),         lambda m: np.rot90(m, k=2)),
    (lambda img: img.rotate(270, expand=True),         lambda m: np.rot90(m, k=1)),
]

def tta_predict(model, pil_img, device):
    model.eval()
    preds = []
    for aug, deaug in tta_transforms:
        aug_img = aug(pil_img).resize((256,256))
        inp = transform(aug_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)  # (1,1,256,256)
        prob = out.squeeze().cpu().numpy()  # (256,256)
        prob = deaug(prob)
        # center-crop any expanded rotations back to 256×256
        if prob.shape != (256,256):
            h, w = prob.shape
            sx, sy = (w-256)//2, (h-256)//2
            prob = prob[sy:sy+256, sx:sx+256]
        preds.append(prob)
    return np.stack(preds,0).mean(0)

# --- Segmentation metrics ---
def dice_score(prob_mask, binary_mask, smooth=1e-6):
    pred = (prob_mask > 0.5).astype(np.uint8)
    tgt  = binary_mask
    inter = (pred & tgt).sum()
    union = pred.sum() + tgt.sum()
    return (2*inter + smooth) / (union + smooth)

def iou_score(prob_mask, binary_mask, smooth=1e-6):
    pred = (prob_mask > 0.5).astype(np.uint8)
    tgt  = binary_mask
    inter = (pred & tgt).sum()
    union = pred.sum() + tgt.sum() - inter
    return (inter + smooth) / (union + smooth)

def compute_confidence(pred_prob, binary_mask, threshold=0.8):
    # Mask out the confident regions (where prob > threshold)
    confident_pixels = pred_prob[binary_mask == 1]  # Predicted region (tumor)
    
    # Compute average confidence in the confident region
    if len(confident_pixels) == 0:
        return 0  # No confident region found
    else:
        return confident_pixels.mean() * 100  # Confidence as percentage

# --- Load model once ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileViT_QRC_U_Net(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load("best_mobilevit_qrc_unet.pth", map_location=device))
    return model, device

model, device = load_model()

# --- Streamlit UI ---
st.title("🫁 Lung Cancer CT Scan Segmentation (QRC-UNet)")
st.markdown("Upload a **256×256 grayscale CT scan**. Model inference uses TTA for smoother predictions.")

uploaded = st.file_uploader("Choose CT scan image...", type=["png","jpg","jpeg"])
if uploaded:
    pil_img = Image.open(uploaded).convert("L")
    pil_img = pil_img.resize((256,256))
    img_np = np.array(pil_img) / 255.0

    # Run TTA inference
    avg_prob = tta_predict(model, pil_img, device)
    binary_pred = (avg_prob > 0.5).astype(np.uint8)

    # Compute metrics (against its own thresholded mask)
    dice = dice_score(avg_prob, binary_pred)
    iou  = iou_score (avg_prob, binary_pred)
    confidence = compute_confidence(avg_prob, binary_pred)

    # Build overlay: highlight predicted region at max intensity
    overlay = img_np.copy()
    overlay[binary_pred==1] = 1.0

    # Plot results
    fig, axs = plt.subplots(1,3,figsize=(15,5))
    axs[0].imshow(img_np,    cmap="gray"); axs[0].set_title("Original CT");      axs[0].axis("off")
    axs[1].imshow(overlay,   cmap="gray"); axs[1].set_title("Overlay Prediction");axs[1].axis("off")
    axs[2].imshow(binary_pred,cmap="gray"); axs[2].set_title("Predicted Mask");   axs[2].axis("off")
    st.pyplot(fig)

    # Show metrics
    st.markdown("### 📊 Segmentation Metrics (Self-consistent)")
    st.markdown(f"- **Confidence**: `{confidence:.2f}%`")
    st.markdown(f"- **Dice Score**: `{dice:.4f}`")
    st.markdown(f"- **IoU Score**: `{iou:.4f}`")
