import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from io import BytesIO


from model.improved_unet import improved_unet

# 1) Model loading function
@st.cache_data
def load_model(model_name):
    
    if model_name == "improved_unet":
        model = improved_unet(num_classes=1, input_channels=1)
        ckpt = torch.load("./model/test_weights_att.pth", map_location="cpu")
        state = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(state)
    elif model_name == "improved_unet_test1":
        model = improved_unet(num_classes=1, input_channels=1)
        ckpt = torch.load("./model/test_weights_att.pth", map_location="cpu")
        state = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(state)
    elif model_name == "improved_unet_test2":
        model = improved_unet(num_classes=1, input_channels=1)
        ckpt = torch.load("./model/test_weights_att.pth", map_location="cpu")
        state = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(state)
    elif model_name == "improved_unet_test3":
        model = improved_unet(num_classes=1, input_channels=1)
        ckpt = torch.load("./model/test_weights.pth", map_location="cpu")
        state = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(state)
    # More models can be added here
    model.eval()
    return model

# 2) Segmentation inference function
def run_segmentation(model, pil_img):
    # Convert to tensor with shape (1, 1, H, W)
    img = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img)[None, None, :, :]
    with torch.no_grad():
        out = model(tensor)
        if isinstance(out, tuple):
            out = out[-1]
        out = F.interpolate(out, size=tensor.shape[2:], mode="bilinear", align_corners=False)
        prob = out.squeeze().cpu().numpy()
    # Binarize
    mask = (prob > 0.5).astype(np.uint8) * 255
    return prob, mask

# 3) Convert mask to a downloadable byte stream
def get_image_download_bytes(img_array, fmt="PNG"):
    buf = BytesIO()
    Image.fromarray(img_array).save(buf, format=fmt)
    byte_data = buf.getvalue()
    return byte_data

# -------- Streamlit UI --------
st.title("MRI Liver Segmentation Demo")

# Model selection
model_name = st.sidebar.selectbox(
    "Select model",
    ["improved_unet", "improved_unet_test1", "improved_unet_test2", "improved_unet_test3"]
)
model = load_model(model_name)

# Image upload
uploaded = st.file_uploader("Upload a medical image", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded)

    # Column layout
    col1, col2 = st.columns(2)

    # Display the original image
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)

    if st.button("Run Segmentation"):
        prob_map, mask = run_segmentation(model, img)

        # Overlay visualization
        overlay = np.array(img.convert("RGB"))
        overlay[mask == 255] = [255, 0, 0]  # Highlight in red

        # Display the overlay image
        with col2:
            st.image(overlay, caption="Overlay Image", use_container_width=True)

        # Download button
        overlay_bytes = get_image_download_bytes(overlay)
        st.download_button(
            label="Download Overlay Image",
            data=overlay_bytes,
            file_name="overlay.png",
            mime="image/png"
        )