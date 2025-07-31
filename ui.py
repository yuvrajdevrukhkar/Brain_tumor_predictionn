

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# --------------------
# Config
IMG_SIZE = 224
NUM_CLASSES = 4
MODEL_PATH = "resnet_transfer.pth"


class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Load Trained Model
@st.cache_resource
def load_model():
    resnet = models.resnet18(pretrained=False)  # pretrained=False during loading

    # Replace the classifier head (same as training)
    in_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, NUM_CLASSES)
    )

    # Load trained weights
    resnet.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    resnet.to(device)
    resnet.eval()
    return resnet

model = load_model()

# --------------------
# Image Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --------------------
# Streamlit UI
st.title("🧠 Brain Tumor Prediction")
st.write("Upload an MRI image to classify the tumor type:")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)
        pred_label = class_names[pred_class]

    # Display
    st.markdown(f"### 🧾 Prediction: `{pred_label.upper()}`")
    st.write("### 📊 Probabilities:")
    for i, prob in enumerate(probs):
        st.write(f"- **{class_names[i]}**: `{prob:.2f}`")
