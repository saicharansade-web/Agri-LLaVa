import streamlit as st
import torch, json, io
from PIL import Image
from torchvision import models, transforms

st.title("🌿 AgriLLaVA Mini")

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    with open("class_labels.json", "r") as f:
        class_names = json.load(f)

    NUM_CLASSES = len(class_names)
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load("models/agri_feature_alignment.pth", map_location="cpu"))
    model.eval()
    return model, class_names

img_model, class_names = load_model()

infer_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ---------- Prediction Function ----------
def predict(image_pil, topk=3):
    img = infer_transform(image_pil).unsqueeze(0)
    with torch.no_grad():
        logits = img_model(img)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
    idxs = probs.argsort()[-topk:][::-1]
    return [(class_names[i], float(probs[i])) for i in idxs]

# ---------- UI ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Live Camera")
    cam = st.camera_input("Take a photo")

with col2:
    st.subheader("📁 Upload Image")
    upload = st.file_uploader("Choose an image", ["jpg","png","jpeg"])

image = None
if cam: image = Image.open(io.BytesIO(cam.getvalue()))
if upload: image = Image.open(upload)

if image:
    st.image(image, width=300)
    preds = predict(image)
    st.write("### 🔍 Predictions:")
    for name, p in preds:
        st.write(f"**{name}** : {p*100:.2f}%")
else:
    st.info("Upload or capture a plant image to analyze 🌿")
