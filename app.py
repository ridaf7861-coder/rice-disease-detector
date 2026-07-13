
import streamlit as st
import torch, numpy as np
from PIL import Image
from torchvision import transforms
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

st.set_page_config(page_title="Rice Disease Detector", page_icon="🌾")

classes = ['bacterial_leaf_blight','bacterial_leaf_streak','bacterial_panicle_blight',
           'blast','brown_spot','dead_heart','downy_mildew','hispa','normal','tungro']

TREATMENT = {
    "bacterial_leaf_blight":"Bacterial leaf blight. Use certified disease-free seed, avoid excess nitrogen, ensure drainage.",
    "blast":"Rice blast (fungal). Apply recommended fungicides (e.g. tricyclazole), avoid dense planting.",
    "brown_spot":"Brown spot (fungal). Correct potassium/silicon deficiency; use fungicide seed treatment.",
    "bacterial_leaf_streak":"Bacterial leaf streak. Use resistant varieties and clean seed; avoid leaf injury.",
    "bacterial_panicle_blight":"Bacterial panicle blight. Use tolerant varieties; avoid high nitrogen and heat stress.",
    "dead_heart":"Dead heart - often stem borer. Inspect for larvae; use pheromone traps or insecticides.",
    "downy_mildew":"Downy mildew. Improve drainage, remove infected plants, apply fungicide.",
    "hispa":"Rice hispa (insect). Remove affected leaf tips; avoid excess nitrogen.",
    "normal":"Healthy leaf - no disease detected. Maintain good field practices.",
    "tungro":"Tungro (viral, leafhopper-spread). Control vectors; use resistant varieties.",
}

@st.cache_resource
def load_model():
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(classes))
    m.load_state_dict(torch.load("best_model_adapted.pth", map_location="cpu"))
    m.eval()
    return m

model = load_model()
norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
eval_tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), norm])

st.title("🌾 Rice Disease Detector")
st.write("Upload a photo of a rice leaf. The model predicts the disease, shows where it focused (Grad-CAM), and gives treatment guidance.")

uploaded = st.file_uploader("Upload a rice leaf photo", type=["jpg","jpeg","png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)
    col1.image(image, caption="Input", use_container_width=True)

    arr = np.array(image.resize((224,224)))/255.0
    tensor = eval_tf(image).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor); probs = torch.softmax(out,1)[0]
        pred = out.argmax().item(); conf = probs[pred].item()

    cam = GradCAM(model=model, target_layers=[model.blocks[-1]])
    gray = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(pred)])[0]
    overlay = show_cam_on_image(arr.astype(np.float32), gray, use_rgb=True)
    col2.image(overlay, caption="Where the model looked (Grad-CAM)", use_container_width=True)

    st.subheader(f"Diagnosis: {classes[pred]}")
    st.write(f"**Confidence:** {conf:.0%}")
    st.info(TREATMENT.get(classes[pred], ""))
