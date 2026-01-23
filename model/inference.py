import torch
import torch.nn.functional as F
import streamlit as st

from huggingface_hub import hf_hub_download
from utils.config import HF_REPO_ID

from pathlib import Path
from PIL import Image
import numpy as np

from model.encoder import FeatureExtractionModel

from utils.config import CLAHEConfig, BlurConfig, ThresholdConfig
from utils.preprocessing import get_test_transform, preprocess_image_array


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = Path("sct-signature/model/25_auc_0.1051.pt")


@st.cache_resource  # type: ignore
def load_model(ckpt_path: str) -> FeatureExtractionModel:
    
    cached_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=ckpt_path
    )
    
    model = FeatureExtractionModel(
        "efficientnet_v2_m", 
        256, 
        None
    )
    
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.to(DEVICE)
    model.eval()
    return model

CLAHE = CLAHEConfig()
BLUR = BlurConfig()
THRESHOLD = ThresholdConfig()

def extract_embedding(pil_image: Image.Image, model: FeatureExtractionModel) -> torch.Tensor:
    # model = load_model()
    
    image_np = np.array(pil_image)
    
    image_np = preprocess_image_array(
        image_np,
        clahe_params=CLAHE,
        blur_params=BLUR,
        threshold_params=THRESHOLD
    )
    
    transform = get_test_transform()
    tensor = transform(Image.fromarray(image_np)) # pyright: ignore[reportUnknownVariableType]
    tensor = tensor.unsqueeze(0).to(DEVICE) # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportAttributeAccessIssue]
    
    with torch.no_grad():
        embedding = model(tensor)
    
    embedding = F.normalize(embedding, p=2, dim=1) 
    
    return embedding.squeeze(0)

def cosine_similarity(
    emb1: torch.Tensor,
    emb2: torch.Tensor
) -> float:
    return F.cosine_similarity(emb1, emb2, dim=0).item()
