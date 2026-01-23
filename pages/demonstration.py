import streamlit as st
from PIL import Image
from model.inference import extract_embedding, cosine_similarity, load_model
from model.encoder import FeatureExtractionModel
from utils.config import MODELS
from typing import Optional, Tuple

def verify_signatures(img_1: Image.Image, img_2: Image.Image, model: FeatureExtractionModel) -> float:
    emb1 = extract_embedding(img_1, model)
    emb2 = extract_embedding(img_2, model)
    return cosine_similarity(emb1, emb2)

@st.cache_resource
def get_model(ckpt: str) -> FeatureExtractionModel:
    return load_model(ckpt)

@st.cache_resource
def load_example_images() -> dict[str, Image.Image]:
    return {
        "original_11_1": Image.open("sct-signature/static/original_11_1.png"),
        "original_11_4": Image.open("sct-signature/static/original_11_4.png"),
        "forged_11_20": Image.open("sct-signature/static/forgeries_11_20.png"),
        "original_21_1": Image.open("sct-signature/static/original_21_1.png"),
        "original_8_4": Image.open("sct-signature/static/original_8_4.png"),
        "original_8_5": Image.open("sct-signature/static/original_8_5.png"),
        "forged_8_12": Image.open("sct-signature/static/forgeries_8_12.png"),
    }

def display_signature_pair(
    img_1: Image.Image,
    img_2: Image.Image,
    caption_1: str,
    caption_2: str    
) -> None:
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.image(img_1, caption_1, width=384)
    with col2:
        st.image(img_2, caption_2, width=384)

def display_verification_result(score: Optional[float], threshold: float) -> None:
    st.markdown("### Score")
    st.metric("Similarity score", f"{score:.3f}" if score is not None else "None")
    
    if score is not None:
        if score >= threshold:
            st.success("✅ High similarity - signatures are likely from the same writer.")
        else:
            st.warning("❌ Different writers - signatures are likely from different writers.")

def render_sidebar() -> Tuple[str, float, FeatureExtractionModel]:
    st.sidebar.header("Controls")
    st.sidebar.markdown("Augment variables to change how the model performs.")
    
    st.sidebar.subheader("Model Selection")
    st.sidebar.markdown("Each model was trained with different loss functions. This sct-signature allows comparison between models in real time.")
    
    model_name = st.sidebar.selectbox(
        "Select verification model",
        list(MODELS.keys())
    )
    
    st.sidebar.markdown(f"Selected model: **{model_name}**")
    model_info = MODELS[model_name]
    st.sidebar.caption(f" {model_info['description']}")
    
    st.sidebar.subheader("Threshold manipulation")
    st.sidebar.markdown(f"Recommended threshold for **{model_name}**: **{model_info['threshold']}**")
    
    threshold = st.sidebar.slider(
        "Decision Threshold",
        0.0, 1.0, 0.75,
        help="Higher values make verification stricter"
    )
    
    st.sidebar.header("How To Interpret")
    st.sidebar.markdown("""
        The score is calculated with **cosine similarity [-1.0, 1.0]**: 
        
        - Scores **closer to 1.0** are similar 
        - Scores **closer to 0.0** and below are dissimilar
        
        The decision made is based on the selected **threshold**.
    """)
    
    model = get_model(str(model_info["ckpt"]))
    return model_name, threshold, model

def render_verification_section(model: FeatureExtractionModel, threshold: float) -> None:    
    placeholder = Image.new("RGB", (1, 1), (255, 255, 255))

    img_1 = placeholder
    img_2 = placeholder
    
    st.markdown("## Verification")
    
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        img_1_file = st.file_uploader("Upload first signature", type=["png", "jpg"])
        if img_1_file:
            img_1 = Image.open(img_1_file)
            st.image(img_1, caption="Signature 1", width=384)
    
    with col2:
        img_2_file = st.file_uploader("Upload second signature", type=["png", "jpg"])
        if img_2_file:
            img_2 = Image.open(img_2_file)
            st.image(img_2, caption="Signature 2", width=384)
    
    button = st.button("Verify!", "verify_signature", "Click to begin verification", type="primary")
    
    score = None
    if button:
        if img_1_file and img_2_file:
            score = verify_signatures(img_1, img_2, model)
        elif not img_1_file:
            st.error("Please upload the first signature")
        elif not img_2_file:
            st.error("Please upload the second signature")
    
    display_verification_result(score, threshold)

def render_example_pair(
    title: str,
    img_1: Image.Image,
    img_2: Image.Image,
    caption_1: str,
    caption_2: str,
    model: FeatureExtractionModel,
    threshold: float,
    calculate: bool
) -> None:
    """Render a single example signature pair."""
    st.markdown(f"### {title}")
    display_signature_pair(img_1, img_2, caption_1, caption_2)
    
    score = None
    if calculate:
        score = verify_signatures(img_1, img_2, model)
    
    display_verification_result(score, threshold)


def render_examples_section(
    images: dict[str, Image.Image], 
    model: FeatureExtractionModel, 
    threshold: float
) -> None:
    st.markdown("## Examples")
    
    st.info("""
        **Note:** Example signatures shown are from the model's test set, which was 
        held out during training. Performance on completely novel signatures from 
        different sources may vary.
    """)
    
    calculate = st.button("Calculate All Examples!")
    
    render_example_pair(
        "Original - Original Pair",
        images["original_11_1"],
        images["original_11_4"],
        "Original Signature ID 11 #1",
        "Original Signature ID 11 #4",
        model,
        threshold,
        calculate
    )
    
    render_example_pair(
        "Original - Hard Forgery Pair",
        images["original_11_1"],
        images["forged_11_20"],
        "Original Signature ID 11 #1",
        "Forged Signature ID 11 #20",
        model,
        threshold,
        calculate
    )
    
    render_example_pair(
        "Original - Easy Forgery Pair",
        images["original_11_1"],
        images["original_21_1"],
        "Original Signature ID 11 #1",
        "Original Signature ID 21 #1",
        model,
        threshold,
        calculate
    )

def render_error_analysis_section(images: dict[str, Image.Image], threshold: float) -> None:
    st.markdown("## Error Analysis")
    
    st.info("The following section uses a model trained with $L_{SC+}$")
    lsc_model = get_model(str(MODELS["SCT+"]["ckpt"]))
    
    st.markdown("""
        As humans, we can perceive signatures holistically; we instantly 
        notice differences in overall style, flow, and character formation. 
        Models, however, work differently: they extract numerical features from the images 
        (patterns of strokes, curves, and spatial relationships). 
        
        In the case of **false positives**, if a forged signature happens to share many of 
        these learned features with a genuine one, the model may incorrectly classify them 
        as matching, even when the forgery appears obviously different to the human eye. 

        Additionally, the model may exhibit counterintuitive behavior where it successfully 
        differentiates difficult forgeries of one type yet fails at less difficult ones from 
        another. This is because the model's decisions are based on patterns learned from its 
        training data, whereby a lower volume of certain forgery types may not provide 
        sufficient statistical information to the model. 
        
        The same reasoning can be applied to the occurrence of **false negatives**, where 
        extreme differences between the query signature and the reference signature cause 
        the model to misclassify it as a forgery. In this case, however, the failure stems 
        not from the model but from natural variation - the signer may have produced a 
        signature that differs significantly from their reference due to inconsistent signing.
        
        See examples below:
    """)
    
    st.markdown("### False Positive")
    display_signature_pair(
        images["original_8_4"],
        images["forged_8_12"],
        "Original Signature ID 8 #4",
        "Forged Signature ID 8 #12"
    )
    false_positive_score = verify_signatures(images["original_8_4"], images["forged_8_12"], lsc_model)
    display_verification_result(false_positive_score, threshold)
    
    st.markdown("### False Negative")
    display_signature_pair(
        images["original_8_4"],
        images["original_8_5"],
        "Original Signature ID 8 #4",
        "Original Signature ID 8 #5"
    )
    false_negative_score = verify_signatures(images["original_8_4"], images["original_8_5"], lsc_model)
    display_verification_result(false_negative_score, threshold)

def main():
    st.title("Offline Signature Verification")
    st.warning("This sct-signature does not store your signatures!")
    st.markdown("Compare two signatures and determine whether they belong to the same writer")
    
    # Load resources
    images = load_example_images()
    _model_name, threshold, model = render_sidebar()
    
    # Render sections
    render_verification_section(model, threshold)
    render_examples_section(images, model, threshold)
    render_error_analysis_section(images, threshold)

if __name__ == "__main__":
    main()