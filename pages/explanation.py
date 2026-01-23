import streamlit as st

@st.cache_data
def create_anchor(text: str) -> str:
    anchor_id = text.lower().replace(" ", "-").replace("#", "")
    st.markdown(f'<a name="{anchor_id}"></a>', unsafe_allow_html=True)
    st.header(text)
    return anchor_id

st.sidebar.title("Table of Contents")
st.sidebar.markdown("[Go to Project Overview](#project-overview)")
st.sidebar.markdown("[Go to Approach](#approach)")
st.sidebar.markdown("[Go to Model Architecture](#model-architectrue)")
st.sidebar.markdown("[Go to Data & Training](#data-and-training)")
st.sidebar.markdown("[Go to Evaluation](#evaluation)")
st.sidebar.markdown("[Go to Key Design Decisions](#key-design-decisions)")
st.sidebar.markdown("[Go to Limitations & Future Works](#limitations-and-future-works)")



st.title("How It Works")

st.markdown("""
            
    > [Source Code](https://github.com/HappyPotatoHead/signature-verification-sct-plus)
    >
    > [Detailed Explanation](https://potatogarden.surge.sh/AI--and--Deep-Learning/Offline-Signature-Verification)
    
    ## Project Overviews
    
    Offline signature verification aims to determine whether two handwritten signatures belong to the same individual.
    This is challenging due to **high intra-class variability** and **skilled forgeries**. This project in particular aims to **maintain
    intra-class generalisability** while **maximising distance from skilled forgeries**.  
    
    ## Approach
    
    This project utilises deep metric learning approach to learn a writer-invariant embedding space. 
    Signatures from the same writer are mapped close together, while forgeries are pushed further apart.
    This project utilises [EfficientNetV2](https://arxiv.org/abs/2104.00298) as the feature extraction backbone and
    the custom loss function is inspired from [Hard negative examples are hard, but useful](https://arxiv.org/abs/2007.12749).
    
""")

st.text(" ")

left_column, centre_column, right_column = st.columns([1,2,1])
with centre_column:
    st.image("static/ideal_triplet_mining.png", "Embedding space", "stretch" )

st.markdown("""
    ## Model Architecture
    
    - CNN backbone ([EfficientNetV2](https://arxiv.org/abs/2104.00298)) for feature extraction
    - Custom loss function inspired from [Hard negative examples are hard, but useful](https://arxiv.org/abs/2007.12749), $L_{SC+}$
    - Projection head with gradual dimensionality reduction
    - L2-normalised embeddings
    - Cosine similarity for comparison
    
    _[View specifics about backbone here](https://potatogarden.surge.sh/AI--and--Deep-Learning/Offline-Signature-Verification#backbone)_
    
    
    ## Data & Training
    
    - Dataset: CEDAR offline signature dataset
    - ID-based train/test split
    - Transfer learning from ImageNetV1 weights
    - Linear warm-up with cosine decay learning
    - Extended PK sampling
    
    ## Evaluation
    
    - ROC-AUC used as the primary metric
    - Performance evaluated across multiple thresholds
    - Confusion matrix 
    - Histogram visualisation
""")

st.text(" ")

left_column, centre_column, right_column = st.columns([1,2,1])
with centre_column:
    st.image("static/lsc+_roc.png", "Embedding space", "stretch" )


st.markdown("""
                
    ## Key Design Decisions
    
    - Metric learning over classification to support unseen writers
    - Cosine similarity for scale-invariant comparison
    - Partial freezing of the backbone to reduce overfitting
    
    ## Limitations & Future Works
    
    - Limited availability of large, diverse, realistic signature datasets
    - Performance degrades for highly skilled forgeries
    - Future work includes:
        - Expansion of signature datasets
        - Script-aware routing
        - Multilingual datasets 
""")

