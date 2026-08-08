import streamlit as st
import torch
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.model import MathTransformer
from src.vocab import MathTokenizer
from src.pipeline import ProductionPipeline

# ==========================================
# 1. Page Configuration & Futuristic Amber CSS
# ==========================================
st.set_page_config(
    page_title="Axiom: Neuro-Symbolic LLM",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Deep Sci-Fi Radial Background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #3a1c00 0%, #050200 100%);
    }
    
    header {visibility: hidden;}
    
    /* Sidebar - Deep Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(10, 5, 0, 0.7) !important;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 140, 0, 0.2);
    }
    
    /* User Chat Bubble - Bright Orange Gradient */
    .stChatMessage.user {
        background: linear-gradient(135deg, #FF8C00 0%, #E65C00 100%);
        border-radius: 20px 20px 0px 20px;
        padding: 1rem;
        box-shadow: 0 10px 20px rgba(255, 140, 0, 0.15);
        border: none;
    }
    .stChatMessage.user * { color: #ffffff !important; text-shadow: 0 1px 2px rgba(0,0,0,0.2); }
    
    /* Assistant Chat Bubble - Deep Space with Glowing Amber Border */
    .stChatMessage.assistant {
        background: rgba(20, 10, 0, 0.85);
        border-radius: 20px 20px 20px 0px;
        padding: 1rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 140, 0, 0.3);
    }
    
    /* MathJax Rendering Container - Amber Glow */
    .katex-display {
        background: rgba(0, 0, 0, 0.6);
        padding: 15px;
        border-radius: 12px;
        border-left: 4px solid #FF8C00;
        overflow-x: auto;
        color: #FFDBA6 !important;
    }
    
    /* Inputs and Accents */
    .stChatInputContainer {
        border-radius: 30px;
        border: 1px solid rgba(255, 140, 0, 0.4);
        box-shadow: 0 0 15px rgba(255, 140, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Sidebar Profile (Tech-Focused)
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #FF8C00 !important;'>⚙️ Axiom Diagnostics</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🧠 Architecture Specs:")
    st.markdown("- **Core:** PyTorch Transformer")
    st.markdown("- **Parameters:** Custom Init")
    st.markdown("- **Kernel:** SymPy Symbolic Engine")
    st.markdown("- **State:** Locked for Inference")
    st.markdown("---")
    st.info(
        "🔥 **Neuro-Symbolic Engine**\n\n"
        "This system operates on a dual-pathway. "
        "It utilizes a neural network for intuitive pattern recognition, "
        "and routes it through a deterministic mathematical verifier."
    )

# ==========================================
# 3. Model Initialization
# ==========================================
@st.cache_resource
def load_production_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(base_dir, "weights", "vocab.json")
    model_path_pt = os.path.join(base_dir, "weights", "model.pt")
    model_path_pth = os.path.join(base_dir, "weights", "model.pth")
    
    if os.path.exists(model_path_pt):
        final_model_path = model_path_pt
    elif os.path.exists(model_path_pth):
        final_model_path = model_path_pth
    else:
        return None
        
    if not os.path.exists(vocab_path):
        return None
        
    tokenizer = MathTokenizer()
    tokenizer.load(vocab_path)
    
    model = MathTransformer(vocab_size=tokenizer.vocab_size).to(device)
    state_dict = torch.load(final_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval() 
    
    return ProductionPipeline(model, tokenizer, device)

pipeline = load_production_model()

# ==========================================
# 4. Stream Generator
# ==========================================
def generate_step_by_step(response_string):
    blocks = response_string.split('\n\n')
    for block in blocks:
        yield block + '\n\n'
        time.sleep(0.7) 

# ==========================================
# 5. Web UI Main Logic
# ==========================================
st.markdown("<h1 style='text-align: center; background: -webkit-linear-gradient(#FFB700, #FF5E00); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🔥 Axiom: Neuro-Symbolic Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #FDF4E3;'>100% Deterministic Mathematical Precision.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if pipeline is None:
    st.error("🚨 **System Uninitialized:** Cannot find the neural weights. Please ensure your `.pt` or `.pth` file and `vocab.json` are inside the `weights/` folder.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 **Axiom Online.** Neural weights loaded successfully. Ask *'tell me about yourself'*, or enter a differential equation like `$dy/dx = x * y$`."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Initiate neural query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing neural pathways..."):
            time.sleep(0.8)
        with st.spinner("Executing symbolic verification kernel..."):
            time.sleep(0.8)
            
        raw_response = pipeline.process_query(prompt)
        response_placeholder = st.empty()
        streamed_response = st.write_stream(generate_step_by_step(raw_response))
            
    st.session_state.messages.append({"role": "assistant", "content": streamed_response})