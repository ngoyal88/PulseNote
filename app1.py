import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
try:
    import fitz  # PyMuPDF
except ImportError:
    st.error("PyMuPDF not installed. Run: pip install PyMuPDF")
    st.stop()
import io

# ------------------------------
# CONFIG
# ------------------------------
MODEL_PATH = "models/soap_generator"  # your trained model folder

@st.cache_resource
def load_model():
    """Load tokenizer + model once and cache them."""
    try:
        # Force PyTorch backend only
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            use_fast=True,
            local_files_only=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.float32
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        return tokenizer, model, device
    except Exception as e:
        st.error(f"❌ Failed to load model from {MODEL_PATH}")
        st.error(f"Error details: {str(e)}")
        st.info("💡 Make sure:")
        st.info("1. Model files exist in models/soap_generator/")
        st.info("2. PyTorch is installed (not TensorFlow)")
        st.info("3. Run: pip uninstall tensorflow tensorflow-gpu")
        st.stop()


tokenizer, model, device = load_model()

# ------------------------------
# OCR SECTION
# ------------------------------

def extract_text_from_pdf(uploaded_file):
    try:
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        return text
    except Exception as e:
        return f"PDF Extraction Error: {str(e)}"

def extract_text_from_image(uploaded_file):
    try:
        import pytesseract
        img = Image.open(uploaded_file)
        return pytesseract.image_to_string(img)
    except ImportError:
        return "❌ Tesseract not installed. Install it from: https://github.com/UB-Mannheim/tesseract/wiki"
    except Exception as e:
        return f"OCR Error: {str(e)}"


# ------------------------------
# Inference Function
# ------------------------------

def parse_soap_sections(soap_text):
    """Parse raw SOAP text into structured sections."""
    sections = {
        "Subjective": "",
        "Objective": "",
        "Assessment": "",
        "Plan": ""
    }
    
    # First, try to split by section markers on the same line
    text = soap_text.strip()
    
    # Use regex to find section boundaries
    import re
    
    # Look for patterns like "SUBJECTIVE:", "OBJECTIVE:", etc.
    patterns = [
        (r'SUBJECTIVE:\s*', 'Subjective'),
        (r'OBJECTIVE:\s*', 'Objective'),
        (r'ASSESSMENT:\s*', 'Assessment'),
        (r'PLAN:\s*', 'Plan'),
        (r'S:\s*', 'Subjective'),
        (r'O:\s*', 'Objective'),
        (r'A:\s*', 'Assessment'),
        (r'P:\s*', 'Plan')
    ]
    
    # Find all section positions
    matches = []
    for pattern, section_name in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            matches.append((match.start(), match.end(), section_name))
    
    # Sort by position
    matches.sort(key=lambda x: x[0])
    
    # Extract content between sections
    for i, (start, end, section_name) in enumerate(matches):
        # Get content until next section or end of text
        if i + 1 < len(matches):
            content_end = matches[i + 1][0]
        else:
            content_end = len(text)
        
        content = text[end:content_end].strip()
        
        # Only update if this section hasn't been filled or is empty
        if not sections[section_name] or len(content) > len(sections[section_name]):
            sections[section_name] = content
    
    return sections

def format_soap_output(sections):
    """Format SOAP sections into readable markdown."""
    formatted = ""
    
    icons = {
        "Subjective": "🗣️",
        "Objective": "🔬",
        "Assessment": "🩺",
        "Plan": "📋"
    }
    
    for section, content in sections.items():
        if content.strip():
            formatted += f"### {icons.get(section, '📌')} **{section}**\n\n"
            formatted += f"{content}\n\n---\n\n"
    
    return formatted if formatted else "No structured output generated."

def generate_soap(text):
    """Generate SOAP note using your fine-tuned T5 model."""

    prompt = "Generate SOAP note: " + text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=4,
        early_stopping=True
    )

    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse and structure the output
    sections = parse_soap_sections(raw_output)
    formatted = format_soap_output(sections)
    
    return formatted, sections


# ------------------------------
# STREAMLIT UI
# ------------------------------
st.set_page_config(
    page_title="MediCode – SOAP Generator",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 MediCode – Clinical SOAP Note Generator")
st.write("A fully local, privacy-friendly medical summarizer.")


# ------------------------------
# SIDEBAR — UPLOADS
# ------------------------------

st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF or Image",
    type=["pdf", "png", "jpg", "jpeg"]
)

extracted_text = ""

if uploaded_file:
    st.sidebar.success("File uploaded!")

    if uploaded_file.type == "application/pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
    else:
        extracted_text = extract_text_from_image(uploaded_file)

    st.sidebar.write("Extracted Text:")
    st.sidebar.text_area(" ", extracted_text, height=200)


# ------------------------------
# MAIN UI — INPUT + OUTPUT
# ------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Clinical Note (Input)")
    input_text = st.text_area(
        "Paste or edit clinical text:",
        value=extracted_text,
        height=400
    )

    run_button = st.button("Generate SOAP Note", type="primary")

with col2:
    st.subheader("📋 SOAP Note (Output)")
    output_box = st.empty()


# ------------------------------
# RUN GENERATION
# ------------------------------

if run_button:
    if not input_text.strip():
        st.warning("Please enter or upload text first.")
    else:
        with st.spinner("Generating SOAP note..."):
            formatted_soap, sections = generate_soap(input_text)

            # Display result
            output_box.markdown(f"""
## 🩺 **Generated SOAP Note**

{formatted_soap}
""")

        # Create downloadable text
        download_text = ""
        for section, content in sections.items():
            if content.strip():
                download_text += f"{section.upper()}\n{'-' * 50}\n{content}\n\n"
        
        # Download button
        st.download_button(
            "⬇️ Download SOAP Note",
            data=download_text,
            file_name="soap_note.txt",
            mime="text/plain"
        )