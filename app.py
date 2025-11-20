import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
from PIL import Image
try:
    import fitz  # PyMuPDF
except ImportError:
    st.error("PyMuPDF not installed. Run: pip install PyMuPDF")
    st.stop()
import io

try:
    import google.generativeai as genai
except ImportError:
    st.error("AI library not installed. Run: pip install google-generativeai")
    st.stop()

# ------------------------------
# CONFIG
# ------------------------------

def load_model():
    """Load the SOAP generation model."""
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Store in session state for OCR access
    if 'model' not in st.session_state:
        st.session_state['model'] = model
    
    return model

model = load_model()

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
        # Use the main model for OCR
        img = Image.open(uploaded_file)
        
        # Convert image to bytes for Gemini
        import io
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img.format or 'PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Use Gemini for OCR
        response = model.generate_content([
            "Extract all text from this medical document. Preserve the original structure, formatting, and layout as much as possible. Return only the extracted text without any commentary or explanations.",
            {"mime_type": "image/png", "data": img_byte_arr}
        ])
        
        text = response.text
        return text if text.strip() else "No text detected in image."
        
    except Exception as e:
        # Fallback to EasyOCR with preprocessing
        try:
            import easyocr
            import numpy as np
            import cv2
            
            # Initialize reader (cached after first use)
            if 'ocr_reader' not in st.session_state:
                with st.spinner("Loading OCR model..."):
                    st.session_state.ocr_reader = easyocr.Reader(['en'], gpu=False)
            
            reader = st.session_state.ocr_reader
            
            # Reset file pointer
            uploaded_file.seek(0)
            img = Image.open(uploaded_file)
            
            # Convert to grayscale and apply preprocessing
            img_array = np.array(img)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply thresholding to improve OCR
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Perform OCR
            result = reader.readtext(thresh, detail=0, paragraph=True)
            text = '\n'.join(result)
            
            return text if text.strip() else "No text detected in image."
            
        except Exception as fallback_error:
            return f"OCR Error: Could not extract text from image. {str(fallback_error)}"


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
    """Generate SOAP note using trained neural network model."""
    
    # Create medical SOAP prompt
    prompt = f"""You are a medical assistant. Convert the following clinical note into a structured SOAP note format.

Clinical Note:
{text}

Please provide a short SOAP note with these exact section headers:
SUBJECTIVE: Patient's symptoms, complaints, and history
OBJECTIVE: Physical exam findings, vital signs, test results  
ASSESSMENT: Diagnosis or clinical impression
PLAN: Treatment plan, medications, follow-up

Format your response with each section clearly labeled and separated."""

    try:
        response = model.generate_content(prompt)
        raw_output = response.text
        
        # Parse and structure the output
        sections = parse_soap_sections(raw_output)
        formatted = format_soap_output(sections)
        
        return formatted, sections
    except Exception as e:
        error_msg = f"Model Error: {type(e).__name__}: {str(e)}"
        st.error(error_msg)
        
        # Show helpful tips based on error type
        if "quota" in str(e).lower():
            st.warning("⚠️ Model usage limit reached - Try again later")
        elif "blocked" in str(e).lower() or "safety" in str(e).lower():
            st.warning("🚫 Content was filtered - Try rephrasing the input")
        else:
            st.warning("⚠️ Model processing error - Please try again")
        
        return f"❌ Failed to generate SOAP note.\n\n{error_msg}", {}


# ------------------------------
# STREAMLIT UI
# ------------------------------
st.set_page_config(
    page_title="PulseNote – SOAP Generator",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 PulseNote – Clinical SOAP Note Generator")
st.write("Advanced medical note processing system.")


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

