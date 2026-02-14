# app.py - FIXED VERSION
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import pdf2image
import io
import re
import os
from datetime import datetime
import plotly.express as px
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

# Configure poppler path
if os.path.exists("/usr/bin/pdftoppm"):
    os.environ["PATH"] += os.pathsep + "/usr/bin"
elif os.path.exists("/usr/local/bin/pdftoppm"):
    os.environ["PATH"] += os.pathsep + "/usr/local/bin"

# Import Reference Data
from medical_reference import REFERENCE_RANGES, TEST_CATEGORIES, CRITICAL_VALUES

# Import Anthropic
try:
    import anthropic
except ImportError:
    anthropic = None

# Page Configuration
st.set_page_config(
    page_title="MedLab AI - Longitudinal Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- RAG INITIALIZATION ---
@st.cache_resource
def get_rag_system():
    try:
        from rag_components import MedLabRAG
        return MedLabRAG()
    except Exception as e:
        print(f"RAG init failed: {e}")
        return None

rag_system = get_rag_system()

# --- CSS ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1e3a8a; }
    .status-ok { color: #059669; font-weight: bold; }
    .status-err { color: #dc2626; font-weight: bold; }
    .rag-badge { 
        padding: 5px 10px; 
        border-radius: 15px; 
        font-size: 0.8rem; 
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
    .rag-active { background-color: #d1fae5; color: #065f46; border: 1px solid #34d399; }
    .rag-inactive { background-color: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def extract_text(file_bytes, file_type):
    """Extract text from PDF/Image"""
    try:
        if file_type == "application/pdf":
            try:
                images = pdf2image.convert_from_bytes(file_bytes)
                return "\n".join([pytesseract.image_to_string(img) for img in images])
            except Exception as e:
                return f"Error (Poppler): {e}"
        else:
            return pytesseract.image_to_string(Image.open(io.BytesIO(file_bytes)))
    except Exception as e:
        return f"Error: {e}"

def parse_lab_values(text):
    """Regex extraction"""
    data = {}
    patterns = {
        'Hemoglobin': r'(?:Hemoglobin|Hb|HGB)[\s:]*(\d+\.?\d*)',
        'RBC': r'(?:RBC|Red Blood Cell)[\s:]*(\d+\.?\d*)',
        'WBC': r'(?:WBC|White Blood Cell)[\s:]*(\d+\.?\d*)',
        'Platelets': r'(?:Platelets|PLT)[\s:]*(\d{3,})',
        'Glucose_Fasting': r'(?:Fasting Glucose|FBS)[\s:]*(\d+\.?\d*)',
        'HbA1c': r'(?:HbA1c|Glycated)[\s:]*(\d+\.?\d*)',
        'Creatinine': r'(?:Creatinine)[\s:]*(\d+\.?\d*)',
        'TSH': r'(?:TSH)[\s:]*(\d+\.?\d*)',
        'Total_Cholesterol': r'(?:Total Cholesterol)[\s:]*(\d+\.?\d*)',
        'LDL': r'(?:LDL)[\s:]*(\d+\.?\d*)',
        'HDL': r'(?:HDL)[\s:]*(\d+\.?\d*)',
        'Triglycerides': r'(?:Triglycerides)[\s:]*(\d+\.?\d*)',
        'ALT': r'(?:ALT)[\s:]*(\d+\.?\d*)',
        'AST': r'(?:AST)[\s:]*(\d+\.?\d*)',
    }
    for k, v in patterns.items():
        m = re.findall(v, text, re.IGNORECASE)
        if m:
            try:
                data[k] = float(m[0].replace('<','').replace('>','').strip())
            except: pass
    return data

def get_claude_analysis(api_key, df, gender, age, rag_context):
    """Get AI Analysis"""
    if not api_key or not anthropic:
        return "⚠️ Claude API Key missing or library not installed."
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""
        Analyze these lab results for a {age}yo {gender}.
        
        DATA:
        {df.to_csv()}
        
        MEDICAL CONTEXT (RAG):
        {rag_context}
        
        Provide:
        1. Trend Analysis (Improving/Worsening)
        2. Critical Alerts
        3. Recommendations
        """
        
        msg = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
    except Exception as e:
        return f"AI Error: {e}"

def create_pdf(patient, df, analysis):
    """Generate PDF"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elems = []
    
    elems.append(Paragraph("Medical Report", styles['Title']))
    elems.append(Paragraph(f"Patient: {patient['gender']}, {patient['age']}", styles['Normal']))
    elems.append(Spacer(1, 12))
    
    elems.append(Paragraph("Analysis", styles['Heading2']))
    for line in analysis.split('\n'):
        if line.strip(): elems.append(Paragraph(line, styles['Normal']))
    
    elems.append(Spacer(1, 12))
    elems.append(Paragraph("Data Summary", styles['Heading2']))
    
    # Data Table
    data = [['Parameter'] + [str(c).split()[0] for c in df.T.columns]]
    for i, r in df.T.iterrows():
        data.append([i] + [str(x) for x in r.values])
        
    t = Table(data)
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
    elems.append(t)
    
    doc.build(elems)
    buffer.seek(0)
    return buffer

# --- MAIN ---
def main():
    if 'timeline' not in st.session_state: st.session_state.timeline = {}
    if 'files' not in st.session_state: st.session_state.files = []
    if 'analysis' not in st.session_state: st.session_state.analysis = ""

    st.markdown('<h1 class="main-header">🧬 MedLab AI</h1>', unsafe_allow_html=True)
    
    # RAG Status Indicator
    if rag_system and hasattr(rag_system, 'vector_store') and rag_system.vector_store:
        st.markdown('<div class="rag-badge rag-active">✅ RAG System Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rag-badge rag-inactive">⚠️ RAG System Inactive (Basic Mode)</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Settings")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 0, 120, 35)
        api_key = st.text_input("Claude API Key", type="password")
        if st.button("Reset"):
            st.session_state.timeline = {}
            st.session_state.files = []
            st.rerun()

    st.markdown("### Upload Reports")
    uploads = st.file_uploader("Upload PDF/Images", accept_multiple_files=True, type=['pdf','png','jpg'])
    
    if uploads:
        new_files = [f for f in uploads if f.name not in st.session_state.files]
        if new_files:
            for f in new_files:
                text = extract_text(f.read(), f.type)
                data = parse_lab_values(text)
                
                # Simple Date Guessing or Default to today
                date_match = re.search(r'\d{2,4}[-/]\d{2}[-/]\d{2,4}', text)
                date_val = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")
                
                st.session_state.timeline[date_val] = data
                st.session_state.files.append(f.name)
            st.success(f"Processed {len(new_files)} new files!")
            st.rerun()

    if st.session_state.timeline:
        df = pd.DataFrame.from_dict(st.session_state.timeline, orient='index').sort_index()
        df.index = pd.to_datetime(df.index)
        
        tab1, tab2 = st.tabs(["Charts", "Analysis"])
        
        with tab1:
            params = st.multiselect("Parameters", df.columns, default=list(df.columns)[:3])
            if params:
                st.plotly_chart(px.line(df, y=params, markers=True), use_container_width=True)
                st.dataframe(df)

        with tab2:
            if st.button("Generate Analysis"):
                with st.spinner("Analyzing..."):
                    # Get RAG context for latest data
                    latest = df.iloc[-1].to_dict()
                    rag_ctx = rag_system.enhance_analysis(latest) if rag_system else "RAG N/A"
                    
                    # Get Claude Analysis
                    st.session_state.analysis = get_claude_analysis(api_key, df, gender, age, rag_ctx)
            
            if st.session_state.analysis:
                st.markdown(st.session_state.analysis)
                
                pdf = create_pdf({'gender':gender, 'age':age}, df, st.session_state.analysis)
                st.download_button("Download PDF", pdf, "report.pdf", "application/pdf")

if __name__ == "__main__":
    main()
