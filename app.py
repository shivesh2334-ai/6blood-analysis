# app.py - COMPREHENSIVE VERSION
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
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import base64

# Configure poppler path for different environments
if os.path.exists("/usr/bin/pdftoppm"):
    os.environ["PATH"] += os.pathsep + "/usr/bin"
elif os.path.exists("/usr/local/bin/pdftoppm"):
    os.environ["PATH"] += os.pathsep + "/usr/local/bin"

# Import Custom Modules
from medical_reference import REFERENCE_RANGES, TEST_CATEGORIES, CRITICAL_VALUES

# Try importing AI components
try:
    import anthropic
except ImportError:
    anthropic = None

# Initialize RAG System
@st.cache_resource
def get_rag_system():
    try:
        from rag_components import MedLabRAG
        return MedLabRAG()
    except Exception as e:
        st.warning(f"RAG system running in basic mode. Error: {e}")
        return None

rag_system = get_rag_system()

# Page Config
st.set_page_config(
    page_title="MedLab AI - Longitudinal & RAG Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1e3a8a; margin-bottom: 0.5rem; }
    .sub-header { color: #64748b; margin-bottom: 2rem; }
    .card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .critical { color: #dc2626; font-weight: bold; }
    .abnormal { color: #d97706; font-weight: bold; }
    .normal { color: #059669; }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def extract_text_from_document(file_bytes, file_type):
    """Extract text from PDF or Image bytes"""
    text = ""
    try:
        if file_type == "application/pdf":
            try:
                images = pdf2image.convert_from_bytes(file_bytes)
                for img in images:
                    text += pytesseract.image_to_string(img) + "\n"
            except Exception as e:
                return f"Error (Poppler/PDF): {str(e)}"
        else:
            image = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error processing file: {str(e)}"

def extract_date_from_text(text: str) -> str:
    """Attempt to find a report date in the text"""
    # Regex patterns for common date formats
    date_patterns = [
        r'\d{2}[-/]\d{2}[-/]\d{4}',       # DD-MM-YYYY or MM-DD-YYYY
        r'\d{4}[-/]\d{2}[-/]\d{2}',       # YYYY-MM-DD
        r'\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4}', # 12 Jan 2023
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},?\s\d{4}' # Jan 12, 2023
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0]
    return datetime.now().strftime("%Y-%m-%d") # Default to today if not found

def parse_lab_values(text: str) -> dict:
    """Extract lab values using the comprehensive regex patterns"""
    parsed_data = {}
    
    # Combined patterns from previous version (simplified for brevity but effective)
    patterns = {
        'Hemoglobin': r'(?:Hemoglobin|Hb|HGB)[\s:]*(\d+\.?\d*)',
        'RBC': r'(?:RBC|Red Blood Cell)[\s:]*(\d+\.?\d*)',
        'WBC': r'(?:WBC|White Blood Cell)[\s:]*(\d+\.?\d*)',
        'Platelets': r'(?:Platelets|PLT)[\s:]*(\d{3,})',
        'Hematocrit': r'(?:Hematocrit|Hct|HCT)[\s:]*(\d+\.?\d*)',
        'MCV': r'(?:MCV)[\s:]*(\d+\.?\d*)',
        'Glucose_Fasting': r'(?:Fasting Glucose|FBS)[\s:]*(\d+\.?\d*)',
        'HbA1c': r'(?:HbA1c|Glycated)[\s:]*(\d+\.?\d*)',
        'Creatinine': r'(?:Creatinine)[\s:]*(\d+\.?\d*)',
        'eGFR': r'(?:eGFR)[\s:]*(\d+\.?\d*)',
        'BUN': r'(?:BUN|Urea)[\s:]*(\d+\.?\d*)',
        'TSH': r'(?:TSH)[\s:]*(\d+\.?\d*)',
        'Total_Cholesterol': r'(?:Total Cholesterol)[\s:]*(\d+\.?\d*)',
        'Triglycerides': r'(?:Triglycerides)[\s:]*(\d+\.?\d*)',
        'LDL': r'(?:LDL)[\s:]*(\d+\.?\d*)',
        'HDL': r'(?:HDL)[\s:]*(\d+\.?\d*)',
        'ALT': r'(?:ALT|SGPT)[\s:]*(\d+\.?\d*)',
        'AST': r'(?:AST|SGOT)[\s:]*(\d+\.?\d*)',
        'Vitamin_D': r'(?:Vitamin D|25-OH)[\s:]*(\d+\.?\d*)',
        'Vitamin_B12': r'(?:Vitamin B12)[\s:]*(\d+\.?\d*)',
    }
    
    for param, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                val = matches[0]
                if isinstance(val, tuple): val = val[0]
                # Clean value
                val = float(str(val).replace('<','').replace('>','').strip())
                parsed_data[param] = val
            except:
                continue
    return parsed_data

def get_claude_comparative_analysis(api_key, history_df, gender, age, rag_context=""):
    """Use Claude to analyze longitudinal data"""
    if not api_key or not anthropic:
        return "⚠️ Claude API key missing or library not installed."
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        csv_data = history_df.to_csv()
        
        prompt = f"""
        You are an expert medical consultant. Analyze the following longitudinal blood test data for a {age}-year-old {gender}.
        
        HISTORY DATA (Date-wise):
        {csv_data}
        
        MEDICAL CONTEXT (RAG System):
        {rag_context}
        
        Please provide a comparative analysis report:
        1. **Trend Analysis**: Identify improving or worsening trends.
        2. **Correlation**: Connect different parameters (e.g., Kidney function and Electrolytes).
        3. **Critical Alerts**: Highlight any values that are currently dangerous.
        4. **Recommendations**: Suggest next steps, lifestyle changes, or follow-up tests.
        
        Format as a clean professional report.
        """
        
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error calling Claude AI: {str(e)}"

def create_pdf_report(patient_info, history_df, ai_analysis, rag_insights):
    """Generate PDF Report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Header
    elements.append(Paragraph("Longitudinal Medical Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Patient Info
    info = f"<b>Patient:</b> {patient_info['gender']}, {patient_info['age']} yrs | <b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}"
    elements.append(Paragraph(info, styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # AI Analysis
    elements.append(Paragraph("AI Comparative Analysis (Claude)", styles['Heading2']))
    for line in ai_analysis.split('\n'):
        if line.strip():
            elements.append(Paragraph(line, styles['Normal']))
            elements.append(Spacer(1, 4))
    
    # RAG Insights
    if rag_insights:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Medical Reference Insights", styles['Heading2']))
        elements.append(Paragraph(rag_insights, styles['Normal']))
    
    # Data Table
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Historical Data Summary", styles['Heading2']))
    
    # Pivot dataframe for PDF table
    df_t = history_df.transpose().round(2)
    data = [['Parameter'] + [str(c).split()[0] for c in df_t.columns]]
    for idx, row in df_t.iterrows():
        data.append([idx] + [str(x) for x in row.values])
        
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTSIZE', (0,0), (-1,-1), 8),
    ]))
    elements.append(t)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- MAIN APP LOGIC ---

def main():
    # Session State Init
    if 'timeline_data' not in st.session_state:
        st.session_state.timeline_data = {}
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'ai_report' not in st.session_state:
        st.session_state.ai_report = ""

    st.markdown('<h1 class="main-header">🧬 MedLab AI: Longitudinal & RAG Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Patient Profile")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 0, 120, 35)
        
        st.header("AI Settings")
        claude_key = st.text_input("Claude API Key", type="password")
        
        st.header("Data Management")
        if st.button("Reset All Data"):
            st.session_state.timeline_data = {}
            st.session_state.processed_files = []
            st.session_state.ai_report = ""
            st.rerun()

    # File Upload
    st.markdown("### 1. Upload Reports (PDF/Images)")
    uploaded_files = st.file_uploader("Upload multiple historical reports to build timeline", 
                                      accept_multiple_files=True, type=['pdf', 'jpg', 'png'])
    
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        
        if new_files:
            st.info(f"Processing {len(new_files)} new files...")
            temp_data = []
            
            # 1. Extraction Phase
            for f in new_files:
                text = extract_text_from_document(f.read(), f.type)
                parsed = parse_lab_values(text)
                date_str = extract_date_from_text(text)
                temp_data.append({'file': f.name, 'parsed': parsed, 'date_guess': date_str})
            
            # 2. Confirmation Phase (Boxes)
            if temp_data:
                st.write("### 🗓️ Confirm Report Dates")
                with st.form("confirm_dates"):
                    cols = st.columns(len(temp_data)) if len(temp_data) < 3 else st.columns(3)
                    
                    results = {}
                    for i, item in enumerate(temp_data):
                        with cols[i % 3]:
                            st.markdown(f"**{item['file']}**")
                            # Try parsing guess
                            try:
                                d_val = pd.to_datetime(item['date_guess']).date()
                            except:
                                d_val = datetime.now().date()
                                
                            sel_date = st.date_input(f"Date", value=d_val, key=f"d_{i}")
                            st.json(item['parsed'], expanded=False)
                            results[item['file']] = {'date': sel_date, 'data': item['parsed']}
                    
                    if st.form_submit_button("Add to Timeline"):
                        for fname, res in results.items():
                            d_str = res['date'].strftime("%Y-%m-%d")
                            if d_str not in st.session_state.timeline_data:
                                st.session_state.timeline_data[d_str] = {}
                            st.session_state.timeline_data[d_str].update(res['data'])
                            st.session_state.processed_files.append(fname)
                        st.rerun()

    # Analysis Dashboard
    if st.session_state.timeline_data:
        st.markdown("---")
        df = pd.DataFrame.from_dict(st.session_state.timeline_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        tab1, tab2, tab3 = st.tabs(["📈 Visual Timeline", "🤖 AI Analysis & Report", "📋 Raw Data"])
        
        with tab1:
            st.subheader("Comparative Trends")
            params = st.multiselect("Select Parameters", df.columns.tolist(), default=list(df.columns)[:3])
            
            if params:
                fig = px.line(df, y=params, markers=True, title="Parameter History")
                st.plotly_chart(fig, use_container_width=True)
                
                # Delta metrics
                if len(df) > 1:
                    latest = df.iloc[-1]
                    prev = df.iloc[-2]
                    st.write("#### Changes from Previous Report")
                    cols = st.columns(4)
                    for i, p in enumerate(params):
                        val = latest.get(p, 0)
                        diff = val - prev.get(p, 0)
                        cols[i%4].metric(p, f"{val}", f"{diff:.2f}")

        with tab2:
            st.subheader("Claude AI & RAG Analysis")
            
            if st.button("Generate Comparative Analysis"):
                with st.spinner("Consulting Knowledge Base (RAG) & Claude AI..."):
                    # 1. Get RAG Insights for latest data
                    latest_data = df.iloc[-1].to_dict()
                    rag_insights = rag_system.enhance_analysis(latest_data) if rag_system else "RAG Unavailable"
                    
                    # 2. Get Claude Analysis
                    analysis = get_claude_comparative_analysis(
                        claude_key, df, gender, age, rag_context=rag_insights
                    )
                    
                    st.session_state.ai_report = {
                        'text': analysis,
                        'rag': rag_insights
                    }
            
            if st.session_state.ai_report:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("### 🩺 Clinical Analysis")
                    st.markdown(st.session_state.ai_report['text'])
                with col2:
                    st.markdown("### 📚 Reference Context")
                    st.info(st.session_state.ai_report['rag'])
                
                # PDF Generation
                pdf_file = create_pdf_report(
                    {'gender': gender, 'age': age},
                    df,
                    st.session_state.ai_report['text'],
                    st.session_state.ai_report['rag']
                )
                
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_file,
                    file_name="Longitudinal_Lab_Report.pdf",
                    mime="application/pdf"
                )

        with tab3:
            st.dataframe(df.style.highlight_max(axis=0))

if __name__ == "__main__":
    main()
