# app.py - COMPLETE VERSION WITH RAG + ALL FEATURES
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import pdf2image
import io
import re
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import hashlib
import base64
from io import BytesIO
import anthropic
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# RAG Imports with error handling
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    RAG_AVAILABLE = True
except ImportError:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import FAISS
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
        RAG_AVAILABLE = True
    except ImportError:
        RAG_AVAILABLE = False
        HuggingFaceEmbeddings = None
        FAISS = None
        RecursiveCharacterTextSplitter = None
        Document = None

# Configure page
st.set_page_config(
    page_title="MedLab AI Analyzer - RAG-Enhanced Multi-Report Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'parsed_reports' not in st.session_state:
    st.session_state.parsed_reports = {}
if 'correction_mode' not in st.session_state:
    st.session_state.correction_mode = False
if 'claude_client' not in st.session_state:
    st.session_state.claude_client = None
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .report-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #0ea5e9;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .date-header {
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        font-weight: bold;
        margin: 15px 0;
    }
    .trend-improving { color: #059669; font-weight: bold; }
    .trend-worsening { color: #dc2626; font-weight: bold; }
    .trend-stable { color: #6b7280; }
    .parameter-box {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .abnormal-high { color: #dc2626; font-weight: bold; font-size: 1.2rem; }
    .abnormal-low { color: #2563eb; font-weight: bold; font-size: 1.2rem; }
    .normal { color: #059669; font-weight: bold; font-size: 1.2rem; }
    .rag-insight {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 5px solid #22c55e;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .claude-analysis {
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
        border-left: 5px solid #9333ea;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
    }
    .status-badge {
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 10px 20px;
        border-radius: 20px;
        font-size: 0.9rem;
        z-index: 1000;
        color: white;
    }
    .rag-active { background: #22c55e; }
    .rag-inactive { background: #6b7280; }
</style>
""", unsafe_allow_html=True)

# Import reference data
from medical_reference import REFERENCE_RANGES, CRITICAL_VALUES

# Configure poppler path
if os.path.exists("/usr/bin/pdftoppm"):
    os.environ["PATH"] += os.pathsep + "/usr/bin"
elif os.path.exists("/usr/local/bin/pdftoppm"):
    os.environ["PATH"] += os.pathsep + "/usr/local/bin"

# ==================== RAG SYSTEM CLASS ====================
class MedLabRAG:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.initialized = False
        
        if not RAG_AVAILABLE or HuggingFaceEmbeddings is None:
            st.warning("RAG dependencies not available. Running in basic mode.")
            return
            
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize embeddings and vector store"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            if os.path.exists("medical_vectorstore"):
                try:
                    self.vectorstore = FAISS.load_local("medical_vectorstore", self.embeddings)
                    self.initialized = True
                except Exception as e:
                    st.info("Creating new medical knowledge base...")
                    self._create_knowledge_base()
            else:
                self._create_knowledge_base()
                
        except Exception as e:
            st.error(f"RAG initialization failed: {e}")
    
    def _create_knowledge_base(self):
        """Create medical knowledge base"""
        medical_texts = self._load_medical_knowledge()
        
        if not medical_texts or RecursiveCharacterTextSplitter is None or Document is None:
            return
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        documents = [Document(page_content=text, metadata={"source": "medical_knowledge"}) 
                    for text in medical_texts]
        
        chunks = text_splitter.split_documents(documents)
        
        if chunks and self.embeddings:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            try:
                self.vectorstore.save_local("medical_vectorstore")
                self.initialized = True
            except Exception as e:
                st.error(f"Could not save vectorstore: {e}")
    
    def _load_medical_knowledge(self) -> List[str]:
        """Load medical knowledge base"""
        return [
            "Iron Deficiency Anemia: Low MCV (<80 fL), high RDW, low ferritin. Next: Iron studies, blood loss evaluation.",
            "Vitamin B12 Deficiency: Macrocytic anemia (MCV >100), hypersegmented neutrophils. Next: B12 level, intrinsic factor.",
            "Acute Leukemia: Blasts >20% in blood, pancytopenia. Next: Urgent hematology, bone marrow biopsy.",
            "Diabetes Mellitus: HbA1c ≥6.5%, fasting glucose ≥126. Next: Ophthalmology, microalbumin, statin.",
            "Hashimoto's Thyroiditis: Elevated TSH, positive anti-TPO. Next: Levothyroxine, annual TSH monitoring.",
            "Rheumatoid Arthritis: Symmetric polyarthritis, positive RF/anti-CCP. Next: Methotrexate, DMARDs.",
            "Acute Kidney Injury: Creatinine rise >0.3 in 48h. Next: Urinalysis, renal ultrasound, stop nephrotoxins.",
            "NAFLD: Elevated ALT/AST, metabolic syndrome. Next: Weight loss, glucose control, hepatitis screen.",
            "Multiple Myeloma: CRAB features, monoclonal spike. Next: SPEP, free light chains, skeletal survey.",
        ]
    
    def enhance_analysis(self, categorized_tests: Dict, rule_based_analysis: Dict) -> str:
        """Enhance analysis with RAG"""
        if not self.initialized or not self.vectorstore:
            return "RAG system not available."
        
        try:
            query_parts = []
            for category, tests in categorized_tests.items():
                for test, value in tests.items():
                    if isinstance(value, (int, float)):
                        ref = REFERENCE_RANGES.get(test, {})
                        if 'range' in ref:
                            low, high = ref['range']
                        else:
                            low, high = ref.get('male', (0, 0))
                        
                        if value < low or value > high:
                            query_parts.append(f"{test} {value}")
            
            if not query_parts:
                return "All parameters normal."
            
            query = "Laboratory abnormalities: " + ", ".join(query_parts[:5])
            docs = self.vectorstore.similarity_search(query, k=3)
            context = "\n".join([d.page_content for d in docs])
            
            return f"""
            **RAG-Enhanced Insights:**
            
            {context}
            
            *Retrieved from medical knowledge base*
            """
        except Exception as e:
            return f"RAG error: {str(e)}"
    
    def query(self, question: str) -> str:
        """Query knowledge base"""
        if not self.initialized:
            return "RAG not initialized"
        try:
            docs = self.vectorstore.similarity_search(question, k=2)
            return "\n".join([d.page_content for d in docs])
        except Exception as e:
            return f"Query failed: {e}"

# ==================== HELPER FUNCTIONS ====================
def extract_text_from_document(uploaded_file):
    """Extract text from documents"""
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            poppler_paths = [None, "/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"]
            images = None
            
            for poppler_path in poppler_paths:
                try:
                    if poppler_path and os.path.exists(poppler_path):
                        images = pdf2image.convert_from_bytes(uploaded_file.read(), poppler_path=poppler_path)
                    else:
                        images = pdf2image.convert_from_bytes(uploaded_file.read())
                    break
                except Exception:
                    uploaded_file.seek(0)
                    continue
            
            if images is None:
                return ""
                
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
        else:
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return ""

def parse_date_from_text(text: str) -> str:
    """Extract date from text"""
    patterns = [
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',
        r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{2,4})',
        r'Date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'Report Date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                match = matches[0]
                if isinstance(match, tuple) and len(match) >= 3:
                    day, month, year = match[0], match[1], match[2]
                    if len(year) == 2:
                        year = '20' + year
                    try:
                        if isinstance(month, str) and month.isalpha():
                            date_obj = datetime.strptime(f"{day} {month} {year}", "%d %b %Y")
                        else:
                            date_obj = datetime(int(year), int(month), int(day))
                        return date_obj.strftime('%Y-%m-%d')
                    except:
                        pass
            except:
                continue
    
    return datetime.now().strftime('%Y-%m-%d')

def parse_lab_values(text: str) -> Dict:
    """Parse lab values"""
    parsed_data = {}
    
    patterns = {
        'RBC': r'(?:RBC)[\s:]*(\d+\.?\d*)',
        'Hemoglobin': r'(?:Hemoglobin|Hb)[\s:]*(\d+\.?\d*)',
        'Hematocrit': r'(?:Hematocrit|Hct)[\s:]*(\d+\.?\d*)',
        'MCV': r'(?:MCV)[\s:]*(\d+\.?\d*)',
        'MCH': r'(?:MCH)[\s:]*(\d+\.?\d*)',
        'MCHC': r'(?:MCHC)[\s:]*(\d+\.?\d*)',
        'RDW': r'(?:RDW)[\s:]*(\d+\.?\d*)',
        'WBC': r'(?:WBC)[\s:]*(\d+\.?\d*)',
        'Platelets': r'(?:Platelets|PLT)[\s:]*(\d+)',
        'MPV': r'(?:MPV)[\s:]*(\d+\.?\d*)',
        'Neutrophils': r'(?:Neutrophils|NEUT)[\s:]*(\d+\.?\d*)',
        'Lymphocytes': r'(?:Lymphocytes|LYMPH)[\s:]*(\d+\.?\d*)',
        'Monocytes': r'(?:Monocytes|MONO)[\s:]*(\d+\.?\d*)',
        'Eosinophils': r'(?:Eosinophils|EO)[\s:]*(\d+\.?\d*)',
        'Basophils': r'(?:Basophils|BASO)[\s:]*(\d+\.?\d*)',
        'Blasts': r'(?:Blasts)[\s:]*(\d+\.?\d*)',
        'ALT': r'(?:ALT|SGPT)[\s:]*(\d+\.?\d*)',
        'AST': r'(?:AST|SGOT)[\s:]*(\d+\.?\d*)',
        'ALP': r'(?:ALP)[\s:]*(\d+\.?\d*)',
        'GGT': r'(?:GGT)[\s:]*(\d+\.?\d*)',
        'Total_Bilirubin': r'(?:Total Bilirubin)[\s:]*(\d+\.?\d*)',
        'Direct_Bilirubin': r'(?:Direct Bilirubin)[\s:]*(\d+\.?\d*)',
        'Albumin': r'(?:Albumin)[\s:]*(\d+\.?\d*)',
        'Total_Protein': r'(?:Total Protein)[\s:]*(\d+\.?\d*)',
        'Creatinine': r'(?:Creatinine)[\s:]*(\d+\.?\d*)',
        'BUN': r'(?:BUN)[\s:]*(\d+\.?\d*)',
        'eGFR': r'(?:eGFR)[\s:]*(\d+\.?\d*)',
        'Sodium': r'(?:Sodium|Na)[\s:]*(\d+\.?\d*)',
        'Potassium': r'(?:Potassium|K)[\s:]*(\d+\.?\d*)',
        'Calcium': r'(?:Calcium)[\s:]*(\d+\.?\d*)',
        'Glucose_Fasting': r'(?:Fasting Glucose|FBS)[\s:]*(\d+\.?\d*)',
        'HbA1c': r'(?:HbA1c|A1c)[\s:]*(\d+\.?\d*)',
        'TSH': r'(?:TSH)[\s:]*(\d+\.?\d*)',
        'Free_T4': r'(?:Free T4|FT4)[\s:]*(\d+\.?\d*)',
        'Total_Cholesterol': r'(?:Total Cholesterol)[\s:]*(\d+\.?\d*)',
        'HDL': r'(?:HDL)[\s:]*(\d+\.?\d*)',
        'LDL': r'(?:LDL)[\s:]*(\d+\.?\d*)',
        'Triglycerides': r'(?:Triglycerides|TG)[\s:]*(\d+\.?\d*)',
        'CRP': r'(?:CRP)[\s:]*(\d+\.?\d*)',
        'ESR': r'(?:ESR)[\s:]*(\d+\.?\d*)',
        'RF': r'(?:RF)[\s:]*(\d+\.?\d*)',
        'Anti_CCP': r'(?:Anti-CCP)[\s:]*(\d+\.?\d*)',
        'Vitamin_D': r'(?:Vitamin D|25-OH)[\s:]*(\d+\.?\d*)',
        'Vitamin_B12': r'(?:Vitamin B12)[\s:]*(\d+\.?\d*)',
        'Ferritin': r'(?:Ferritin)[\s:]*(\d+\.?\d*)',
        'Iron': r'(?:Iron|Serum Iron)[\s:]*(\d+\.?\d*)',
    }
    
    for param, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                val = matches[0]
                if isinstance(val, tuple):
                    val = val[0]
                val = str(val).replace('<', '').replace('>', '').strip()
                parsed_data[param] = float(val) if val.replace('.','').isdigit() else val
            except:
                parsed_data[param] = matches[0]
    
    return parsed_data

def get_status_class(test: str, value: float, gender: str = 'male'):
    """Get status styling"""
    if test not in REFERENCE_RANGES:
        return "normal", "✓", "Unknown"
    
    ref = REFERENCE_RANGES[test]
    unit = ref.get('unit', '')
    
    if 'male' in ref and 'female' in ref:
        low, high = ref[gender]
    else:
        low, high = ref['range']
    
    if value < low:
        return "abnormal-low", "↓", f"Low ({low}-{high})"
    elif value > high:
        return "abnormal-high", "↑", f"High ({low}-{high})"
    else:
        return "normal", "✓", f"Normal ({low}-{high})"

def calculate_trend(dates: List[str], values: List[float]) -> str:
    """Calculate trend"""
    if len(values) < 2:
        return "stable"
    
    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]
    
    if abs(slope) < 0.05 * np.mean(values):
        return "stable"
    elif slope > 0:
        return "worsening" if values[-1] > np.mean(values) else "improving"
    else:
        return "improving" if values[-1] < np.mean(values) else "worsening"

def create_comparative_dataframe(reports: Dict) -> pd.DataFrame:
    """Create comparative DataFrame"""
    all_tests = set()
    for report in reports.values():
        all_tests.update(report.keys())
    
    data = []
    for test in sorted(all_tests):
        row = {'Test': test}
        values = []
        for date in sorted(reports.keys()):
            val = reports[date].get(test, None)
            row[date] = val
            if val is not None:
                values.append((date, val))
        
        if len(values) >= 2:
            latest_val = values[-1][1]
            prev_val = values[-2][1]
            change = latest_val - prev_val
            pct_change = (change / prev_val * 100) if prev_val != 0 else 0
            row['Change'] = f"{change:+.2f} ({pct_change:+.1f}%)"
            row['Trend'] = calculate_trend([v[0] for v in values], [v[1] for v in values])
        else:
            row['Change'] = "N/A"
            row['Trend'] = "N/A"
        
        data.append(row)
    
    return pd.DataFrame(data)

def initialize_claude():
    """Initialize Claude"""
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))
        if api_key:
            return anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        st.warning(f"Claude not available: {e}")
    return None

def get_claude_comparative_analysis(reports: Dict, gender: str, age: int) -> str:
    """Get Claude analysis"""
    client = st.session_state.claude_client
    if not client:
        return "Claude not configured. Add ANTHROPIC_API_KEY to secrets."
    
    dates = sorted(reports.keys())
    summary = f"Patient: {gender}, {age} years\nDates: {', '.join(dates)}\n\nKey Changes:\n"
    
    # Find significant changes
    all_tests = set()
    for report in reports.values():
        all_tests.update(report.keys())
    
    changes = []
    for test in all_tests:
        values = [(d, reports[d].get(test)) for d in dates if reports[d].get(test) is not None]
        if len(values) >= 2:
            first, last = values[0][1], values[-1][1]
            if isinstance(first, (int, float)) and isinstance(last, (int, float)):
                pct = ((last - first) / first * 100) if first != 0 else 0
                if abs(pct) > 10:
                    changes.append(f"{test}: {first} → {last} ({pct:+.1f}%)")
    
    summary += "\n".join(changes[:15])
    
    prompt = f"""As a senior lab specialist, analyze these trends:

{summary}

Provide:
1. Clinical significance of changes
2. Possible diagnoses
3. Red flags
4. Next steps
5. Overall trajectory"""

    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Claude error: {str(e)}"

def generate_pdf_report(reports: Dict, comparative_df: pd.DataFrame, claude_analysis: str, 
                       rag_analysis: str, gender: str, age: int) -> BytesIO:
    """Generate PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    elements = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24, 
                               textColor=colors.HexColor('#1e40af'), alignment=1, spaceAfter=30)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14,
                                  textColor=colors.HexColor('#1e40af'), spaceAfter=12, spaceBefore=12)
    
    # Title
    elements.append(Paragraph("MedLab AI - Comprehensive Laboratory Report", title_style))
    elements.append(Paragraph(f"<b>Patient:</b> {gender}, {age} years", styles['Normal']))
    elements.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Summary
    elements.append(Paragraph("Executive Summary", heading_style))
    elements.append(Paragraph(f"• Reports: {len(reports)}", styles['Normal']))
    elements.append(Paragraph(f"• Date Range: {min(reports.keys())} to {max(reports.keys())}", styles['Normal']))
    elements.append(Paragraph(f"• Parameters: {len(set().union(*[set(r.keys()) for r in reports.values()]))}", styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Comparative Table
    elements.append(Paragraph("Comparative Results", heading_style))
    table_data = [['Parameter'] + sorted(reports.keys()) + ['Trend']]
    
    for _, row in comparative_df.iterrows():
        trend_sym = {'improving': '↗', 'worsening': '↘', 'stable': '→', 'N/A': '-'}.get(row['Trend'], '-')
        table_row = [row['Test']] + [str(row.get(d, '-')) for d in sorted(reports.keys())] + [trend_sym]
        table_data.append(table_row)
    
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.3*inch))
    
    # AI Analysis
    if claude_analysis:
        elements.append(PageBreak())
        elements.append(Paragraph("Claude AI Analysis", heading_style))
        for para in claude_analysis.split('\n\n'):
            if para.strip():
                elements.append(Paragraph(para.replace('\n', '<br/>'), styles['Normal']))
                elements.append(Spacer(1, 0.1*inch))
    
    # RAG Analysis
    if rag_analysis and "not available" not in rag_analysis:
        elements.append(Paragraph("RAG-Enhanced Insights", heading_style))
        elements.append(Paragraph(rag_analysis.replace('\n', '<br/>'), styles['Normal']))
    
    # Disclaimer
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(
        "<b>Disclaimer:</b> AI-generated for educational purposes. Verify with healthcare professionals.",
        ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8, textColor=colors.grey)
    ))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

def categorize_tests(tests: Dict) -> Dict[str, Dict]:
    """Categorize tests"""
    categories = {
        'Hematology': ['RBC', 'Hemoglobin', 'Hematocrit', 'MCV', 'MCH', 'MCHC', 'RDW', 
                      'WBC', 'Platelets', 'MPV', 'Neutrophils', 'Lymphocytes', 'Monocytes',
                      'Eosinophils', 'Basophils', 'Blasts'],
        'Liver_Function': ['ALT', 'AST', 'ALP', 'GGT', 'Total_Bilirubin', 'Direct_Bilirubin',
                          'Albumin', 'Total_Protein'],
        'Kidney_Function': ['Creatinine', 'BUN', 'eGFR', 'Sodium', 'Potassium', 'Calcium'],
        'Metabolic': ['Glucose_Fasting', 'HbA1c', 'Insulin'],
        'Endocrine': ['TSH', 'Free_T4'],
        'Lipid_Profile': ['Total_Cholesterol', 'HDL', 'LDL', 'Triglycerides'],
        'Immunology': ['CRP', 'ESR', 'RF', 'Anti_CCP'],
        'Vitamins': ['Vitamin_D', 'Vitamin_B12', 'Ferritin', 'Iron']
    }
    
    result = {cat: {} for cat in categories.keys()}
    for test, value in tests.items():
        for cat, cat_tests in categories.items():
            if test in cat_tests:
                result[cat][test] = value
                break
    
    return {k: v for k, v in result.items() if v}

def display_date_wise_reports(reports: Dict, gender: str, rag_system: MedLabRAG):
    """Display reports by date with RAG insights"""
    for date in sorted(reports.keys()):
        st.markdown(f'<div class="date-header">📅 {date}</div>', unsafe_allow_html=True)
        
        report_data = reports[date]
        categorized = categorize_tests(report_data)
        
        # RAG insight for this report
        if rag_system and rag_system.initialized:
            with st.expander("🧠 RAG Insight for this report"):
                rag_insight = rag_system.enhance_analysis(categorized, {})
                st.markdown(f'<div class="rag-insight">{rag_insight}</div>', unsafe_allow_html=True)
        
        # Display by category
        for cat_name, cat_tests in categorized.items():
            if cat_tests:
                with st.expander(f"{cat_name.replace('_', ' ')} ({len(cat_tests)} parameters)"):
                    cols = st.columns(3)
                    col_idx = 0
                    for test, value in cat_tests.items():
                        with cols[col_idx % 3]:
                            if isinstance(value, (int, float)):
                                status_class, icon, ref_text = get_status_class(test, value, gender)
                                st.markdown(f"""
                                <div class="parameter-box">
                                    <small>{test.replace('_', ' ')}</small><br>
                                    <span class="{status_class}">{value} {icon}</span><br>
                                    <small style="color: #6b7280;">{ref_text}</small>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"**{test}**: {value}")
                        col_idx += 1

# ==================== MAIN APPLICATION ====================
def main():
    st.markdown('<h1 class="main-header">🧬 MedLab AI Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">RAG-Enhanced Multi-Report Comparative Analysis</p>', unsafe_allow_html=True)
    
    # Initialize systems
    if st.session_state.claude_client is None:
        st.session_state.claude_client = initialize_claude()
    
    if st.session_state.rag_system is None and RAG_AVAILABLE:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = MedLabRAG()
    
    # Status badges
    rag_status = "rag-active" if (st.session_state.rag_system and st.session_state.rag_system.initialized) else "rag-inactive"
    rag_text = "🧠 RAG Active" if (st.session_state.rag_system and st.session_state.rag_system.initialized) else "🧠 RAG Offline"
    st.markdown(f'<div class="status-badge {rag_status}">{rag_text}</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Patient Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=0, max_value=120, value=35)
        
        st.header("Upload Reports")
        uploaded_files = st.file_uploader(
            "Choose files (PDF, JPG, PNG)", 
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("🔍 Process All Reports"):
            progress_bar = st.progress(0)
            for idx, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    text = extract_text_from_document(uploaded_file)
                    if text:
                        report_date = parse_date_from_text(text)
                        values = parse_lab_values(text)
                        
                        # Handle duplicate dates
                        base_date = report_date
                        counter = 1
                        while report_date in st.session_state.parsed_reports:
                            report_date = f"{base_date}_{counter}"
                            counter += 1
                        
                        st.session_state.parsed_reports[report_date] = values
                        st.success(f"✓ {uploaded_file.name}: {len(values)} parameters ({base_date})")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            st.balloons()
        
        # Manual entry
        with st.expander("Manual Entry"):
            manual_date = st.date_input("Report Date", datetime.now())
            manual_test = st.selectbox("Test", list(REFERENCE_RANGES.keys()))
            manual_value = st.number_input("Value", step=0.01)
            
            if st.button("Add Value"):
                date_str = manual_date.strftime('%Y-%m-%d')
                if date_str not in st.session_state.parsed_reports:
                    st.session_state.parsed_reports[date_str] = {}
                st.session_state.parsed_reports[date_str][manual_test] = manual_value
                st.success(f"Added {manual_test}: {manual_value}")
        
        # Query RAG directly
        if st.session_state.rag_system and st.session_state.rag_system.initialized:
            with st.expander("🔍 Query Medical Knowledge"):
                query = st.text_input("Ask medical question")
                if query and st.button("Search"):
                    result = st.session_state.rag_system.query(query)
                    st.info(result)
        
        if st.session_state.parsed_reports and st.button("🗑️ Clear All Data"):
            st.session_state.parsed_reports = {}
            st.rerun()
    
    # Main content
    if st.session_state.parsed_reports:
        tabs = st.tabs(["📋 Date-wise Reports", "📊 Comparative Analysis", "🤖 AI Analysis", "📑 Final Report"])
        
        with tabs[0]:
            st.subheader("Reports by Date")
            display_date_wise_reports(st.session_state.parsed_reports, gender.lower(), st.session_state.rag_system)
        
        with tabs[1]:
            st.subheader("Comparative Analysis")
            if len(st.session_state.parsed_reports) >= 2:
                comp_df = create_comparative_dataframe(st.session_state.parsed_reports)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                
                # Visualizations
                st.subheader("Trend Charts")
                available_tests = [t for t in REFERENCE_RANGES.keys() 
                                  if any(t in r for r in st.session_state.parsed_reports.values())]
                selected_param = st.selectbox("Select parameter", available_tests)
                
                if selected_param:
                    dates, values = [], []
                    for date in sorted(st.session_state.parsed_reports.keys()):
                        val = st.session_state.parsed_reports[date].get(selected_param)
                        if val is not None:
                            dates.append(date)
                            values.append(val)
                    
                    if values:
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers', name=selected_param))
                        
                        if selected_param in REFERENCE_RANGES:
                            ref = REFERENCE_RANGES[selected_param]
                            low, high = ref.get('range', ref.get(gender.lower(), (0, 0)))
                            fig.add_hline(y=low, line_dash="dash", line_color="green")
                            fig.add_hline(y=high, line_dash="dash", line_color="green")
                        
                        fig.update_layout(title=f"{selected_param} Trend", xaxis_title="Date", yaxis_title="Value")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Upload at least 2 reports for comparison")
        
        with tabs[2]:
            st.subheader("AI-Powered Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Claude AI Analysis")
                if len(st.session_state.parsed_reports) >= 2:
                    if st.button("🤖 Generate Claude Analysis"):
                        with st.spinner("Analyzing with Claude..."):
                            analysis = get_claude_comparative_analysis(
                                st.session_state.parsed_reports, gender, age
                            )
                            st.session_state.claude_analysis = analysis
                    
                    if 'claude_analysis' in st.session_state:
                        st.markdown(f'<div class="claude-analysis">{st.session_state.claude_analysis}</div>', 
                                  unsafe_allow_html=True)
            
            with col2:
                st.markdown("### RAG-Enhanced Insights")
                if st.session_state.rag_system and st.session_state.rag_system.initialized:
                    # Aggregate all abnormalities for RAG
                    all_categorized = {}
                    for date, report in st.session_state.parsed_reports.items():
                        cat = categorize_tests(report)
                        for c, tests in cat.items():
                            if c not in all_categorized:
                                all_categorized[c] = {}
                            all_categorized[c].update(tests)
                    
                    rag_insight = st.session_state.rag_system.enhance_analysis(all_categorized, {})
                    st.markdown(f'<div class="rag-insight">{rag_insight}</div>', unsafe_allow_html=True)
                else:
                    st.info("RAG system not available")
        
        with tabs[3]:
            st.subheader("Generate Final Report")
            
            if len(st.session_state.parsed_reports) >= 1:
                report_title = st.text_input("Report Title", "Laboratory Analysis Report")
                include_claude = st.checkbox("Include Claude Analysis", value=True)
                include_rag = st.checkbox("Include RAG Insights", value=True)
                
                if st.button("📄 Generate PDF"):
                    with st.spinner("Generating PDF..."):
                        comp_df = create_comparative_dataframe(st.session_state.parsed_reports)
                        claude_text = st.session_state.get('claude_analysis', '') if include_claude else ''
                        
                        rag_text = ""
                        if include_rag and st.session_state.rag_system:
                            all_cat = {}
                            for report in st.session_state.parsed_reports.values():
                                cat = categorize_tests(report)
                                for c, t in cat.items():
                                    if c not in all_cat:
                                        all_cat[c] = {}
                                    all_cat[c].update(t)
                            rag_text = st.session_state.rag_system.enhance_analysis(all_cat, {})
                        
                        pdf_buffer = generate_pdf_report(
                            st.session_state.parsed_reports,
                            comp_df,
                            claude_text,
                            rag_text,
                            gender,
                            age
                        )
                        
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=pdf_buffer,
                            file_name=f"{report_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
    else:
        st.info("👆 Upload lab reports to begin analysis")

if __name__ == "__main__":
    main()
