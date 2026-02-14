# medical_reference.py - UPDATED VERSION
# Comprehensive reference ranges for all laboratory parameters

REFERENCE_RANGES = {
    # Hematology
    'RBC': {'male': (4.5, 5.5), 'female': (4.0, 5.0), 'unit': 'x10^12/L'},
    'Hemoglobin': {'male': (13.5, 17.5), 'female': (12.0, 16.0), 'unit': 'g/dL'},
    'Hematocrit': {'male': (41, 50), 'female': (36, 44), 'unit': '%'},
    'MCV': {'range': (80, 100), 'unit': 'fL'},
    'MCH': {'range': (27, 33), 'unit': 'pg'},
    'MCHC': {'range': (32, 36), 'unit': 'g/dL'},
    'RDW': {'range': (11.5, 14.5), 'unit': '%'},
    'WBC': {'range': (4.5, 11.0), 'unit': 'x10^9/L'},
    'Platelets': {'range': (150, 450), 'unit': 'x10^9/L'},
    'MPV': {'range': (7.5, 11.5), 'unit': 'fL'},
    'Neutrophils': {'range': (40, 70), 'unit': '%'},
    'Lymphocytes': {'range': (20, 40), 'unit': '%'},
    'Monocytes': {'range': (2, 8), 'unit': '%'},
    'Eosinophils': {'range': (1, 4), 'unit': '%'},
    'Basophils': {'range': (0.5, 1), 'unit': '%'},
    'Reticulocytes': {'range': (0.5, 2.5), 'unit': '%'},
    'Blasts': {'range': (0, 0), 'unit': '%'},
    
    # Liver Function
    'ALT': {'range': (7, 56), 'unit': 'U/L'},
    'AST': {'range': (10, 40), 'unit': 'U/L'},
    'ALP': {'range': (44, 147), 'unit': 'U/L'},
    'GGT': {'male': (10, 71), 'female': (6, 42), 'unit': 'U/L'},
    'Total_Bilirubin': {'range': (0.1, 1.2), 'unit': 'mg/dL'},
    'Direct_Bilirubin': {'range': (0, 0.3), 'unit': 'mg/dL'},
    'Indirect_Bilirubin': {'range': (0.1, 0.9), 'unit': 'mg/dL'},
    'Total_Protein': {'range': (6.0, 8.3), 'unit': 'g/dL'},
    'Albumin': {'range': (3.5, 5.0), 'unit': 'g/dL'},
    'Globulin': {'range': (2.3, 3.5), 'unit': 'g/dL'},
    'A_G_Ratio': {'range': (1.0, 2.2), 'unit': 'ratio'},
    
    # Kidney Function
    'Creatinine': {'male': (0.7, 1.3), 'female': (0.6, 1.1), 'unit': 'mg/dL'},
    'BUN': {'range': (7, 20), 'unit': 'mg/dL'},
    'eGFR': {'range': (90, 120), 'unit': 'mL/min/1.73m2'},
    'Uric_Acid': {'male': (3.5, 7.2), 'female': (2.6, 6.0), 'unit': 'mg/dL'},
    'Sodium': {'range': (135, 145), 'unit': 'mEq/L'},
    'Potassium': {'range': (3.5, 5.0), 'unit': 'mEq/L'},
    'Chloride': {'range': (98, 106), 'unit': 'mEq/L'},
    'Bicarbonate': {'range': (22, 29), 'unit': 'mEq/L'},
    'Calcium': {'range': (8.5, 10.5), 'unit': 'mg/dL'},
    'Phosphorus': {'range': (2.5, 4.5), 'unit': 'mg/dL'},
    'Magnesium': {'range': (1.7, 2.2), 'unit': 'mg/dL'},
    
    # Metabolic/Diabetes
    'Glucose_Fasting': {'range': (70, 100), 'unit': 'mg/dL'},
    'Glucose_Random': {'range': (70, 140), 'unit': 'mg/dL'},
    'HbA1c': {'range': (4.0, 5.6), 'unit': '%'},
    'Insulin': {'range': (2.6, 24.9), 'unit': 'μU/mL'},
    'C_Peptide': {'range': (0.8, 3.1), 'unit': 'ng/mL'},
    
    # Thyroid
    'TSH': {'range': (0.4, 4.0), 'unit': 'μIU/mL'},
    'T3': {'range': (80, 200), 'unit': 'ng/dL'},
    'T4': {'range': (5.0, 12.0), 'unit': 'μg/dL'},
    'Free_T3': {'range': (2.3, 4.2), 'unit': 'pg/mL'},
    'Free_T4': {'range': (0.8, 1.8), 'unit': 'ng/dL'},
    'Anti_TPO': {'range': (0, 35), 'unit': 'IU/mL'},
    'Anti_Thyroglobulin': {'range': (0, 40), 'unit': 'IU/mL'},
    
    # Lipids
    'Total_Cholesterol': {'range': (0, 200), 'unit': 'mg/dL'},
    'HDL': {'male': (40, 60), 'female': (50, 60), 'unit': 'mg/dL'},
    'LDL': {'range': (0, 100), 'unit': 'mg/dL'},
    'Triglycerides': {'range': (0, 150), 'unit': 'mg/dL'},
    'VLDL': {'range': (5, 40), 'unit': 'mg/dL'},
    'Non_HDL_Cholesterol': {'range': (0, 130), 'unit': 'mg/dL'},
    
    # Rheumatology/Immunology
    'RF': {'range': (0, 20), 'unit': 'IU/mL'},
    'Anti_CCP': {'range': (0, 20), 'unit': 'U/mL'},
    'dsDNA': {'range': (0, 100), 'unit': 'IU/mL'},
    'ESR': {'male': (0, 15), 'female': (0, 20), 'unit': 'mm/hr'},
    'CRP': {'range': (0, 10), 'unit': 'mg/L'},
    'ASO': {'range': (0, 200), 'unit': 'IU/mL'},
    
    # Coagulation
    'PT': {'range': (11, 13.5), 'unit': 'seconds'},
    'INR': {'range': (0.8, 1.2), 'unit': 'ratio'},
    'aPTT': {'range': (25, 35), 'unit': 'seconds'},
    'Fibrinogen': {'range': (200, 400), 'unit': 'mg/dL'},
    'D_Dimer': {'range': (0, 500), 'unit': 'ng/mL'},
    
    # Tumor Markers
    'AFP': {'range': (0, 10), 'unit': 'ng/mL'},
    'CEA': {'non-smoker': (0, 2.5), 'smoker': (0, 5.0), 'unit': 'ng/mL'},
    'CA_125': {'range': (0, 35), 'unit': 'U/mL'},
    'CA_19_9': {'range': (0, 37), 'unit': 'U/mL'},
    'PSA': {'range': (0, 4.0), 'unit': 'ng/mL'},
    'CA_15_3': {'range': (0, 30), 'unit': 'U/mL'},
    
    # Vitamins/Minerals
    'Vitamin_D': {'range': (30, 100), 'unit': 'ng/mL'},
    'Vitamin_B12': {'range': (200, 900), 'unit': 'pg/mL'},
    'Folate': {'range': (2.7, 17.0), 'unit': 'ng/mL'},
    'Iron': {'male': (65, 175), 'female': (50, 170), 'unit': 'μg/dL'},
    'Ferritin': {'male': (20, 300), 'female': (10, 120), 'unit': 'ng/mL'},
    'TIBC': {'range': (250, 400), 'unit': 'μg/dL'},
    'Transferrin_Saturation': {'range': (20, 50), 'unit': '%'},
}

# Critical values requiring immediate attention
CRITICAL_VALUES = {
    'Hemoglobin': (7, 20),
    'WBC': (2, 30),
    'Platelets': (20, 1000),
    'Potassium': (2.5, 6.5),
    'Sodium': (120, 160),
    'Glucose_Fasting': (40, 400),
    'Creatinine': (0, 10),
    'Calcium': (6, 14),
    'pH': (7.2, 7.6),
    'pCO2': (20, 60),
    'pO2': (40, 200),
    'Bicarbonate': (10, 40),
    'INR': (0, 5),
    'Blasts': (0, 5),
}

# Test categorization for UI organization
TEST_CATEGORIES = {
    'Hematology': {
        'icon': '🩸',
        'color': '#8b5cf6',
        'tests': ['CBC', 'Coagulation', 'Hemolysis workup', 'Blood cancer screening']
    },
    'Metabolism': {
        'icon': '⚡',
        'color': '#f59e0b',
        'tests': ['Glucose metabolism', 'Lipid profile', 'Electrolytes', 'Acid-base']
    },
    'Endocrinology': {
        'icon': '🦋',
        'color': '#ec4899',
        'tests': ['Thyroid function', 'Adrenal function', 'Pituitary markers', 'Diabetes monitoring']
    },
    'Hepatology': {
        'icon': '🫁',
        'color': '#10b981',
        'tests': ['Liver enzymes', 'Synthetic function', 'Biliary markers', 'Viral hepatitis']
    },
    'Nephrology': {
        'icon': '🫘',
        'color': '#3b82f6',
        'tests': ['Renal function', 'Electrolytes', 'Acid-base', 'Proteinuria markers']
    },
    'Immunology': {
        'icon': '🛡️',
        'color': '#f97316',
        'tests': ['Autoimmune markers', 'Immunoglobulins', 'Complement', 'Allergies']
    },
    'Rheumatology': {
        'icon': '🦴',
        'color': '#6366f1',
        'tests': ['Arthritis panel', 'Connective tissue', 'Vasculitis', 'Inflammatory markers']
    },
    'Oncology': {
        'icon': '🔬',
        'color': '#dc2626',
        'tests': ['Tumor markers', 'Hematologic malignancy', 'Paraneoplastic', 'Monitoring']
    }
}
