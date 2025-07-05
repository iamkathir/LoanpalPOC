from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
import google.generativeai as genai
import json
import re
from fastapi import Request
from fastapi.responses import JSONResponse, HTMLResponse
import datetime
import ast
import sqlite3

app = FastAPI(title="Best Bank Loan Credit Engine", description="Get the best bank loan recommendation based on user profile and bank policy.")

# Enable CORS for all origins (for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "AIzaSyDo0ENrEXvk-8sxwaEIxRGcCxNff27cFww"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

class UserProfile(BaseModel):
    name: str
    type_of_loan: str
    address: str
    loan_amount: str
    date: str
    employment_type: str  # 'Salaried' or 'Self-Employed'
    monthly_income: int
    existing_loan: bool
    existing_loan_amount: Optional[int] = None
    has_coapplicant: Optional[bool] = False
    coapplicant_monthly_income: Optional[int] = None
    coapplicant_existing_loan: Optional[bool] = False
    coapplicant_existing_loan_amount: Optional[int] = None
    phone_number: Optional[str] = None  # <-- Add phone number field
    loan_repayment_issue: Optional[bool] = False  # New field for repayment issue

class EligibleBank(BaseModel):
    bank: str
    interest_rate: str
    loan_amount: str
    tenure: str
    emi: str

class RecommendationResponse(BaseModel):
    name: str
    type_of_loan: str
    address: str
    loan_amount: str
    date: str
    eligible_banks: List[EligibleBank]
    enhancement_insights: List[str]
    total_emi: str

class AllBanksUserProfile(BaseModel):
    product: str
    cibil_score: int
    employment_type: str
    monthly_income: int
    emi_after_loan: int
    age: int
    property_type: str
    property_approval: str
    property_value: int
    loan_amount_requested: int
    loan_tenure_requested: int
    is_woman: bool

class ApplicantBusinessDetails(BaseModel):
    years_of_operation: Optional[int] = None
    itr_available: Optional[bool] = None
    itr_years: Optional[int] = None

class ExistingLoanDetails(BaseModel):
    loan_amount: Optional[int] = 0
    monthly_emi: Optional[int] = 0

class Applicant(BaseModel):
    name: str
    age: int
    cibil_score: int
    income_type: str
    income_mode: str
    income_amount: int
    business_details: ApplicantBusinessDetails
    existing_loan: ExistingLoanDetails
    itr_3years: Optional[bool] = None  # Added for eligibility logic
    phone_number: str  # <-- Now required, not optional

class LoanDetails(BaseModel):
    type: str
    property_value: int
    property_approval: str
    loan_amount_requested: Optional[int] = None  # <-- Added for UI integration

class CoApplicantBusinessDetails(BaseModel):
    years_of_operation: Optional[int] = None
    itr_available: Optional[bool] = None
    itr_years: Optional[int] = None

class CoApplicantExistingLoanDetails(BaseModel):
    loan_amount: Optional[int] = 0
    monthly_emi: Optional[int] = 0

class CoApplicant(BaseModel):
    exists: bool = False
    income_type: Optional[str] = None
    income_mode: Optional[str] = None
    income_amount: Optional[int] = 0
    business_details: Optional[CoApplicantBusinessDetails] = None
    existing_loan: Optional[CoApplicantExistingLoanDetails] = None
    # Remove salary_type, use income_mode for UI/API
    existing_emi: Optional[int] = 0
    missing_or_late_payment: Optional[bool] = None
    itr_3years: Optional[bool] = None  # Added for UI integration

class LoanFormInput(BaseModel):
    applicant: Applicant
    loan_details: LoanDetails
    co_applicant: Optional[CoApplicant] = None
    phone_number: Optional[str] = None  # <-- Add phone number field
    loan_repayment_issue: Optional[bool] = False  # <-- Add repayment issue at top level

def flatten_loan_form_input(data: LoanFormInput) -> Dict[str, Any]:
    # Map nested input to flat structure for policy eligibility
    applicant = data.applicant
    co_app = data.co_applicant or CoApplicant()
    loan = data.loan_details
    # Calculate total income and EMI
    co_income = co_app.income_amount or 0
    co_emi = co_app.existing_emi if hasattr(co_app, 'existing_emi') and co_app.existing_emi else 0
    total_income = applicant.income_amount + co_income
    total_emi = (applicant.existing_loan.monthly_emi or 0) + co_emi
    requested_loan_amount = loan.loan_amount_requested if hasattr(loan, 'loan_amount_requested') and loan.loan_amount_requested is not None else 0
    # Get co-applicant age if present
    coapp_age = getattr(co_app, 'age', None)
    return {
        "product": "HL" if loan.type == "home_loan" else "LAP",
        "cibil_score": applicant.cibil_score,
        # Map applicant.income_type to 'business' if 'business', else 'salaried' for backend logic
        "employment_type": ("business" if applicant.income_type.lower() == "business" else "salaried"),
        "income_mode": applicant.income_mode,  # <-- Added for eligibility logic
        "monthly_income": total_income,
        "emi_after_loan": total_emi,
        "age": applicant.age,
        "coapp_age": coapp_age,
        "property_type": "Residential" if loan.type == "home_loan" else "Commercial",
        "property_approval": loan.property_approval,
        "property_value": loan.property_value,
        "loan_amount_requested": requested_loan_amount,
        "loan_tenure_requested": 20,  # Default or add to form if needed
        "is_woman": False,  # Add logic if needed
        # Co-applicant fields for advanced logic
        "coapp_income_mode": getattr(co_app, 'income_mode', None),
        # Map co_app.income_type to 'business' if 'business' else 'salaried' for backend logic
        "coapp_income_type": (getattr(co_app, 'income_type', '').lower() if getattr(co_app, 'income_type', '').lower() == 'business' else 'salaried'),
        "coapp_income_amount": co_income,
        "coapp_existing_emi": co_emi,
        "coapp_missing_or_late_payment": getattr(co_app, 'missing_or_late_payment', None),
        "itr_3years": getattr(applicant, 'itr_3years', None),
        "coapp_itr_3years": getattr(co_app, 'itr_3years', None),
        "phone_number": getattr(data, 'phone_number', None),  # <-- Add phone number to flat dict
        "loan_repayment_issue": getattr(data, 'loan_repayment_issue', False),
    }

@app.post("/recommend-loan", response_model=RecommendationResponse)
def recommend_loan(user_profile: UserProfile = Body(...)):
    # Accept phone_number from user_profile if present (for plain HTML form)
    phone_number = None
    if isinstance(user_profile, dict):
        phone_number = user_profile.get('phone_number')
        if phone_number:
            user_profile['phone_number'] = phone_number
    prompt = f"""
You are a credit engine for a bank. Based on the following user profile and the detailed bank policy table below, calculate and recommend loan eligibility for all major banks. Use the policy table for all calculations, including FOIR (Fixed Obligations to Income Ratio), LTV (Loan to Value), minimum CIBIL, income, age, employment, property type, and all other criteria. If a field is not present in the user profile, assume the most conservative value.

Bank Policy Table (Python dict):
{json.dumps(BANK_POLICIES, indent=2)}

If the user has a co-applicant, include the co-applicant's monthly income and existing loan obligations in all calculations (FOIR, total income, total EMI, etc). If the co-applicant has an existing loan, include their EMI in the total EMI calculation.

For each bank, provide:
- bank: Bank name
- interest_rate: Interest rate range or value
- loan_amount: Eligible loan amount (as a string, e.g., "2000000" or "20 Lakhs")
- tenure: Recommended tenure (as a string, e.g., "20 years")
- emi: Calculated EMI for the eligible loan (as a string, e.g., "16500" or "₹16,500")
- remarks: Short eligibility remarks (e.g., "Eligible", "CIBIL below minimum", etc)

Also provide:
- enhancement_insights: A list of suggestions (strings) for improving eligibility or getting a better rate. Example: ["Add co-applicant", "Close existing loans"]
- total_emi: The total EMI (including all existing loans for both applicant and co_applicant, as a string)

Return a JSON object with these fields:
- name
- type_of_loan
- address
- loan_amount (as a string)
- date
- eligible_banks (list of all banks as above)
- enhancement_insights (a list of strings)
- total_emi (as a string)

User Profile: {user_profile.dict()}

Respond ONLY with a valid JSON object matching the above structure. All numeric fields must be returned as strings. Do not include any extra text, markdown, or code block formatting.
"""
    response = model.generate_content(prompt)
    # Remove markdown/code block formatting if present
    text = response.text.strip()
    text = re.sub(r'^```json|^```|```$', '', text, flags=re.MULTILINE).strip()
    try:
        result = json.loads(text)
        # Fix enhancement_insights if it's returned as a single string instead of a list
        if isinstance(result.get("enhancement_insights"), str):
            result["enhancement_insights"] = [result["enhancement_insights"]]
    except Exception:
        return {"result": text}
    return result



# --- CIBIL ROI Parser Utilities ---
def parse_cibil_roi_conditions(conditions: str):
    """
    Parses the 'conditions' string for CIBIL-based ROI slabs.
    Returns a list of dicts: [{min_cibil, max_cibil, roi_min, roi_max}]
    Handles patterns like:
      - CIBIL 700+ -> 7.50–7.80%
      - CIBIL 750+ -> 8.50%
      - CIBIL 700–729: 9.15%
      - CIBIL <700: 9.45%
      - CIBIL >730 → 9.35%
      - CIBIL 700–730 → 9.85–12%
    """
    slabs = []
    # Remove commas and normalize dashes
    cond = conditions.replace(",", ".").replace("–", "-").replace("→", ":").replace("—", "-")
    # Patterns
    patterns = [
        # CIBIL >730 → 9.35%
        (r"CIBIL\s*>\s*(\d+)\s*[:\-\u2192>]+\s*([\d.]+)%", lambda m: {"min_cibil": int(m.group(1))+1, "max_cibil": 900, "roi_min": float(m.group(2)), "roi_max": float(m.group(2))}),
        # CIBIL <700: 9.45%
        (r"CIBIL\s*<\s*(\d+)\s*[:\-\u2192>]+\s*([\d.]+)%", lambda m: {"min_cibil": 0, "max_cibil": int(m.group(1))-1, "roi_min": float(m.group(2)), "roi_max": float(m.group(2))}),
        # CIBIL 700+ -> 7.50–7.80% or CIBIL 750+ -> 8.50%
        (r"CIBIL\s*(\d+)\+\s*[:\-\u2192>]+\s*([\d.]+)(?:-([\d.]+))?%", lambda m: {"min_cibil": int(m.group(1)), "max_cibil": 900, "roi_min": float(m.group(2)), "roi_max": float(m.group(3)) if m.group(3) else float(m.group(2))}),
        # CIBIL 700–730 → 9.85–12%
        (r"CIBIL\s*(\d+)[-–](\d+)\s*[:\-\u2192>]+\s*([\d.]+)(?:-([\d.]+))?%", lambda m: {"min_cibil": int(m.group(1)), "max_cibil": int(m.group(2)), "roi_min": float(m.group(3)), "roi_max": float(m.group(4)) if m.group(4) else float(m.group(3))}),
        # CIBIL 700–729: 9.15%
        (r"CIBIL\s*(\d+)[-–](\d+)[:\s]+([\d.]+)%", lambda m: {"min_cibil": int(m.group(1)), "max_cibil": int(m.group(2)), "roi_min": float(m.group(3)), "roi_max": float(m.group(3))}),
    ]
    for pat, fn in patterns:
        for m in re.finditer(pat, cond):
            try:
                slab = fn(m)
                # Defensive: skip if cibil or roi is not a number
                if not (isinstance(slab["min_cibil"], int) and isinstance(slab["roi_min"], float)):
                    continue
                slabs.append(slab)
            except Exception:
                continue
    slabs.sort(key=lambda x: x["min_cibil"])
    return slabs

def get_dynamic_cibil_roi(cibil: int, conditions: str):
    slabs = parse_cibil_roi_conditions(conditions)
    for slab in slabs:
        if slab["min_cibil"] <= cibil <= slab["max_cibil"]:
            return slab["roi_min"]
    return None

# Bank policy table as a Python dictionary
BANK_POLICIES = {
    "SBI": {
        "min_cibil": 700,
        "min_income": 25000,
        "itr_for_business_profile": True,
        "foir_slabs": [
            {"min_income": 0, "max_income": 500000, "foir": 0.55},
            {"min_income": 500001, "max_income": 800000, "foir": 0.60},
            {"min_income": 800001, "max_income": 1000000, "foir": 0.65},
            {"min_income": 1000001, "max_income": float('inf'), "foir": 0.70}
        ],
        "min_age": 21,
        "max_age": 70,
        "employment_types": ["Salaried", "Self-Employed"],
        "property_types": ["Residential", "Commercial", "Plots"],
        "approved_layout": True,
        "ltv_hl_slabs": [
            {"max_value": 3000000, "ltv": 0.9},
            {"max_value": 7500000, "ltv": 0.8},
            {"max_value": float('inf'), "ltv": 0.75}
        ],
        "ltv_lap": (0, 0.65),
        "loan_amount_range": (1000000, 500000000),
        "tenure_range": {"hl": (5, 30), "lap": (7, 30)},
        "interest_rate_hl": [
            {"min_cibil": 700, "max_cibil": 719, "roi": 7.80},
            {"min_cibil": 720, "max_cibil": 739, "roi": 7.70},
            {"min_cibil": 740, "max_cibil": 759, "roi": 7.60},
            {"min_cibil": 760, "max_cibil": 799, "roi": 7.55},
            {"min_cibil": 800, "max_cibil": float('inf'), "roi": 7.50}
        ],
        "interest_rate_lap": (9.20, 10.30),
        "processing_fee": (0.005, 0.015),
        "accept_cash_salary": False,
        "accept_surrogate_income": {"hl": False, "lap": True},
        "min_banking_vintage": {"hl": 0, "lap": 12},
        "conditions": "See detailed table for NRI, Realty, P-LAP, Top-Up, etc."
    },
    "Canara": {
        "min_cibil": 700,
        "min_income": 25000,
        "itr_for_business_profile": True,
        "foir": 0.75,
        "min_age": 21,
        "max_age": 75,
        "employment_types": ["Salaried", "Self-Employed"],
        "property_types": ["Residential", "Commercial", "Plots"],
        "approved_layout": True,
        "ltv_hl_slabs": [
            {"max_value": 3000000, "ltv": 0.9},
            {"max_value": 7500000, "ltv": 0.8},
            {"max_value": float('inf'), "ltv": 0.75}
        ],
        "ltv_lap": (0, 0.5),
        "loan_amount_range": (1000000, float('inf')),
        "tenure_range": {"hl": (5, 30), "lap": (7, 30)},
        "interest_rate_hl": (7.45, 7.6),
        "interest_rate_lap": (9.3, 10.3),
        "processing_fee": (0.005, 0.015),
        "accept_cash_salary": False,
        "accept_surrogate_income": False,
        "min_banking_vintage": 0,
        "conditions": "FOIR Details: <\u20B93L \u2192 35%, \u20B93	6L \u2192 45%, \u20B96	10L \u2192 50%, >\u20B910L \u2192 55%;\n\nLTV Details :\nLTV  HL \u2192 80%, LTV LAP \u2192 60%\n\nROI (HL) Account Salary:\nRack CIBIL 700 = 9.45%, CIBIL 701-729: 9.15%, CIBIL 730-749/NTC: 9.00%, CIBIL 750+: 8.75%\nROI (HL) Cash Salary:\nRack CIBIL 700: 9.50%,                                                                                     ASM Offer C700: 9.05%, 700	729: 8.40%, 730	749: 8.25%, 750+: 8.15%\n\nZSM Negotiated C700: 8.95%, 700	729: 8.30%, 730	749: 8.15%, 750	779: 8.05%, 780	799: 8.00%, 800+: 7.90%\n\nNSM Negotiated C700: 8.25%, 700	729: 8.20%, 730	749: 8.00%, 750	799: 7.95%, 800+: 7.90%\n\nSurrogate Income ROI Add-on: +0.15% to +0.50%\n\nTop-Up: Existing Customer +0.50%, BT+Top-Up +0.25%, Plot+Equity +0.50%\u21B5Bank Employee Rate: 7.90% (Repo + 2.40%)\n\nROI (LAP) Account Salary: 9.20	10.30% based on profile"
    },
    "HDFC": {
        "min_cibil": 700,
        "min_income": 20000,
        "itr_for_business_profile": False,
        "foir_slabs": [
            {"max_income": 300000, "foir": 0.35},
            {"max_income": 600000, "foir": 0.45},
            {"max_income": 1000000, "foir": 0.50},
            {"max_income": float('inf'), "foir": 0.55}
        ],
        "min_age": 21,
        "max_age": 63,
        "employment_types": ["Salaried", "Self-Employed"],
        "property_types": ["Residential", "Commercial", "Plots"],
        "approved_layout": True,
        "ltv_hl": (0, 0.8),
        "ltv_lap": (0, 0.6),
        "loan_amount_range": (300000, float('inf')),
        "tenure_range": {"hl": (7, 30), "lap": (7, 30)},
        "interest_rate_hl": (8.15, 9.60),
        "interest_rate_lap": (9.5, 10.5),
        "processing_fee": (0.005, 0.015),
        "accept_cash_salary": True,
        "accept_surrogate_income": True,
        "min_banking_vintage": 12,
        "conditions": "See detailed table for ROI slabs by CIBIL, salary mode, and negotiation."
    },
    "KVB": {
        "min_cibil": 700,
        "min_income": 30000,
        "itr_for_business_profile": True,
        "foir_slabs": [
            {"max_income": 100000, "foir": 0.60},
            {"max_income": 200000, "foir": 0.70},
            {"max_income": float('inf'), "foir": 0.80}
        ],
        "min_age": 21,
        "max_age": 60,
        "employment_types": ["Salaried", "Self-Employed"],
        "property_types": ["Residential", "Commercial"],
        "approved_layout": True,
        "ltv_hl": (0, 0.8),
        "ltv_lap": (0, 0.8),
        "loan_amount_range": (1000000, float('inf')),
        "tenure_range": {"hl": (7, 30), "lap": (7, 30)},
        "interest_rate_hl": 8.25,
        "interest_rate_lap": 10.25,
        "processing_fee": (0.01, 0.02),
        "accept_cash_salary": False,
        "accept_surrogate_income": False,
        "min_banking_vintage": 0,
        "conditions": "ROI (HL): CIBIL 700+ ->8.25; ROI (LAP): CIBIL 700+ ->10.25"
    },
    "Aditya Birla": {
        "min_cibil": 700,
        "min_income": 15000,
        "itr_for_business_profile": False,
        "foir_slabs": [
            {"max_income": 30000, "foir": 0.50},
            {"max_income": 40000, "foir": 0.60},
            {"max_income": float('inf'), "foir": 0.65}
        ],
        "min_age": 21,
        "max_age": 65,
        "employment_types": ["Salaried", "Self-Employed"],
        "property_types": ["Residential", "Commercial", "Plots"],
        "approved_layout": True,
        "ltv_hl": (0, 0.85),
        "ltv_lap": (0, 0.85),
        "loan_amount_range": (1000000, 10000000),
        "tenure_range": {"hl": (8, 25), "lap": (8, 25)},
        "interest_rate_hl": (9.35, 10),
        "interest_rate_lap": (10.35, 14),
        "processing_fee": (0.01, 0.02),
        "accept_cash_salary": True,
        "accept_surrogate_income": True,
        "min_banking_vintage": 12,
        "conditions": "FOIR: Cash salary max ₹30K→50%, Account salary: ₹15K→60%, >₹40K→65%; LTV: HL/LAP by property class; ROI: HL Account CIBIL>730→9.35%, 700–730→9.85–12%; HL Cash CIBIL>700→10.50%; LAP Account: 10.35–14%"
    },
    "ICICI": {
        "min_cibil": 720,
        "min_income": 35000,
        "itr_for_business_profile": True,
        "foir": {"salaried": 0.60, "business": 0.70},
        "min_age": 21,
        "max_age": 65,
        "employment_types": ["Salaried", "Self-Employed"],
        "property_types": ["Residential", "Commercial", "Plots"],
        "approved_layout": True,
        "ltv_hl": (0, 0.85),
        "ltv_lap": (0, 0.60),
        "loan_amount_range": (2000000, 70000000),
        "tenure_range": {"hl": (5, 30), "lap": (5, 30)},
        "interest_rate_hl": [
            {"min_cibil": 800, "roi": 8.10},
            {"min_cibil": 780, "roi": 8.30},
            {"min_cibil": 750, "roi": 8.50},
            {"min_cibil": 720, "roi": 8.75},
            {"min_cibil": 0, "roi": 8.75}
        ],
        "interest_rate_lap": [
            {"min_cibil": 780, "roi": 9.40},
            {"min_cibil": 0, "roi": 10.50}
        ],
        "processing_fee": (0.01, 0.02),
        "accept_cash_salary": False,
        "accept_surrogate_income": False,
        "min_banking_vintage": 0,
        "conditions": "FOIR: Salaried 60%, Business 70%. ROI (HL): Normal 8.75%. ₹1Cr+ Loan → CIBIL ≥800: 8.10%, 780–799: 8.30%, 750–779: 8.50%, 720–749: 8.75%. ROI (LAP): Normal 10.50%. ₹1Cr+ & CIBIL ≥780: 9.40%."
    }
}

def calculate_max_loan_amount(monthly_income, existing_emi, foir, interest_rate, tenure_years):
    # FOIR: (existing_emi + new_emi) / monthly_income <= foir
    # new_emi = (foir * monthly_income) - existing_emi
    max_emi = (foir * monthly_income) - existing_emi
    if max_emi <= 0:
        return 0, 0
    # Calculate max loan using EMI formula: EMI = [P * r * (1+r)^n] / [(1+r)^n-1]
    r = interest_rate / (12 * 100)  # monthly interest rate
    n = tenure_years * 12
    if r == 0:
        max_loan = max_emi * n
    else:
        max_loan = max_emi * ((1 + r) ** n - 1) / (r * (1 + r) ** n)
    return int(max_loan), int(max_emi)

def calculate_amortization_loan_amount(emi, tenure, roi):
    monthly_rate = roi / 12 / 100
    n_months = tenure * 12
    if monthly_rate == 0 or n_months == 0:
        return 0, 0
    loan_amount = emi * (1 - (1 + monthly_rate) ** -n_months) / monthly_rate
    total_interest = (emi * n_months) - loan_amount
    return loan_amount, total_interest

def calculate_eligible_tenure(current_age, maturity_age, min_tenure, max_tenure):
    tenure_by_age = maturity_age - current_age
    eligible_tenure = max(min_tenure, min(max_tenure, tenure_by_age))
    if eligible_tenure > 20:
        eligible_tenure = 20
    return eligible_tenure

def check_eligibility_policy(user_profile: dict):
    results = []
    coapp_missing_or_late_payment = user_profile.get("coapp_missing_or_late_payment", False)
    for bank, policy in BANK_POLICIES.items():
        eligible = True
        remarks = []
        min_cibil = policy.get("min_cibil", 0)
        cibil = user_profile.get("cibil_score", 0)
        income_mode = user_profile.get("income_mode", "").lower()
        # --- Cash salary strict validation (MUST be first) ---
        if income_mode == "cash":
            if not policy.get("accept_cash_salary", False):
                eligible = False
                remarks.append("Bank does not accept cash salary applicants.")
                results.append({
                    "bank": bank,
                    "eligible": "No",
                    "remarks": ", ".join(remarks),
                    "loan_amount": "0",
                    "emi": "0",
                    "tenure": "0 years",
                    "roi": "-",
                    "maturity_age": "-",
                    "total_interest": "\u20B90.00"
                })
                continue
            elif cibil < min_cibil:
                eligible = False
                remarks.append(f"CIBIL score below minimum required ({min_cibil}) for cash salary in {bank}")
                results.append({
                    "bank": bank,
                    "eligible": "No",
                    "remarks": ", ".join(remarks),
                    "loan_amount": "0",
                    "emi": "0",
                    "tenure": "0 years",
                    "roi": "-",
                    "maturity_age": "-",
                    "total_interest": "\u20B90.00"
                })
                continue
        else:
            if cibil < min_cibil:
                eligible = False
                remarks.append(f"CIBIL score below minimum required ({min_cibil}) for {bank}")
                results.append({
                    "bank": bank,
                    "eligible": "No",
                    "remarks": ", ".join(remarks),
                    "loan_amount": "0",
                    "emi": "0",
                    "tenure": "0 years",
                    "roi": "-",
                    "maturity_age": "-",
                    "total_interest": "\u20B90.00"
                })
                continue
        # FOIR
        foir_limit = 0.7  # default
        # If co-applicant income mode is business and ITR for 3 years is not filed, add a remark and mark as not eligible
        if user_profile.get("coapp_income_mode") == "business":
            if not user_profile.get("coapp_itr_3years", False):
                eligible = False
                remarks.append("Co-applicant business profile: ITR not filed for past 3 years. Not eligible as per standard policy.")
        # If main applicant income mode is business and ITR for 3 years is not filed, add a remark and mark as not eligible
        if user_profile.get("income_mode") == "business":
            if not user_profile.get("itr_3years", False):
                eligible = False
                remarks.append("Applicant business profile: ITR not filed for past 3 years. Not eligible as per standard policy.")
        if bank == "HDFC":
            annual_income = user_profile.get("monthly_income", 0) * 12
            foir_slabs = policy.get("foir_slabs", [])
            for slab in foir_slabs:
                if annual_income <= slab["max_income"]:
                    foir_limit = slab["foir"]
                    break
        elif isinstance(policy.get("foir"), tuple):
            foir_limit = policy["foir"][1]
        else:
            foir_limit = policy.get("foir", 0.7)
        # Handle FOIR as dict (e.g., ICICI)
        if isinstance(foir_limit, dict):
            emp_type = user_profile.get("employment_type", "salaried").lower()
            foir_limit = foir_limit.get(emp_type, 0.7)
        # --- Maturity Age Calculation: Use max(applicant age, coapplicant age) ---
        current_age = max(user_profile.get("age", 0), user_profile.get("coapp_age", 0) or 0)
        maturity_age = policy.get("max_age", 70)
        product = user_profile.get("product")
        loan_type = "hl" if product == "HL" else "lap"
        tenure_range = policy.get("tenure_range", {"hl": (5, 30), "lap": (7, 30)})
        min_tenure, max_tenure = tenure_range.get(loan_type, (5, 30))
        eligible_tenure = calculate_eligible_tenure(current_age, maturity_age, min_tenure, max_tenure)
        # Affordability calculation (max EMI)
        total_income = user_profile.get("monthly_income", 0)
        total_emi = user_profile.get("emi_after_loan", 0)
        max_emi = (foir_limit * total_income) - total_emi
        if max_emi <= 0:
            eligible = False
            remarks.append("No EMI affordability")
            max_emi = 0
        # ROI (policy-based for all banks, now using CIBIL and profile-based logic)
        roi = None
        product = user_profile.get("product")
        # --- Custom ROI logic for HDFC and Aditya Birla (cash/account salary) ---
        if bank == "HDFC":
            roi = get_hdfc_roi(cibil, income_mode, policy.get("conditions", ""))
        elif bank == "Aditya Birla":
            roi = get_adityabirla_roi(cibil, income_mode, policy.get("conditions", ""))
        elif bank in ("SBI", "ICICI"):
            # Use ROI slabs from interest_rate_hl for HL, interest_rate_lap for LAP
            if product == "HL":
                roi = get_roi_from_slabs(cibil, policy.get("interest_rate_hl", []))
            elif product == "LAP":
                roi = get_roi_from_slabs(cibil, policy.get("interest_rate_lap", []))
        elif bank == "Canara" and product == "HL":
            # Use new Canara ROI logic
            # Determine salary type: if income_mode is 'cash', use cash, else account
            salary_type = "cash" if income_mode == "cash" else "account"
            roi = get_canara_roi(cibil, salary_type, policy.get("conditions", ""))
        elif bank == "KVB":
            roi = get_kvb_roi(cibil, product, policy.get("conditions", ""))
        else:
            roi = get_dynamic_cibil_roi(cibil, policy.get("conditions", ""))
        # If not found, fallback to old logic
        if roi is None:
            slabs = None
            if product == "HL":
                slabs = policy.get("interest_rate_hl_slabs")
                if slabs:
                    matched = False
                    loan_amt, _ = calculate_amortization_loan_amount((foir_limit * user_profile.get("monthly_income", 0)) - user_profile.get("emi_after_loan", 0), calculate_eligible_tenure(user_profile.get("age", 0), policy.get("max_age", 70), policy.get("tenure_range", (5, 30))[0], policy.get("tenure_range", (5, 30))[1]), slabs[0]["roi"] if slabs else 8.0)
                    for slab in slabs:
                        if slab["min_amount"] <= loan_amt < slab["max_amount"]:
                            roi = slab["roi"]
                            matched = True
                            break
                    if not matched and len(slabs) > 0:
                        roi = slabs[-1]["roi"]
                else:
                    # Safely extract ROI value for interest_rate_hl
                    roi_val = policy.get("interest_rate_hl", (8.0, 9.0))
                    if isinstance(roi_val, (list, tuple)):
                        roi = roi_val[0]
                    elif isinstance(roi_val, (int, float)):
                        roi = roi_val
                    else:
                        roi = 8.0
            elif product == "LAP":
                cibil_slabs = policy.get("interest_rate_lap_slabs")
                if cibil_slabs:
                    loan_amt, _ = calculate_amortization_loan_amount((foir_limit * user_profile.get("monthly_income", 0)) - user_profile.get("emi_after_loan", 0), calculate_eligible_tenure(user_profile.get("age", 0), policy.get("max_age", 70), policy.get("tenure_range", (5, 30))[0], policy.get("tenure_range", (5, 30))[1]), cibil_slabs[0]["slabs"][0]["roi"] if cibil_slabs and cibil_slabs[0]["slabs"] else 10.0)
                    for cibil_slab in cibil_slabs:
                        min_cibil = cibil_slab.get("min_cibil", 0)
                        max_cibil = cibil_slab.get("max_cibil", 9999)
                        if min_cibil <= cibil <= max_cibil:
                            for slab in cibil_slab["slabs"]:
                                if slab["min_amount"] <= loan_amt < slab["max_amount"]:
                                    roi = slab["roi"]
                                    break
                            if roi is not None:
                                break
                    if roi is None:
                        roi = policy.get("interest_rate_lap", (10.0, 12.0))[0]
                else:
                    roi = policy.get("interest_rate_lap", (10.0, 12.0))[0]
            else:
                roi = 8.0  # fallback default
        if roi is None:
            roi = 8.0
        # Amortization calculation for eligible loan amount
        loan_amount, total_interest = calculate_amortization_loan_amount(max_emi, eligible_tenure, roi)
        # LTV check
        ltv_range = policy.get("ltv_hl", (0, 0.8)) if user_profile.get("product") == "HL" else policy.get("ltv_lap", (0, 0.6))
        ltv_limit = ltv_range[1] if isinstance(ltv_range, tuple) else ltv_range
        max_loan_ltv = int(user_profile.get("property_value", 0) * ltv_limit)
        final_max_loan = min(int(loan_amount), max_loan_ltv)
        # Now, after final_max_loan is set, do the slab matching and logging
        if user_profile.get("product") == "HL" and slabs:
            matched = False
            for slab in slabs:
                if slab["min_amount"] <= final_max_loan < slab["max_amount"]:
                    roi = slab["roi"]
                    matched = True
                    break
            if not matched and len(slabs) > 0:
                roi = slabs[-1]["roi"]
        elif user_profile.get("product") == "LAP" and slabs:
            for cibil_slab in slabs:
                min_cibil = cibil_slab.get("min_cibil", 0)
                max_cibil = cibil_slab.get("max_cibil", 9999)
                if min_cibil <= cibil <= max_cibil:
                    for slab in cibil_slab["slabs"]:
                        if slab["min_amount"] <= final_max_loan < slab["max_amount"]:
                            roi = slab["roi"]
                            break
                    if roi is not None:
                        break
            if roi is None and len(slabs) > 0:
                roi = slabs[-1]["slabs"][-1]["roi"]
        # FOIR eligibility
        foir_eligibility = total_emi / max(user_profile.get("monthly_income", 1), 1)
        if foir_eligibility > foir_limit:
            eligible = False
            remarks.append("FOIR not in range")
        # Co-applicant missing/late payment check
        if coapp_missing_or_late_payment:
            remarks.append("Since your co-applicant is having history of missing/late payment, this will need to be confirmed with bank side only after negotiation.")
        # Output
        results.append({
            "bank": bank,
            "eligible": "Yes" if eligible else "No",
            "remarks": ", ".join(remarks) if remarks else ("Eligible" if eligible else "Not eligible"),
            "loan_amount": str(final_max_loan if eligible else 0),
            "emi": str(int(max_emi) if eligible else 0),
            "tenure": f"{eligible_tenure} years" if eligible else "0 years",
            "roi": str(roi) if eligible else "-",
            "maturity_age": str(current_age + eligible_tenure) if eligible else "-",
            "total_interest": f"₹{total_interest:,.2f}" if eligible else "₹0.00"
        })
    return {"eligible_banks": results}



@app.post("/policy-eligibility-nested")
def policy_eligibility_nested(data: LoanFormInput = Body(...)):
    """
    Accepts nested JSON input from the new loan form and returns policy eligibility.
    """
    flat = flatten_loan_form_input(data)
    return check_eligibility_policy(flat)

@app.post("/full-eligibility")
def full_eligibility(data: LoanFormInput = Body(...)):
    """
    Accepts nested JSON input, calculates income-wise (affordability) eligibility, applies all bank policies, and returns amortization schedule for each bank using Gemini model and the full policy table.
    """
    # Enforce applicant phone_number presence before any processing
    applicant_phone = getattr(data.applicant, 'phone_number', None)
    if not applicant_phone or not str(applicant_phone).strip():
        return JSONResponse(status_code=400, content={"error": "Applicant phone_number is required."})
    flat = flatten_loan_form_input(data)
    # Use repayment issue from top-level LoanFormInput
    loan_repayment_issue = getattr(data, 'loan_repayment_issue', False)
    results = []
    for bank, policy in BANK_POLICIES.items():
        eligible = True
        remarks = []
        min_cibil = policy.get("min_cibil", 0)
        cibil = flat.get("cibil_score", 0)
        income_mode = flat.get("income_mode", "").lower()
        # --- Cash salary strict validation (MUST match check_eligibility_policy) ---
        if income_mode == "cash":
            if not policy.get("accept_cash_salary", False):
                eligible = False
                remarks.append("Bank does not accept cash salary applicants.")
                results.append({
                    "bank": bank,
                    "eligible": "No",
                    "remarks": ", ".join(remarks),
                    "loan_amount": "0",
                    "emi": "0",
                    "tenure": "0 years",
                    "roi": "-",
                    "maturity_age": "-",
                    "total_interest": "₹0.00"
                })
                continue
            elif cibil < min_cibil:
                eligible = False
                remarks.append(f"CIBIL score below minimum required ({min_cibil}) for cash salary in {bank}")
                results.append({
                    "bank": bank,
                    "eligible": "No",
                    "remarks": ", ".join(remarks),
                    "loan_amount": "0",
                    "emi": "0",
                    "tenure": "0 years",
                    "roi": "-",
                    "maturity_age": "-",
                    "total_interest": "₹0.00"
                })
                continue
        else:
            if cibil < min_cibil:
                eligible = False
                remarks.append(f"CIBIL score below minimum required ({min_cibil}) for {bank}")
                results.append({
                    "bank": bank,
                    "eligible": "No",
                    "remarks": ", ".join(remarks),
                    "loan_amount": "0",
                    "emi": "0",
                    "tenure": "0 years",
                    "roi": "-",
                    "maturity_age": "-",
                    "total_interest": "₹0.00"
                })
                continue
        # FOIR
        foir_limit = 0.7  # default
        if bank == "HDFC":
            annual_income = flat.get("monthly_income", 0) * 12
            foir_slabs = policy.get("foir_slabs", [])
            for slab in foir_slabs:
                if annual_income <= slab["max_income"]:
                    foir_limit = slab["foir"]
                    break
        elif isinstance(policy.get("foir"), tuple):
            foir_limit = policy["foir"][1]
        else:
            foir_limit = policy.get("foir", 0.7)
        # Handle FOIR as dict (e.g., ICICI)
        if isinstance(foir_limit, dict):
            emp_type = flat.get("employment_type", "salaried").lower()
            foir_limit = foir_limit.get(emp_type, 0.7)
        # --- Maturity Age Calculation: Use max(applicant age, coapplicant age) ---
        current_age = max(flat.get("age", 0), flat.get("coapp_age", 0) or 0)
        maturity_age = policy.get("max_age", 70)
        product = flat.get("product")
        loan_type = "hl" if product == "HL" else "lap"
        tenure_range = policy.get("tenure_range", {"hl": (5, 30), "lap": (7, 30)})
        min_tenure, max_tenure = tenure_range.get(loan_type, (5, 30))
        eligible_tenure = calculate_eligible_tenure(current_age, maturity_age, min_tenure, max_tenure)
        total_income = flat.get("monthly_income", 0)
        total_emi = flat.get("emi_after_loan", 0)
        max_emi = (foir_limit * total_income) - total_emi
        if max_emi <= 0:
            eligible = False
            remarks.append("No EMI affordability")
            max_emi = 0
        roi = None
        product = flat.get("product")
        # --- Custom ROI logic for HDFC and Aditya Birla (cash/account salary) ---
        if bank == "HDFC":
            income_mode = flat.get("income_mode", "").lower()
            roi = get_hdfc_roi(cibil, income_mode, policy.get("conditions", ""))
        elif bank == "Aditya Birla":
            income_mode = flat.get("income_mode", "").lower()
            roi = get_adityabirla_roi(cibil, income_mode, policy.get("conditions", ""))
        elif bank in ("SBI", "ICICI"):
            if product == "HL":
                roi = get_roi_from_slabs(cibil, policy.get("interest_rate_hl", []))
            elif product == "LAP":
                roi = get_roi_from_slabs(cibil, policy.get("interest_rate_lap", []))
        elif bank == "Canara" and product == "HL":
            salary_type = "cash" if income_mode == "cash" else "account"
            roi = get_canara_roi(cibil, salary_type, policy.get("conditions", ""))
        elif bank == "KVB":
            roi = get_kvb_roi(cibil, product, policy.get("conditions", ""))
        else:
            roi = get_dynamic_cibil_roi(cibil, policy.get("conditions", ""))
        if roi is None:
            slabs = None
            if product == "HL":
                slabs = policy.get("interest_rate_hl_slabs")
                if slabs:
                    matched = False
                    loan_amt, _ = calculate_amortization_loan_amount((foir_limit * flat.get("monthly_income", 0)) - flat.get("emi_after_loan", 0), calculate_eligible_tenure(flat.get("age", 0), policy.get("max_age", 70), policy.get("tenure_range", (5, 30))[0], policy.get("tenure_range", (5, 30))[1]), slabs[0]["roi"] if slabs else 8.0)
                    for slab in slabs:
                        if slab["min_amount"] <= loan_amt < slab["max_amount"]:
                            roi = slab["roi"]
                            matched = True
                            break
                    if not matched and len(slabs) > 0:
                        roi = slabs[-1]["roi"]
            elif product == "LAP":
                cibil_slabs = policy.get("interest_rate_lap_slabs")
                if cibil_slabs:
                    loan_amt, _ = calculate_amortization_loan_amount((foir_limit * flat.get("monthly_income", 0)) - flat.get("emi_after_loan", 0), calculate_eligible_tenure(flat.get("age", 0), policy.get("max_age", 70), policy.get("tenure_range", (5, 30))[0], policy.get("tenure_range", (5, 30))[1]), cibil_slabs[0]["slabs"][0]["roi"] if cibil_slabs and cibil_slabs[0]["slabs"] else 10.0)
                    for cibil_slab in cibil_slabs:
                        min_cibil = cibil_slab.get("min_cibil", 0)
                        max_cibil = cibil_slab.get("max_cibil", 9999)
                        if min_cibil <= cibil <= max_cibil:
                            for slab in cibil_slab["slabs"]:
                                if slab["min_amount"] <= loan_amt < slab["max_amount"]:
                                    roi = slab["roi"]
                                    break
                            if roi is not None:
                                break
                    if roi is None:
                        roi = policy.get("interest_rate_lap", (10.0, 12.0))[0]
                else:
                    roi = policy.get("interest_rate_lap", (10.0, 12.0))[0]
            else:
                roi = 8.0  # fallback default
        # Ensure roi is a float, not a dict
        if isinstance(roi, dict):
            roi = roi.get("roi", 8.0)
        if isinstance(roi, (list, tuple)) and len(roi) > 0:
            roi = roi[0] if isinstance(roi[0], (int, float)) else 8.0
        if roi is None:
            roi = 8.0
        loan_amount, total_interest = calculate_amortization_loan_amount(max_emi, eligible_tenure, roi)
        ltv_range = policy.get("ltv_hl", (0, 0.8)) if flat.get("product") == "HL" else policy.get("ltv_lap", (0, 0.6))
        ltv_limit = ltv_range[1] if isinstance(ltv_range, tuple) else ltv_range
        max_loan_ltv = int(flat.get("property_value", 0) * ltv_limit)
        requested_amount = flat.get("loan_amount_requested", 0)
        if requested_amount > 0:
            final_max_loan = min(int(loan_amount), max_loan_ltv, requested_amount)
        else:
            final_max_loan = min(int(loan_amount), max_loan_ltv)
        recalc_emi = max_emi
        recalc_total_interest = total_interest
        if final_max_loan < int(loan_amount):
            monthly_rate = roi / 12 / 100
            n_months = eligible_tenure * 12
            if monthly_rate > 0 and n_months > 0:
                recalc_emi = final_max_loan * monthly_rate * (1 + monthly_rate) ** n_months / ((1 + monthly_rate) ** n_months - 1)
                recalc_total_interest = (recalc_emi * n_months) - final_max_loan
            else:
                recalc_emi = 0
                recalc_total_interest = 0
        ltv = final_max_loan / max(flat.get("property_value", 1), 1)
        if isinstance(ltv_range, tuple):
            if not (ltv_range[0] <= ltv <= ltv_range[1]):
                eligible = False
                remarks.append("LTV not in range")
        else:
            if ltv > ltv_range:
                eligible = False
                remarks.append("LTV too high")
        foir = total_emi / max(flat.get("monthly_income", 1), 1)
        if foir > foir_limit:
            eligible = False
            remarks.append("FOIR not in range")
        # --- ITR for business profile validation ---
        # If applicant is business type and bank requires ITR, check applicant.itr_3years
        requires_itr = policy.get("itr_for_business_profile", False)
        applicant_income_type = flat.get("employment_type", "").lower()
        applicant_itr_3years = getattr(data.applicant, "itr_3years", None)
        # Fix: Accept both boolean true and string 'true' for ITR field, now more robust
        def is_true(val):
            return val is True or str(val).lower() in ("true", "yes", "1")
        if requires_itr and applicant_income_type == "business":
            if not is_true(applicant_itr_3years):
                eligible = False
                remarks.append("Applicant business profile: ITR for past 3 years required as per bank policy.")
        # If co-applicant is business type and bank requires ITR, check coapp_itr_3years
        coapp_income_type = flat.get("coapp_income_mode", "").lower()
        coapp_itr_3years = flat.get("coapp_itr_3years", None)
        if requires_itr and coapp_income_type == "business":
            if not is_true(coapp_itr_3years):
                eligible = True
                remarks.append("Co-applicant business profile: ITR for past 3 years required as per bank policy.")
        # Add loan repayment issue remark if present
        if loan_repayment_issue:
            remarks.append("You have loan repayment issue, it is banks internal call. Please try to negotiate with banks.")
        results.append({
            "bank": bank,
            "eligible": "Yes" if eligible else "No",
            "remarks": ", ".join(remarks) if remarks else ("Eligible" if eligible else "Not eligible"),
            "loan_amount": str(final_max_loan if eligible else 0),
            "emi": str(int(recalc_emi) if eligible else 0),
            "tenure": f"{eligible_tenure} years" if eligible else "0 years",
            "roi": str(roi) if eligible else "-",
            "maturity_age": str(current_age + eligible_tenure) if eligible else "-",
            "total_interest": f"₹{recalc_total_interest:,.2f}" if eligible else "₹0.00"
        })
    # Map to frontend expected structure
    mapped_banks = []
    for b in results:
        # Defensive: always convert eligible to string and compare case-insensitively
        eligible_str = str(b.get("eligible", "No")).strip().lower()
        mapped_banks.append({
            "bank": b.get("bank", ""),
            "eligible": "Yes" if eligible_str == "yes" else "No",
            "remarks": b.get("remarks", ""),
            "loan_amount": b.get("loan_amount", ""),
            "emi": b.get("emi", ""),
            "tenure": b.get("tenure", ""),
            "roi": b.get("roi", "-")
            # Removed 'amortization_schedule' field to eliminate 'Show EMI schedule' from report
        })
    # After all calculations and before returning the response:
    recommendations = generate_recommendations(flat)
    response_obj = {
        "name": flat.get("name", ""),
        "type_of_loan": flat.get("type_of_loan", ""),
        "address": flat.get("address", ""),
        "loan_amount": str(flat.get("loan_amount_requested", "")),
        "date": "",
        "eligible_banks": mapped_banks,
        "enhancement_insights": recommendations,
        "total_emi": str(flat.get("emi_after_loan", 0))
    }
    # Save user input and output to DB
    save_report(data.dict(), response_obj)
    return response_obj

class AmortizationInput(BaseModel):
    emi: float
    tenure: int  # in years
    roi: float   # annual interest rate (%)

class AmortizationOutput(BaseModel):
    eligible_loan_amount: str
    total_interest_payable: str

@app.post("/amortization-calculation", response_model=AmortizationOutput)
def amortization_calculation(data: AmortizationInput = Body(...)):
    """
    Calculate eligible loan amount and total interest payable using amortization method.
    Input: { emi, tenure, roi }
    Output: { eligible_loan_amount, total_interest_payable }
    """
    try:
        emi = float(data.emi)
        roi = float(data.roi)
        tenure = int(data.tenure)
        monthly_rate = roi / 12 / 100
        n_months = tenure * 12
        if monthly_rate == 0 or n_months == 0:
            return JSONResponse(status_code=400, content={"error": "ROI and tenure must be greater than zero."})
        # Amortization formula
        loan_amount = emi * (1 - (1 + monthly_rate) ** -n_months) / monthly_rate
        total_interest = (emi * n_months) - loan_amount
        return AmortizationOutput(
            eligible_loan_amount=f"₹{loan_amount:,.2f}",
            total_interest_payable=f"₹{total_interest:,.2f}"
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

class TenureInput(BaseModel):
    """
    Input model for tenure calculation endpoint.
    Fields:
      - current_age: Applicant's current age (years)
      - maturity_age: Maximum allowed age at loan maturity (years)
      - min_tenure: Minimum tenure allowed by bank (years)
      - max_tenure: Maximum tenure allowed by bank (years)
    """
    current_age: int = Field(..., description="Applicant's current age in years.")
    maturity_age: int = Field(..., description="Maximum age at loan maturity as per bank policy.")
    min_tenure: int = Field(..., description="Minimum tenure allowed by the bank (in years).")
    max_tenure: int = Field(..., description="Maximum tenure allowed by the bank (in years).")

class TenureOutput(BaseModel):
    """
    Output model for tenure calculation endpoint.
    Fields:
      - eligible_tenure: Eligible tenure in years (e.g., '15 years')
    """
    eligible_tenure: str = Field(..., description="Eligible tenure in years, e.g., '15 years'.")

@app.post("/tenure-calculation", response_model=TenureOutput, summary="Calculate eligible tenure", tags=["Calculations"])
def tenure_calculation(data: TenureInput = Body(..., example={
    "current_age": 35,
    "maturity_age": 60,
    "min_tenure": 5,
    "max_tenure": 20
})):
    """
    Calculate eligible tenure based on applicant's age and bank's maturity/tenure limits.\n\n
    - **current_age**: Applicant's current age in years.
    - **maturity_age**: Maximum age at loan maturity as per bank policy.
    - **min_tenure**: Minimum tenure allowed by the bank (in years).
    - **max_tenure**: Maximum tenure allowed by the bank (in years).

    Returns the eligible tenure in years, clamped between min_tenure and max_tenure, and not exceeding the difference between maturity_age and current_age. If the calculated eligible tenure is more than 20, it will be capped at 20 years.
    """
    try:
        tenure_by_age = data.maturity_age - data.current_age
        eligible_tenure = max(data.min_tenure, min(data.max_tenure, tenure_by_age))
        if eligible_tenure > 20:
            eligible_tenure = 20
        return TenureOutput(eligible_tenure=f"{eligible_tenure} years")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# --- Database setup ---
DB_PATH = 'loan_reports.db'
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        user_input TEXT,
        output TEXT
    )''')
    conn.commit()
    conn.close()
init_db()

def save_report(user_input, output):
    # Always use applicant's phone_number for storage and enforce presence
    phone_number = None
    if isinstance(user_input, dict):
        # Get phone_number only from applicant
        if 'applicant' in user_input and isinstance(user_input['applicant'], dict):
            phone_number = user_input['applicant'].get('phone_number')
            if not phone_number or not str(phone_number).strip():
                raise ValueError("Applicant phone_number is required and missing. Report not saved.")
            user_input['phone_number'] = phone_number
        else:
            raise ValueError("Applicant object missing in user_input. Report not saved.")
    else:
        raise ValueError("user_input must be a dict. Report not saved.")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO reports (user_input, output) VALUES (?, ?)', (json.dumps(user_input), json.dumps(output)))
    conn.commit()
    report_id = c.lastrowid
    conn.close()
    return report_id

def get_report(report_id=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if report_id:
        c.execute('SELECT id, created_at, user_input, output FROM reports WHERE id=?', (report_id,))
    else:
        c.execute('SELECT id, created_at, user_input, output FROM reports ORDER BY id DESC LIMIT 1')
    row = c.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "created_at": row[1], "user_input": json.loads(row[2]), "output": json.loads(row[3])}
    return None

@app.get("/report", response_class=HTMLResponse)
def report_page():
    return open("report.html", "r", encoding="utf-8").read()

@app.get("/api/report/latest")
def get_latest_report():
    report = get_report()
    if report:
        return report
    return {"error": "No report found"}

@app.get("/api/report/all")
def get_all_reports():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, created_at, user_input, output FROM reports ORDER BY id DESC')
    rows = c.fetchall()
    conn.close()
    result = []
    for row in rows:
        result.append({
            "id": row[0],
            "created_at": row[1],
            "user_input": json.loads(row[2]),
            "output": json.loads(row[3])
        })
    return result

@app.get("/api/report/byid/{report_id}")
def get_report_by_id(report_id: int):
    report = get_report(report_id)
    if report:
        return report
    return {"error": "No report found"}

def get_hdfc_roi(cibil, income_mode, conditions):
    # Placeholder logic for HDFC ROI selection (customize as needed)
    # You can expand this logic based on actual HDFC ROI slabs if available
    if income_mode == "cash":
        if cibil >= 700:
            return 9.5
    else:
        if cibil >= 730:
            return 8.15
        elif 700 <= cibil < 730:
            return 9.0
    return 9.5

def get_adityabirla_roi(cibil, income_mode, conditions):
    """
    Parse Aditya Birla ROI for HL/LAP based on income_mode (cash/account) and CIBIL.
    """
    if income_mode == 'cash':
        if cibil >= 700:
            return 10.5
        else:
            return None
    # For account salary, parse from conditions if possible
    if cibil > 730:
        return 9.35
    elif 700 <= cibil <= 730:
        return 9.85
    return 10.0  # fallback

def get_canara_roi(cibil, income_mode, conditions):
    """
    Parse Canara Bank ROI slabs from the 'conditions' field and select the correct ROI
    based on CIBIL and salary type (account/cash).
    """
    # Normalize and split the conditions
    cond = conditions.replace("\u2013", "-").replace("\u2014", "-").replace("\u2192", ":").replace("\u21B5", "\n")
    # Account Salary slabs
    account_slabs = []
    cash_slabs = []
    # Parse Account Salary ROI
    account_match = re.search(r'ROI \(HL\) Account Salary:(.*?)(?:ROI|$)', cond, re.DOTALL)
    if account_match:
        lines = account_match.group(1).split("\n")
        for line in lines:
            m = re.search(r'CIBIL (\d+)(?:\s*[-–]\s*(\d+))?(?:/NTC)?: ([\d.]+)%', line)
            if m:
                min_cibil = int(m.group(1))
                max_cibil = int(m.group(2)) if m.group(2) else min_cibil
                roi = float(m.group(3))
                account_slabs.append({"min_cibil": min_cibil, "max_cibil": max_cibil, "roi": roi})
            else:
                m = re.search(r'CIBIL (\d+)\+: ([\d.]+)%', line)
                if m:
                    min_cibil = int(m.group(1))
                    roi = float(m.group(2))
                    account_slabs.append({"min_cibil": min_cibil, "max_cibil": 900, "roi": roi})
    # Parse Cash Salary ROI
    cash_match = re.search(r'ROI \(HL\) Cash Salary:(.*?)(?:ROI|$)', cond, re.DOTALL)
    if cash_match:
        lines = cash_match.group(1).split("\n")
        for line in lines:
            m = re.search(r'CIBIL (\d+)(?:\s*[-–]\s*(\d+))?(?:/NTC)?: ([\d.]+)%', line)
            if m:
                min_cibil = int(m.group(1))
                max_cibil = int(m.group(2)) if m.group(2) else min_cibil
                roi = float(m.group(3))
                cash_slabs.append({"min_cibil": min_cibil, "max_cibil": max_cibil, "roi": roi})
            else:
                m = re.search(r'CIBIL (\d+)\+: ([\d.]+)%', line)
                if m:
                    min_cibil = int(m.group(1))
                    roi = float(m.group(2))
                    cash_slabs.append({"min_cibil": min_cibil, "max_cibil": 900, "roi": roi})
    # Select correct slab
    slabs = account_slabs if income_mode == "account" else cash_slabs
    if not slabs:
        return None
    for slab in slabs:
        if slab["min_cibil"] <= cibil <= slab["max_cibil"]:
            return slab["roi"]
    return slabs[-1]["roi"] if slabs else None

def get_kvb_roi(cibil, product, conditions):
    """
    Parse KVB ROI slabs from the 'conditions' field and select the correct ROI
    based on CIBIL and product (HL/LAP).
    """
    cond = conditions.replace("\u2013", "-").replace("\u2014", "-").replace("\u2192", ":").replace(",", ".")
    # ROI (HL): CIBIL 700+ ->8.25; ROI (LAP): CIBIL 700+ ->10.25
    if product == "HL":
        m = re.search(r'ROI \(HL\): CIBIL (\d+)\+ *[-:>]+ *([\d.]+)', cond)
        if m:
            min_cibil = int(m.group(1))
            roi = float(m.group(2))
            if cibil >= min_cibil:
                return roi
    elif product == "LAP":
        m = re.search(r'ROI \(LAP\): CIBIL (\d+)\+ *[-:>]+ *([\d.]+)', cond)
        if m:
            min_cibil = int(m.group(1))
            roi = float(m.group(2))
            if cibil >= min_cibil:
                return roi
    return None

def get_roi_from_slabs(cibil, slabs):
    """
    Given a CIBIL score and a list of slabs (each with min_cibil, max_cibil, roi),
    return the correct ROI for the user's CIBIL score. If no match, return the last slab's ROI.
    """
    if not slabs or not isinstance(slabs, list):
        return None
    for slab in slabs:
        min_cibil = slab.get("min_cibil", 0)
        max_cibil = slab.get("max_cibil", float('inf'))
        if min_cibil <= cibil <= max_cibil:
            return slab.get("roi")
    # fallback: last slab's ROI
    return slabs[-1].get("roi")

def generate_recommendations(flat):
    """
    Generate recommendation texts based on user profile for enhancement insights.
    """
    recs = []
    cibil = flat.get("cibil_score", 0)
    income = flat.get("monthly_income", 0)
    total_emi = flat.get("emi_after_loan", 0)
    coapp_income = flat.get("coapp_income_amount", 0)
    coapp_exists = coapp_income > 0
    property_value = flat.get("property_value", 0)
    requested_loan = flat.get("loan_amount_requested", 0)
    # CIBIL based
    if cibil < 750:
        recs.append("Improve your CIBIL score above 750 for better rates and eligibility.")
    # EMI/FOIR based
    if total_emi > 0.5 * income:
        recs.append("Reduce your existing EMIs to improve loan eligibility.")
    # Co-applicant
    if not coapp_exists:
        recs.append("Adding a co-applicant with stable income can enhance your eligibility.")
    # Property value vs loan
    if requested_loan > 0 and property_value > 0 and requested_loan > 0.8 * property_value:
        recs.append("Consider requesting a lower loan amount or increasing property value for better LTV.")
    # Income
    if income < 30000:
        recs.append("Increase your monthly income to improve eligibility for more banks.")
    # ITR
    if flat.get("employment_type", "").lower() == "business" and not flat.get("itr_3years", False):
        recs.append("File ITR for the last 3 years to be eligible for more banks.")
    # Default
    if not recs:
        recs.append("Your profile is strong. You are eligible for most banks. Contact us for negotiation support.")
    return recs
