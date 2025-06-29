from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
import google.generativeai as genai
import json
import re
from fastapi import Request
from fastapi.responses import JSONResponse

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
    # New fields for UI integration
    salary_type: Optional[str] = None  # 'salary' or 'business'
    monthly_income: Optional[int] = 0
    existing_emi: Optional[int] = 0
    missing_or_late_payment: Optional[bool] = None

class LoanFormInput(BaseModel):
    applicant: Applicant
    loan_details: LoanDetails
    co_applicant: Optional[CoApplicant] = None

def flatten_loan_form_input(data: LoanFormInput) -> Dict[str, Any]:
    # Map nested input to flat structure for policy eligibility
    applicant = data.applicant
    co_app = data.co_applicant or CoApplicant()
    loan = data.loan_details
    # Calculate total income and EMI
    # Use new co-applicant fields if present
    co_income = co_app.monthly_income if hasattr(co_app, 'monthly_income') and co_app.monthly_income else (co_app.income_amount or 0)
    co_emi = co_app.existing_emi if hasattr(co_app, 'existing_emi') and co_app.existing_emi else (co_app.existing_loan.monthly_emi if co_app.existing_loan else 0)
    total_income = applicant.income_amount + co_income
    total_emi = (applicant.existing_loan.monthly_emi or 0) + co_emi
    requested_loan_amount = loan.loan_amount_requested if hasattr(loan, 'loan_amount_requested') and loan.loan_amount_requested is not None else 0
    # Pass through new co-applicant fields for use in eligibility logic
    return {
        "product": "HL" if loan.type == "home_loan" else "LAP",
        "cibil_score": applicant.cibil_score,
        "employment_type": applicant.income_type.capitalize(),
        "monthly_income": total_income,
        "emi_after_loan": total_emi,
        "age": applicant.age,
        "property_type": "Residential" if loan.type == "home_loan" else "Commercial",
        "property_approval": loan.property_approval,
        "property_value": loan.property_value,
        "loan_amount_requested": requested_loan_amount,
        "loan_tenure_requested": 20,  # Default or add to form if needed
        "is_woman": False,  # Add logic if needed
        # Co-applicant fields for advanced logic
        "coapp_salary_type": getattr(co_app, 'salary_type', None),
        "coapp_monthly_income": co_income,
        "coapp_existing_emi": co_emi,
        "coapp_missing_or_late_payment": getattr(co_app, 'missing_or_late_payment', None)
    }

@app.post("/recommend-loan", response_model=RecommendationResponse)
def recommend_loan(user_profile: UserProfile = Body(...)):
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



# Bank policy table as a Python dictionary
BANK_POLICIES = {
    "SBI": {
        "min_cibil": 700,
        "min_income": 25000,
        "foir": (0.55, 0.70),
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
        "tenure_range": (5, 30),
        "interest_rate_hl": (7.5, 7.8),
        "interest_rate_lap": (9.2, 10.3),
        "processing_fee": (0.005, 0.015),
        "accept_cash_salary": False,
        "accept_surrogate_income": False,
        "min_banking_vintage": 0,
        "conditions": "FOIR: ₹3–5L → 55%, ₹5–8L → 60%, ₹8–10L → 65%, >₹10L → 70% | ROI HL: 7.25–7.80% (profile-based) | ROI LAP: 9.20–10.30% | LTV HL (Salaried): ₹0–30L → 90%, ₹30–75L → 80%, >₹75L → 75% | Top-Up Loan: Same as HL | NRI HL – Salaried: ≤₹75L → 80%, >₹75L → 75% | NRI HL – Self-Employed: ≤₹75L → 75%, >₹75L → 70% | Realty HL (Salaried): ≤₹30L → 75%, ₹30–75L → 70%, >₹75L → 60% | P-LAP: ≤₹1Cr → 65%, ₹1–7.5Cr → 60%"
    },
    "Canara": {
        "min_cibil": 700,
        "min_income": 25000,
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
        "tenure_range": (5, 30),
        "interest_rate_hl_slabs": [
            {"min_amount": 0, "max_amount": 10000000, "roi": 7.55},
            {"min_amount": 10000000, "max_amount": 25000000, "roi": 7.45}
        ],
        "interest_rate_lap_slabs": [
            {"min_cibil": 750, "slabs": [
                {"min_amount": 1000000, "max_amount": 20000000, "roi": 9.5},
                {"min_amount": 20000000, "max_amount": 30000000, "roi": 9.4},
                {"min_amount": 30000000, "max_amount": float('inf'), "roi": 9.3}
            ]},
            {"min_cibil": 700, "max_cibil": 749, "slabs": [
                {"min_amount": 1000000, "max_amount": 20000000, "roi": 9.75},
                {"min_amount": 20000000, "max_amount": 30000000, "roi": 9.65},
                {"min_amount": 30000000, "max_amount": float('inf'), "roi": 9.55}
            ]},
            {"min_cibil": 0, "max_cibil": 699, "slabs": [
                {"min_amount": 1000000, "max_amount": 20000000, "roi": 10.3},
                {"min_amount": 20000000, "max_amount": 30000000, "roi": 10.2},
                {"min_amount": 30000000, "max_amount": float('inf'), "roi": 10.1}
            ]}
        ],
        "processing_fee": (0.005, 0.015),
        "accept_cash_salary": False,
        "accept_surrogate_income": False,
        "min_banking_vintage": 0,
        "conditions": "FOIR Rule: Max 75%, with NTH ≥ 25% or ₹10K (whichever higher); ROI HL: Best case – 7.55%, ₹1L–2.5Cr – 7.45%; ROI LAP: CIBIL 750+ → ₹1L–₹2Cr: 9.50%, ₹2Cr–₹3Cr: 9.40%, >₹3Cr: 9.30%; CIBIL 700–750 → ₹1L–₹2Cr: 9.75%, ₹2Cr–₹3Cr: 9.65%, >₹3Cr: 9.55%; CIBIL <700 → ₹1L–₹2Cr: 10.3%, ₹2Cr–₹3Cr: 10.2%, >₹3Cr: 10.1%; HL LTV: <₹30L → 90%, ₹30–75L → 80%, >₹75L or property >10yrs → 75%; LAP LTV: Fixed 50%; Special HL ₹10L allowed via current year VAO income certificate, manager approval required"
    },
    "HDFC": {
        "min_cibil": 700,
        "min_income": 20000,
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
        "tenure_range": (7, 30),
        "interest_rate_hl": (8.15, 9.6),
        "interest_rate_lap": (9.5, 10.5),
        "processing_fee": (0.005, 0.015),
        "accept_cash_salary": True,
        "accept_surrogate_income": True,
        "min_banking_vintage": 12,
        "conditions": "FOIR Details: <₹3L → 35%, ₹3–6L → 45%, ₹6–10L → 50%, >₹10L → 55%; LTV Details: LTV HL → 80%, LTV LAP → 60%; ROI (HL): Rack – <700: 9.45%, 700–729: 9.15%, 730–749/NTC: 9.00%, 750+: 8.75%; ASM Offer – <700: 9.05%, 700–729: 8.40%, 730–749: 8.25%, 750+: 8.15%; ZSM Negotiated – <700: 8.95%, 700–729: 8.30%, 730–749: 8.15%, 750–779: 8.05%, 780–799: 8.00%, 800+: 7.90%; NSM Negotiated – <700: 8.25%, 700–729: 8.20%, 730–749: 8.00%, 750–799: 7.95%, 800+: 7.90%; Surrogate Income ROI Add-on: +0.15% to +0.50%; Top-Up: Existing Customer +0.50%, BT+Top-Up +0.25%, Plot+Equity +0.50%; Bank Employee Rate: 7.90% (Repo + 2.40%); ROI (LAP): 9.20–10.30% based on profile"
    },
    "KVB": {
        "min_cibil": 750,
        "min_income": 30000,
        "foir": 0.6,
        "min_age": 21,
        "max_age": 60,
        "employment_types": ["Salaried", "Self-Employed"],
        "property_types": ["Residential", "Commercial"],
        "approved_layout": True,
        "ltv_hl": (0, 0.8),
        "ltv_lap": (0, 0.8),
        "loan_amount_range": (1000000, float('inf')),
        "tenure_range": (7, 30),
        "interest_rate_hl": (8.5, 8.5),
        "interest_rate_lap": (11, 11),
        "processing_fee": (0.01, 0.02),
        "accept_cash_salary": False,
        "accept_surrogate_income": False,
        "min_banking_vintage": 0,
        "conditions": "FOIR Details: Cash salary capped at ₹30K → FOIR 50%; Account salary: ₹15K → FOIR 60%, >₹40K (i.e. >₹5L p.a.) → FOIR 65%; LTV Details: HL & LAP → Based on property class; max ~50% for LAP; ROI Details: ROI (HL): CIBIL >730 → 9.35%; CIBIL 700–730 → 9.85–12% (varies by property class); Property Types: Class 1 → Fully approved (best ROI), Class 2 → One approval missing, Class 3 → No approval, Class 4 → Irregular/encroached/<8ft road (high risk); ROI (LAP): 10.35–14% based on property class"
    },
    "Aditya Birla": {
        "min_cibil": 700,
        "min_income": 15000,
        "foir": 0.5,  # Default FOIR, but see conditions for slabs
        "min_age": 21,
        "max_age": 65,
        "employment_types": ["Salaried", "Self-Employed"],
        "property_types": ["Residential", "Commercial", "Plots"],
        "approved_layout": True,
        "ltv_hl": (0, 0.85),  # 70-85% for HL
        "ltv_lap": (0, 0.85), # 70-85% for LAP
        "loan_amount_range": (1000000, 10000000),  # 10L to 1Cr
        "tenure_range": (8, 25),
        "interest_rate_hl": (9.35, 10),  # HL: 9.35-10%
        "interest_rate_lap": (10.35, 14), # LAP: 10.35-14%
        "processing_fee": (0.01, 0.02),
        "accept_cash_salary": True,
        "accept_surrogate_income": True,
        "min_banking_vintage": 12,  # 1 year
        "conditions": "FOIR Details: Cash salary capped at ₹30K → FOIR 50%; Account salary: ₹15K → FOIR 60%, >₹40K (i.e. >₹5L p.a.) → FOIR 65%; LTV Details: HL & LAP → Based on property class; max ~50% for LAP; ROI Details: ROI (HL): CIBIL >730 → 9.35%; CIBIL 700–730 → 9.85–12% (varies by property class); Property Types: Class 1 → Fully approved (best ROI), Class 2 → One approval missing, Class 3 → No approval, Class 4 → Irregular/encroached/<8ft road (high risk); ROI (LAP): 10.35–14% based on property class"
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
    # --- New logic for FOIR, LTV, ROI, Tenure, and Maturity Age ---
    results = []
    coapp_missing_or_late_payment = user_profile.get("coapp_missing_or_late_payment", False)
    for bank, policy in BANK_POLICIES.items():
        eligible = True
        remarks = []
        # FOIR
        foir_limit = 0.7  # default
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
        # Tenure calculation using utility
        current_age = user_profile.get("age", 0)
        maturity_age = policy.get("max_age", 70)
        min_tenure = policy.get("tenure_range", (5, 30))[0]
        max_tenure = policy.get("tenure_range", (5, 30))[1]
        eligible_tenure = calculate_eligible_tenure(current_age, maturity_age, min_tenure, max_tenure)
        # Affordability calculation (max EMI)
        total_income = user_profile.get("monthly_income", 0)
        total_emi = user_profile.get("emi_after_loan", 0)
        max_emi = (foir_limit * total_income) - total_emi
        if max_emi <= 0:
            eligible = False
            remarks.append("No EMI affordability")
            max_emi = 0
        # ROI
        roi = policy.get("interest_rate_hl", (8.0, 9.0))[0] if user_profile.get("product") == "HL" else policy.get("interest_rate_lap", (10.0, 12.0))[0]
        # Amortization calculation for eligible loan amount
        loan_amount, total_interest = calculate_amortization_loan_amount(max_emi, eligible_tenure, roi)
        # LTV check
        ltv_range = policy.get("ltv_hl", (0, 0.8)) if user_profile.get("product") == "HL" else policy.get("ltv_lap", (0, 0.6))
        ltv_limit = ltv_range[1] if isinstance(ltv_range, tuple) else ltv_range
        max_loan_ltv = int(user_profile.get("property_value", 0) * ltv_limit)
        final_max_loan = min(int(loan_amount), max_loan_ltv)
        # LTV eligibility
        ltv = final_max_loan / max(user_profile.get("property_value", 1), 1)
        if isinstance(ltv_range, tuple):
            if not (ltv_range[0] <= ltv <= ltv_range[1]):
                eligible = False
                remarks.append("LTV not in range")
        else:
            if ltv > ltv_range:
                eligible = False
                remarks.append("LTV too high")
        # FOIR check
        foir = total_emi / max(user_profile.get("monthly_income", 1), 1)
        if foir > foir_limit:
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
            "loan_amount": str(final_max_loan),
            "emi": str(int(max_emi)),
            "tenure": f"{eligible_tenure} years",
            "roi": str(roi),
            "maturity_age": str(current_age + eligible_tenure),
            "total_interest": f"₹{total_interest:,.2f}"
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
    flat = flatten_loan_form_input(data)
    results = []
    for bank, policy in BANK_POLICIES.items():
        eligible = True
        remarks = []
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
        # Tenure calculation using utility
        current_age = flat.get("age", 0)
        maturity_age = policy.get("max_age", 70)
        min_tenure = policy.get("tenure_range", (5, 30))[0]
        max_tenure = policy.get("tenure_range", (5, 30))[1]
        eligible_tenure = calculate_eligible_tenure(current_age, maturity_age, min_tenure, max_tenure)
        # Affordability calculation (max EMI)
        total_income = flat.get("monthly_income", 0)
        total_emi = flat.get("emi_after_loan", 0)
        max_emi = (foir_limit * total_income) - total_emi
        if max_emi <= 0:
            eligible = False
            remarks.append("No EMI affordability")
            max_emi = 0
        # ROI
        roi = policy.get("interest_rate_hl", (8.0, 9.0))[0] if flat.get("product") == "HL" else policy.get("interest_rate_lap", (10.0, 12.0))[0]
        # Amortization calculation for eligible loan amount
        loan_amount, total_interest = calculate_amortization_loan_amount(max_emi, eligible_tenure, roi)
        # LTV check
        ltv_range = policy.get("ltv_hl", (0, 0.8)) if flat.get("product") == "HL" else policy.get("ltv_lap", (0, 0.6))
        ltv_limit = ltv_range[1] if isinstance(ltv_range, tuple) else ltv_range
        max_loan_ltv = int(flat.get("property_value", 0) * ltv_limit)
        requested_amount = flat.get("loan_amount_requested", 0)
        # If requested_amount is 0 or not set, do not cap to 0, use calculated value and LTV only
        if requested_amount > 0:
            final_max_loan = min(int(loan_amount), max_loan_ltv, requested_amount)
        else:
            final_max_loan = min(int(loan_amount), max_loan_ltv)
        # Recalculate EMI and amortization if capped by requested amount
        recalc_emi = max_emi
        recalc_total_interest = total_interest
        if final_max_loan < int(loan_amount):
            # Recalculate EMI for capped loan amount
            monthly_rate = roi / 12 / 100
            n_months = eligible_tenure * 12
            if monthly_rate > 0 and n_months > 0:
                recalc_emi = final_max_loan * monthly_rate * (1 + monthly_rate) ** n_months / ((1 + monthly_rate) ** n_months - 1)
                recalc_total_interest = (recalc_emi * n_months) - final_max_loan
            else:
                recalc_emi = 0
                recalc_total_interest = 0
        # LTV eligibility
        ltv = final_max_loan / max(flat.get("property_value", 1), 1)
        if isinstance(ltv_range, tuple):
            if not (ltv_range[0] <= ltv <= ltv_range[1]):
                eligible = False
                remarks.append("LTV not in range")
        else:
            if ltv > ltv_range:
                eligible = False
                remarks.append("LTV too high")
        # FOIR check
        foir = total_emi / max(flat.get("monthly_income", 1), 1)
        if foir > foir_limit:
            eligible = False
            remarks.append("FOIR not in range")
        # Output
        results.append({
            "bank": bank,
            "eligible": "Yes" if eligible else "No",
            "remarks": ", ".join(remarks) if remarks else ("Eligible" if eligible else "Not eligible"),
            "loan_amount": str(final_max_loan),
            "emi": str(int(recalc_emi)),
            "tenure": f"{eligible_tenure} years",
            "roi": str(roi),
            "maturity_age": str(current_age + eligible_tenure),
            "total_interest": f"₹{recalc_total_interest:,.2f}"
        })
    # Map to frontend expected structure
    mapped_banks = []
    for b in results:
        mapped_banks.append({
            "bank": b.get("bank", ""),
            "eligible": b.get("eligible", "No"),
            "remarks": b.get("remarks", ""),
            "loan_amount": b.get("loan_amount", ""),
            "emi": b.get("emi", ""),
            "tenure": b.get("tenure", ""),
            "roi": b.get("roi", "-"),
            "amortization_schedule": []  # You can add detailed schedule if needed
        })
    return {
        "name": flat.get("name", ""),
        "type_of_loan": flat.get("type_of_loan", ""),
        "address": flat.get("address", ""),
        "loan_amount": str(flat.get("loan_amount_requested", "")),
        "date": "",
        "eligible_banks": mapped_banks,
        "enhancement_insights": [],
        "total_emi": str(flat.get("emi_after_loan", 0))
    }

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
