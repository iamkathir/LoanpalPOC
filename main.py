from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import google.generativeai as genai
import json
import re


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

@app.post("/recommend-loan", response_model=RecommendationResponse)
def recommend_loan(user_profile: UserProfile = Body(...)):
    prompt = f"""
You are a credit engine for a bank. Based on the following user profile, calculate and recommend loan eligibility for all major banks using only FOIR (Fixed Obligations to Income Ratio), LTV (Loan to Value), and location as the main criteria. Do not use any bank policy text.

If the user has a co-applicant, include the co-applicant's monthly income and existing loan obligations in all calculations (FOIR, total income, total EMI, etc). If the co-applicant has an existing loan, include their EMI in the total EMI calculation.

For each bank, provide:
- bank: Bank name
- interest_rate: Interest rate range or value
- loan_amount: Eligible loan amount (as a string, e.g., "2000000" or "20 Lakhs")
- tenure: Recommended tenure (as a string, e.g., "20 years")
- emi: Calculated EMI for the eligible loan (as a string, e.g., "16500" or "₹16,500")

Also provide:
- enhancement_insights: Suggestions for improving eligibility or getting a better rate
- total_emi: The total EMI (including all existing loans for both applicant and co-applicant, as a string)

Return a JSON object with these fields:
- name
- type_of_loan
- address
- loan_amount (as a string)
- date
- eligible_banks (list of all banks as above)
- enhancement_insights
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
    except Exception:
        return {"result": text}
    return result
