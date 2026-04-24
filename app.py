import streamlit as st
import pandas as pd
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
import os

# --- Configuration ---
st.set_page_config(page_title="SDR Research Assistant", page_icon="🚀", layout="wide")

# Set keys from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

# Initialize Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# --- Data Structures ---
class InteractionStep(BaseModel):
    day: int = Field(description="The day number in the 30-day sequence (1-30)")
    channel: str = Field(description="Channel: LinkedIn, Email, or Phone")
    strategy: str = Field(description="The core objective or value-add for this touchpoint")
    script: str = Field(description="The actual message/script to send")

class InteractionPlan(BaseModel):
    plan: List[InteractionStep] = Field(description="A 30-day sequence of interaction steps")

# --- Functions ---
def research_company(domain):
    query = f"recent news, tech stack, company mission, and pain points for {domain} to inform B2B sales outreach"
    return tavily.search(query=query, search_depth="advanced")

def generate_structured_plan(research_data, product_info):
    structured_llm = llm.with_structured_output(InteractionPlan)
    prompt = f"""
    You are an expert SDR manager. 
    
    PRODUCT TO SELL: {product_info}
    
    TARGET COMPANY DATA: {research_data}
    
    TASK: Create a 30-day multi-channel (email/LinkedIn) interaction plan.
    The scripts must explicitly connect the target company's pain points (found in the data) 
    to the specific benefits of the PRODUCT TO SELL.
    
    Focus on value-add, not hard selling.
    """
    return structured_llm.invoke(prompt)

# --- UI ---
st.title("🚀 SDR Research Assistant")

# 1. Product Context (Persisted in Session State)
if 'product_desc' not in st.session_state:
    st.session_state['product_desc'] = ""

st.subheader("1. Configure Your Outreach")
product_desc = st.text_area(
    "What are you selling?", 
    value=st.session_state['product_desc'],
    placeholder="e.g., We sell an AI-powered email automation tool that helps SDRs save 10 hours a week."
)
st.session_state['product_desc'] = product_desc

domain = st.text_input("Company Domain of the Prospect", placeholder="e.g., openai.com")

if st.button("Generate 30-Day Plan"):
    if not domain or not product_desc:
        st.warning("Please enter both a domain and your product description.")
    else:
        with st.spinner("Researching and strategizing..."):
            try:
                data = research_company(domain)
                result = generate_structured_plan(data, product_desc)
                st.session_state['editable_plan'] = [step.model_dump() for step in result.plan]
                st.success("Plan generated successfully!")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Editable Table ---
if 'editable_plan' in st.session_state:
    st.subheader("2. Fine-tune your 30-day plan")
    edited_df = st.data_editor(
        pd.DataFrame(st.session_state['editable_plan']), 
        use_container_width=True,
        hide_index=True,
        column_config={"script": st.column_config.TextColumn("Script", width="large")}
    )
    
    st.session_state['editable_plan'] = edited_df.to_dict('records')
    
    csv = edited_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Plan as CSV", data=csv, file_name=f"{domain}_plan.csv", mime='text/csv')