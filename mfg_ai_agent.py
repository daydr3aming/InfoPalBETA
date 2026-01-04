from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import streamlit as st

AZURE_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]

model = AzureChatOpenAI(
    openai_api_base=f"https://flexapimanager.flex.com/openai/mfg-ai-agent/deployments/gpt-4o-mini",
    api_key=AZURE_API_KEY,
    api_version="2024-08-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
