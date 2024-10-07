import os
import re
import asyncio
import gradio as gr
import pdfplumber
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import torch

app = FastAPI()

# Dictionary of terms and their synonyms
term_synonyms = {
    "Capex": ["Capital Expenditure", "Capital Expense", "CapEx", "Property Acquisition"],
    "Equity": ["Shareholders' Equity", "Owners' Equity"],
    "Adjusted EBITDA": ["AEBITDA"],
    "Total Revenue": ["Gross Revenue", "Total Income"],
    "Recurring Revenue": ["Subscription Revenue", "Recurring Income"],
    "Adjustments to EBITDA": ["EBITDA Adjustments"],
    "Net Income": ["Net Profit", "Net Earnings"],
    "D&A (Depreciation & Amortization)": ["Depreciation and Amortization", "D&A"],
    "Cash Interest": ["Interest Expense"],
    "Cash": ["Cash and Cash Equivalents"],
    "Qualifying Cash": [],
    "Super Senior OpCo Debt": [],
    "Total 1L Debt": ["First Lien Debt"],
    "Other Debt": ["Miscellaneous Debt"],
    "Actual Taxes": ["Tax Expense", "Income Taxes"],
    "Dividends": ["Dividend Payments"],
    "Management Fees": ["Executive Compensation"],
    "Modified Liquidity Ratio": [],
    "Debt Amortization": ["Loan Amortization"],
    "Actual Interest": ["Interest Expense"]
}

TERMS = list(term_synonyms.keys())

MODEL_PATH = os.getenv("MODEL_PATH", "/opt/homebrew/var/www/main _api/llama-2-7b-chat.Q4_K_M.gguf")

device = "mps" if torch.backends.mps.is_available() else "cpu"

def extract_financial_terms(pdf_path, selected_terms):
    results = []
    
    # Flatten selected terms with their synonyms
    term_list = {term: synonyms for term, synonyms in term_synonyms.items() if term in selected_terms}
    term_list.update({syn: [] for synonyms in term_list.values() for syn in synonyms})

    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            
            for page_num in range(num_pages):
                page = pdf.pages[page_num]
                text = page.extract_text()
                
                for main_term, synonyms in term_list.items():
                    all_terms = [main_term] + synonyms
                    for term in all_terms:
                        pattern = re.compile(r'{}[^$\d]*(\$?[\d,]+(?:\.\d+)?)'.format(re.escape(term)), re.IGNORECASE)
                        matches = pattern.findall(text)
                        
                        for match in matches:
                            if match.strip():
                                try:
                                    value = float(match.replace('$', '').replace(',', ''))
                                    if value >= 200:  # Exclude values less than 200
                                        results.append({
                                            'label': main_term,
                                            'matched_term': term,
                                            'value': value,
                                            'page': page_num + 1,
                                            'position': 'Exact position not available'
                                        })
                                except ValueError:
                                    print(f"Warning: Could not convert '{match}' to float for term '{term}' on page {page_num + 1}")

    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    
    return results

def load_vector_store(embedding_model, file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(texts, embedding_model)
    return vectorstore

def summarize_document(vectorstore):
    llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=True, callbacks=CallbackManager([StreamingStdOutCallbackHandler()]), temperature=0.7, max_tokens=4096)
    
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(), chain_type="stuff")
    
    return qa_chain.invoke({'query': "Summarize the document."})

async def process_file(file_obj, selected_terms):
    if not file_obj:
        return {"error": "No file provided."}

    file_path = file_obj.name  # Use the file object to get the file path

    if not file_path.lower().endswith('.pdf'):
        return {"error": "Uploaded file is not a PDF."}

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return {"error": "Uploaded file is empty or does not exist."}

    extracted_data = extract_financial_terms(file_path, selected_terms)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = load_vector_store(embedding_model, file_path)

    summary_response = summarize_document(vectorstore)
    
    return {
        "extracted_terms": extracted_data,
        "summary": summary_response.get("result", "No summary available.").strip()
    }

@app.post("/extract-terms/")
async def extract_terms(file_obj: str, selected_terms: list):
    return await process_file(file_obj, selected_terms)

def extract_terms_from_gradio(file_obj, selected_terms):
    result = asyncio.run(process_file(file_obj, selected_terms))
    
    # Prepare JSON for extracted terms
    extracted_terms = result.get("extracted_terms", [])
    summary = result.get("summary", "No summary available.")
    
    return extracted_terms, summary


def run_gradio():
    gr.Interface(
        fn=extract_terms_from_gradio,
        inputs=[
            gr.File(label="Upload PDF Document"),
            gr.CheckboxGroup(label="Select Financial Terms", choices=TERMS)
        ],
        outputs=[
            gr.JSON(label="Extracted Financial Terms"),  # Output for extracted terms
            gr.Textbox(label="Summary", lines=5)          # Output for the document summary
        ],
        title="Financial Document Processor",
        description="Extracts selected financial terms from PDF documents and summarizes them."
    ).launch()


if __name__ == "__main__":
    run_gradio()
