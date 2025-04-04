import streamlit as st
import numpy as np
import json
import time
import pymupdf  
import os
import docx  
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.common.by import By

from supabase import create_client
from dotenv import load_dotenv
load_dotenv()

# ---  Supabase Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---  Hugging Face Embedding Model ---
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ---  Gemini AI Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# ---  Streamlit Page Config ---
st.set_page_config(page_title="Domain & File QA System", layout="wide")
st.title("Domain Scraping & File Upload System")

# ---  Initialize Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---  Function to extract text from files ---
def extract_text(file, file_type):
    try:
        if file_type in ["txt", "py"]:
            return file.read().decode("utf-8")
        elif file_type == "pdf":
            pdf_doc = pymupdf.open(stream=file.read(), filetype="pdf")
            return "\n".join([page.get_text("text") for page in pdf_doc])
        elif file_type == "docx":
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error extracting text: {e}")
    return ""

# ---  Function to split text into chunks ---
def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ---  Web Scraper ---
def scrape_website(domain):
    if not domain.startswith(("http://", "https://")):
        domain = "https://" + domain  

    driver = webdriver.Edge()
    driver.get(domain)
    time.sleep(3)

    all_text = ""
    visited_urls = set()
    urls_to_visit = {domain}

    while urls_to_visit:
        url = urls_to_visit.pop()
        if url in visited_urls:
            continue
        visited_urls.add(url)

        try:
            driver.get(url)
            time.sleep(2)
            page_data = driver.find_element(By.TAG_NAME, "body").text
            all_text += f"\n\nPage: {url}\n{page_data}\n{'='*80}"

           
            navbar_links = driver.find_elements(By.CSS_SELECTOR, "nav a")
            footer_links = driver.find_elements(By.CSS_SELECTOR, "footer a")

            for link in navbar_links + footer_links:
                href = link.get_attribute("href")
                if href and domain in href and href not in visited_urls:
                    urls_to_visit.add(href)

        except Exception as e:
            print(f"Skipping {url} due to error: {e}")
            continue

    driver.quit()
    return all_text


def store_data_in_supabase(text_data, source_type):
    
    supabase_client.table("embeddings").delete().neq("id", -1).execute()  # Deletes all records
    response = supabase_client.table("embeddings").select("id").order("id", desc=True).limit(1).execute()
    max_id = response.data[0]["id"] + 1 if response.data else 1

    text_chunks = chunk_text(text_data)
    success = False

    for chunk in text_chunks:
        embedding = embedding_model.encode(chunk).tolist()
        data = {"id": max_id, "text": chunk, "embedding": json.dumps(embedding), "source_type": source_type}
        response = supabase_client.table("embeddings").insert(data).execute()
        if response.data:
            success = True 
        max_id += 1
    
    if success:
        st.session_state.chat_history = []  # Reset chat history
        st.success("All data chunks stored successfully! Chat history has been reset.")

# ---  Sidebar ---
st.sidebar.header("Options")
option = st.sidebar.selectbox("Select Data Source", ["Scraped Data", "Uploaded Files"])

if option == "Uploaded Files":
    uploaded_files = st.sidebar.file_uploader("Upload files", type=["py", "pdf", "docx", "txt"], accept_multiple_files=True)

    if st.sidebar.button("Upload and Process Files"):
        if uploaded_files:
            combined_text = ""
            for uploaded_file in uploaded_files:
                file_type = uploaded_file.name.split(".")[-1]
                file_content = extract_text(uploaded_file, file_type)
                if file_content:
                    combined_text += file_content + "\n\n"
            if combined_text:
                store_data_in_supabase(combined_text, "file_upload")
        else:
            st.error("Please upload at least one valid file.")
    

elif option == "Scraped Data":
    domain = st.sidebar.text_input("Enter Domain Name")
    if st.sidebar.button("Scrape Website Data"):
        if domain:
            scraped_data = scrape_website(domain)
            # st.subheader("Scraped Data Preview")
            # st.text_area("Preview", scraped_data[:1000], height=200)
            store_data_in_supabase(scraped_data, "scraped")
        else:
            st.error("Please enter a valid domain.")

            
st.subheader("Ask a Question")

# Initialize session state for chat history 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
if st.session_state.chat_history:
    st.subheader("Conversation History")
    for i, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** {chat['question']}")
        st.markdown(f"**A{i+1}:** {chat['answer']}")
        st.download_button(
            f"Download Answer {i+1}",
            chat['answer'],
            file_name=f"answer_{i+1}.txt",
            key=f"download_{i}"
        )
        st.write("---")

# Always show a fresh input field
query_text = st.text_input(
    "Enter your question:", 
    key=f"input_{len(st.session_state.chat_history)}",  # Unique key for each new input
    value=""
)

if st.button("Get Answer"):
    if not query_text.strip():
        st.error("Please enter a question before clicking 'Get Answer'.")
    else:
        with st.spinner("Finding answer..."):
            query_embedding = embedding_model.encode(query_text).tolist()
            response = supabase_client.table("embeddings").select("text, embedding").execute()

            if response.data:
                stored_texts = response.data
                similarities = [(item["text"], np.dot(np.array(json.loads(item["embedding"])), query_embedding)) 
                              for item in stored_texts]
                top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]

                # Generate Answer
                gemini_response = gemini_model.generate_content(
                    f"Answer the question: {query_text}\n\nDocument: {top_matches[0][0]}"
                ).text

                # Store in chat history
                st.session_state.chat_history.append({
                    "question": query_text, 
                    "answer": gemini_response
                })
                
                
                st.rerun()

            
# ---  Display Chat History ---
st.sidebar.subheader("Chat History")
for chat in reversed(st.session_state.chat_history):
    with st.sidebar.expander(chat["question"]):
        st.write(chat["answer"]) 