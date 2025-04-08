import os
import numpy as np
import pandas as pd
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
OPENAI_API_KEY = os.getenv('AZURE_OPEN_AI')
AzureClient = AzureOpenAI(
    api_key=OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_END_POINT")
)
import requests
from bs4 import BeautifulSoup

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return text[:3000]  # Limit to first 3000 characters
    except Exception as e:
        return f"Error extracting text: {e}"


def fetch_text_from_csv(csv_file_path, indices):
    """Fetch text from a CSV file based on given indices."""
    df = pd.read_csv(csv_file_path)
    return df.iloc[indices][['Assessment Name', 'URL', 'Test Type', 'Remote Testing', 'Adaptive/IRT', 'assessment_length_modified']].reset_index(drop=True)

def generate_embedding(text):
    """Generate an embedding for the given text using Azure OpenAI."""
    response = AzureClient.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    embedding = response.data[0].embedding
    return embedding

def load_embeddings(embeddings_file):
    """Load embeddings from an .npy file."""
    if os.path.exists(embeddings_file):
        data = np.load(embeddings_file, allow_pickle=True).item()
        return data
    else:
        return {'ids': np.array([]), 'texts': np.array([]), 'embeddings': np.array([])}

def retrieve_similar(embeddings, query_embedding, top_k=10):
    """Retrieve top-k similar items based on cosine similarity."""
    if embeddings['embeddings'].size == 0:
        return []
    embedding_matrix = embeddings['embeddings']
    query_vec = np.array(query_embedding).reshape(1, -1)
    similarities = cosine_similarity(query_vec, embedding_matrix)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return top_indices

# Streamlit UI
st.set_page_config(page_title="Assessment Finder", layout="wide")

st.title("Assessment Finder")
st.markdown("Find assessments that match your requirements using semantic search.")

# File paths - using local files in the same directory
csv_file_path = "SHL_assessment.csv"  # CSV file in the same directory
embeddings_file_path = st.sidebar.text_input("Embeddings file path", "embeddings.npy")

# Check if files exist
csv_exists = os.path.exists(csv_file_path)
embeddings_exist = os.path.exists(embeddings_file_path)

# Show file status
if csv_exists:
    st.sidebar.success(f"Using CSV file: {csv_file_path}")
else:
    st.sidebar.error(f"CSV file not found: {csv_file_path}")

if embeddings_exist:
    st.sidebar.success(f"Using embeddings file: {embeddings_file_path}")
else:
    st.sidebar.error(f"Embeddings file not found: {embeddings_file_path}")

# Query and search settings
query = st.text_area("Enter your assessment requirements", 
                   placeholder="E.g., I'm hiring for data analysts with SQL and Excel skills. Assessment should be under 30 minutes.")

# Optional job description URL input
st.markdown("**Or** paste a job description URL (we'll try to extract text from it):")
job_url = st.text_input(
    "Job Description URL (optional)", 
    placeholder="https://example.com/job-posting"
)
top_k = st.sidebar.slider("Number of results", min_value=1, max_value=10, value=10)

# Search button
if st.button("Search for Assessments"):
    
    if job_url.strip():
            st.info("Using job description from URL.")
            query_to_use = extract_text_from_url(job_url)
    else:
        query_to_use = query

    if not csv_exists or not embeddings_exist:
        st.error("Please make sure both CSV and embeddings files exist in the correct location.")
    else:
        try:
            with st.spinner("Generating embedding and searching..."):
                # Load embeddings
                embeddings = load_embeddings(embeddings_file_path)
                
                # Generate query embedding
                query_embedding = generate_embedding(query_to_use)
                
                # Retrieve similar items
                indices = retrieve_similar(embeddings, query_embedding, top_k=top_k)
                
                if len(indices) > 0:
                    # Fetch and display results
                    similar_items = fetch_text_from_csv(csv_file_path, indices)
                    st.success(f"Found {len(similar_items)} matching assessments!")
                    
                    # Display results in a table
                    st.dataframe(similar_items, use_container_width=True)
                    
                    # Option to download results
                    csv_download = similar_items.to_csv(index=False)
                    st.download_button(
                        label="Download results as CSV",
                        data=csv_download,
                        file_name="assessment_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No matching assessments found.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Additional information
with st.expander("How to use this tool"):
    st.markdown("""
    1. Make sure your CSV file (SHL_assessment_final.csv) is in the same directory as this app.
    2. Enter the path to your embeddings file (default is 'embeddings.npy' in the same directory).
    3. Enter your query describing the assessment requirements.
    4. Adjust the number of results you want to see.
    5. Click 'Search for Assessments' to find matching assessments.
    6. Download the results as a CSV file if needed.
    
    The tool uses semantic search to find assessments that match your requirements based on embeddings generated from Azure OpenAI.
    """)