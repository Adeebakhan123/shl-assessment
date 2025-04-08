import os
import pandas as pd
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
OPENAI_API_KEY = os.getenv('AZURE_OPEN_AI')
AzureClient = AzureOpenAI(
    api_key=OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_END_POINT")
)

def generate_embedding(text):
    """Generate an embedding for the given text using Azure OpenAI."""
    response = AzureClient.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    embedding = response.data[0].embedding
    return embedding

def process_csv(file_path):
    """Process a CSV file and combine text columns into a single string per row."""
    df = pd.read_csv(file_path)
    text_columns = df.select_dtypes(include=['object']).columns
    df['combined_text'] = df[text_columns].astype(str).agg(' '.join, axis=1)
    return df[['combined_text']]

def save_embeddings(embeddings_dict, file_path):
    """Save embeddings dictionary to an .npy file."""
    np.save(file_path, embeddings_dict)

def embed_and_store(csv_file_path, embeddings_file_path):
    """Generate embeddings from a CSV file and store them."""
    df = process_csv(csv_file_path)
    embeddings = {
        'ids': [],
        'texts': [],
        'embeddings': []
    }
    print(len(df))
    for idx, row in df.iterrows():
        text = row['combined_text']
        embedding = generate_embedding(text)
        embeddings['ids'].append(idx)
        embeddings['texts'].append(text)
        embeddings['embeddings'].append(embedding)
        print(f"Processed row {idx+1}/{len(df)}")
    # Convert lists to numpy arrays
    embeddings['ids'] = np.array(embeddings['ids'])
    embeddings['texts'] = np.array(embeddings['texts'])
    embeddings['embeddings'] = np.array(embeddings['embeddings'])
    save_embeddings(embeddings, embeddings_file_path)

if __name__ == "__main__":
    csv_file_path = "SHL_assessment.csv"  # Replace with your CSV file path
    embeddings_file_path = "embeddings.npy"  # Replace with desired output path
    embed_and_store(csv_file_path, embeddings_file_path)
    print(f"Embeddings saved to {embeddings_file_path}")