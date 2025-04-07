import os
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
import pandas as pd
# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
OPENAI_API_KEY = os.getenv('AZURE_OPEN_AI')
AzureClient = AzureOpenAI(
    api_key=OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_END_POINT")
)


def fetch_text_from_csv(csv_file_path, indices):
    """Fetch text from a CSV file based on given indices."""
    df = pd.read_csv(csv_file_path)
    return df.iloc[indices][['Assessment Name', 'URL', 'Test Type','Remote Testing','Adaptive/IRT',
      'assessment_length_modified']].reset_index(drop=True)


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
    from sklearn.metrics.pairwise import cosine_similarity
    if embeddings['embeddings'].size == 0:
        return []
    embedding_matrix = embeddings['embeddings']
    query_vec = np.array(query_embedding).reshape(1, -1)
    similarities = cosine_similarity(query_vec, embedding_matrix)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    return top_indices

if __name__ == "__main__":
    embeddings_file_path = "embeddings.npy"  # Replace with your embeddings file path
    query = """I am hiring for Java developers who can also collaborate effectively with my business teams. Looking
for an assessment(s) that can be completed in 40 minutes."""  # Replace with your query
    embeddings = load_embeddings(embeddings_file_path)
    query_embedding = generate_embedding(query)
    indexes = retrieve_similar(embeddings, query_embedding)
    similar_items = fetch_text_from_csv("b.csv", indexes)
    with open("c.csv", "w") as fp:
        fp.write(similar_items.to_csv(index=False, header=True))
    print(similar_items)
    
    
