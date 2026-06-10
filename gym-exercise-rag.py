import os
import time
import pandas as pd
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()

# Load the dataset
df = pd.read_excel('Gym Exercises Dataset.xlsx', nrows=50)

# Dropping dublicates if any
df = df.drop_duplicates(subset=["Exercise_Name", "Description_URL", "muscle_gp"])

# Initialize the GenAI client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Creating pinecone index
if not pc.has_index("gym-exercises"):
    pc.create_index(
        name = "gym-exercises",
        dimension = 768,
        metric = "cosine",
        spec = ServerlessSpec(
            cloud = "aws",
            region = "us-east-1"
        )
    )
index = pc.Index("gym-exercises")

# Function to create embeddings for a given text using GenAI
def create_embedding(text):
    while True:
        try:
            response = client.models.embed_content(
                model="gemini-embedding-2-preview",
                contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            print(f"Retrying... {e}")
            time.sleep(10)

# Embed the column Exercise name using embedding model
excercise_embeddings = []
for i, row in df.iterrows():
    excercise_name = row["Exercise_Name"]
    excercise_source = row["Description_URL"]
    muscle_group = row["muscle_gp"]
    excercise_embedding = create_embedding(excercise_name+" "+muscle_group)
    excercise_embeddings.append(excercise_embedding)

    # Upsert vectors into pinecone index
    index.upsert(
    vectors=[
        {
            "id": str(i), 
            "values": excercise_embedding, 
            "metadata": {"text": excercise_name, "source":excercise_source}
        }
    ],
    namespace="excercise-namespace")

# Take a user query and embed it using the same embedding model
query = "What are the best exercises for chest?"
query_embedding = create_embedding(query)

# Retrieve top 3 most similar results
query_res = index.query(
    namespace="excercise-namespace",
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

# Extract excercise name and sources from metadata of each match
retrieved_text = ""
retrieved_source = ""
for res in query_res.matches:
    text = res.metadata["text"]
    source = res.metadata["source"]
    retrieved_text += text + "\n"
    retrieved_source += source + "\n"

# Building prompt that includes query and retrieved text
prompt = query + "\n\n" + retrieved_text 

# Pass retrieved results to LLM to and return answer with source
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    config=types.GenerateContentConfig(
        system_instruction= f"""Your are helpful AI assistant answer 
        the questions asked. Answer according to context provided in
        {retrieved_text}.
        """),
    contents=prompt
)

print("Response:"+ "\n" + response.text + "\n\n" + "Sources:" + "\n" + retrieved_source)