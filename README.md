# gym-exercise-rag
A RAG system that finds gym exercises relevant to user queries using Pinecone vector database and Google Gemini embeddings

## What it does
Loads gym exercises from an Excel file, embeds them using Gemini, stores vectors in Pinecone, retrieves top 3 matching exercises for user queries, and generates an AI-powered answer with source links

## How it works
- Use pandas to load the excel file
- Gemini "gemini-embedding-2-preview" model converts exercise text into vector embeddings
- Create Pinecone index and upsert vectors to Pinecone index with metadata (exercise name and source URL)
- Create query embedding 
- Retrieve top 3 results and extract its excercise name and source
- Build a prompt using query and retrieved text and give it to Gemini "gemini-3-flash-preview" model which responses along with sources

## How to run it
1. Download the dataset from Kaggle: [Gym Exercises Dataset]([https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews](https://www.kaggle.com/datasets/ambarishdeb/gym-exercises-dataset))
2. Install requirements: pip install pandas pinecone-client google-genai python-dotenv
3. Set your OpenAI API key
4. Run the Python file

## What I learned
-How to read data from Excel files
-Creating vector embeddings using Gemini API
-Upserting and querying vectors in Pinecone
-Building RAG pipelines Retrieving top results augment the prompt with retrieved context and generate response using LLM
-Constraining LLM outputs to specific context to avoid hallucinations using prompt engineering
-Handling API rate limits and quota management
