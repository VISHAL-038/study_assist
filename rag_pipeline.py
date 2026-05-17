import os
import faiss
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load .env file — pass the explicit path to be safe
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=api_key)


def build_vector_store(chunks):
    """
    Takes a list of text chunks and:
    1. Converts each chunk to a vector (embedding)
    2. Stores all vectors in a FAISS index for fast searching
    
    Returns: (faiss_index, list_of_chunks)
    We return chunks too so we can retrieve the original text later.
    """
    # Convert all chunks to vectors — shape: (num_chunks, 384)
    embeddings = embedding_model.encode(chunks)

    # 384 is the vector size for "all-MiniLM-L6-v2"
    dimension = embeddings.shape[1]

    # Create a FAISS index — IndexFlatL2 uses simple distance search
    index = faiss.IndexFlatL2(dimension)

    # Add all vectors to the index
    index.add(np.array(embeddings))

    return index, chunks


def retrieve_relevant_chunks(query, index, chunks, top_k=5):
    """
    Given a user's question:
    1. Convert the question to a vector
    2. Search FAISS for the top_k most similar chunk vectors
    3. Return those chunks as context for the LLM
    """
    # Convert the query to a vector
    query_vector = embedding_model.encode([query])  # shape: (1, 384)

    # Search the index — returns distances and indices of top_k matches
    distances, indices = index.search(np.array(query_vector), top_k)

    # Retrieve the actual text of the matching chunks
    relevant_chunks = [chunks[i] for i in indices[0]]

    return relevant_chunks


def ask_groq(query, relevant_chunks):
    """
    Sends the user's question + retrieved context to Groq (LLaMA 3).
    Returns the LLM's answer as a string.
    """
    # Combine all retrieved chunks into one context block
    context = "\n\n".join(relevant_chunks)

    # This is the prompt we send to the LLM
    # We tell it: here is relevant context, now answer the question
    prompt = f"""You are a helpful study assistant.
Use the following context from the student's notes to answer the question.
If the answer is not in the context, say "I couldn't find that in your notes."

Context:
{context}

Question: {query}

Answer:"""

    # Call the Groq API
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract and return just the text of the answer
    return response.choices[0].message.content

def generate_quiz(topic, index, chunks, num_questions=5):
    """
    Given a topic:
    1. Retrieves relevant chunks from the notes
    2. Asks Groq to generate MCQs from those chunks
    3. Returns a list of question dictionaries
    """
    # Reuse our existing retrieval function to find relevant chunks
    relevant_chunks = retrieve_relevant_chunks(topic, index, chunks, top_k=8)
    context = "\n\n".join(relevant_chunks)

    prompt = f"""You are a quiz generator for a student's study notes.
Based on the context below, generate exactly {num_questions} multiple choice questions.

Rules:
- Each question must have exactly 4 options labeled A, B, C, D
- Only one option is correct
- Make questions test understanding, not just memory
- Base every question strictly on the context provided

You MUST respond in this exact JSON format and nothing else:
[
  {{
    "question": "What is ...?",
    "options": {{
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    }},
    "answer": "A",
    "explanation": "Because ..."
  }}
]

Context:
{context}

Topic: {topic}

JSON:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3   # lower = more focused, consistent output
    )

    raw = response.choices[0].message.content.strip()

    # Clean up in case the model wraps output in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    import json
    try:
        questions = json.loads(raw)
    except json.JSONDecodeError:
        questions = []   # return empty list if parsing fails

    return questions


def summarize_topic(topic, index, chunks, length="detailed"):
    """
    Given a topic:
    1. Retrieves relevant chunks from the notes
    2. Asks Groq to write a clean structured summary
    3. Returns the summary as a string
    """
    # Fetch more chunks for summaries so we cover the topic broadly
    relevant_chunks = retrieve_relevant_chunks(topic, index, chunks, top_k=10)
    context = "\n\n".join(relevant_chunks)

    # Adjust instructions based on chosen length
    if length == "short":
        length_instruction = "Write a concise summary in 3-5 bullet points only."
    else:
        length_instruction = """Write a detailed summary with these sections:
- Overview (2-3 sentences)
- Key Concepts (bullet points)
- Important Details (bullet points)
- Quick Recap (1-2 sentences)"""

    prompt = f"""You are a study assistant helping a student understand their notes.
Summarize the topic below using ONLY the context provided.
If the context doesn't cover the topic well, say so honestly.

{length_instruction}

Context:
{context}

Topic to summarize: {topic}

Summary:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()

def ask_groq_with_memory(query, index, chunks, chat_history):
    """
    Conversational Q&A with memory.
    
    chat_history is a list of dicts like:
    [
        {"role": "user", "content": "What is backpropagation?"},
        {"role": "assistant", "content": "Backpropagation is..."},
        {"role": "user", "content": "Give me an example of that"}
    ]
    
    We send the full history to Groq so it understands follow-up questions.
    """
    # Retrieve relevant chunks based on the latest question
    relevant_chunks = retrieve_relevant_chunks(query, index, chunks, top_k=5)
    context = "\n\n".join(relevant_chunks)

    # System message tells the LLM its role and gives it the context
    # This is sent once at the start of every request
    system_message = {
        "role": "system",
        "content": f"""You are a helpful study assistant. 
Answer questions using the context from the student's notes below.
If the answer is not in the context, say "I couldn't find that in your notes."
Keep answers clear and student-friendly.

Relevant context from notes:
{context}"""
    }

    # Build the full message list: system + entire chat history
    messages = [system_message] + chat_history

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message.content.strip()

def summarize_full_pdf(chunks):
    """
    Summarizes the entire PDF by:
    1. Taking evenly spaced chunks from across the whole document
       (so we cover beginning, middle and end — not just the first pages)
    2. Asking Groq to write a full document summary
    """
    # Pick 20 evenly spaced chunks to represent the whole document
    # This avoids hitting the token limit while covering all sections
    total = len(chunks)
    step = max(1, total // 20)
    sampled_chunks = chunks[::step][:20]

    context = "\n\n".join(sampled_chunks)

    prompt = f"""You are a study assistant. A student has uploaded their study material.
Write a comprehensive summary of the entire document based on the content below.

Structure your summary as:
- Document Overview (2-3 sentences about what this document covers)
- Main Topics Covered (bullet list of all major topics found)
- Key Concepts (the most important ideas explained briefly)
- Important Terms (any key terms or definitions found)
- Quick Revision Points (5-7 bullet points a student should remember)

Content:
{context}

Full Document Summary:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()