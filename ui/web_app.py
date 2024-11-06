import streamlit as st
import json
import requests

# Streamlit App Setup
st.title("Q&A App")

# Query pipeline selection
pipeline = st.radio("Select the QA Pipeline", ('RAG', 'Extractive'))

# JSON parameters input
default_params_rag = {
    "BM25Retriever": {"top_k": 10}, 
    "DenseRetriever": {"top_k": 10},
    "Reranker": {"top_k": 4},
    "GenerativeReader": {"temperature": 0.75, "max_new_tokens": 150, "top_p": 0.95, "post_processing": True},
    "Responder": {"threshold": 0.17}
}

default_params_ex = {
    "BM25Retriever": {"top_k": 10}, 
    "DenseRetriever": {"top_k": 10},
    "Reranker": {"top_k": 6},
    "ExtractiveReader": {"top_k": 6},
    "Responder": {"threshold": 0.17}
}

# Display default parameters based on pipeline selection
default_params = default_params_rag if pipeline == "RAG" else default_params_ex
params_input = st.text_area("Query Parameters (in JSON)", json.dumps(default_params, indent=2))

# Query input
query = st.text_input("Enter your query")

# Button to submit the query
if st.button("Submit Query"):
    # Parse JSON parameters from input
    try:
        query_params = json.loads(params_input)
    except json.JSONDecodeError:
        st.error("Invalid JSON format for parameters.")
        query_params = default_params

    # Define endpoint based on pipeline selection
    endpoint = "rag-query" if pipeline == "RAG" else "extractive-query"
    
    # Prepare request body
    request_body = {
        "query": query,
        "params": query_params
    }
    
    # Send the request to the server
    try:
        response = requests.post(url=f"http://localhost:8001/{endpoint}", json=request_body)
        response.raise_for_status()  # Raise an error for unsuccessful requests
        json_response = response.json()
        
        # Display the answer
        answer = json_response.get("answers", [{}])[0].get("answer", "No answer found.")
        st.success("Answer:")
        st.write(answer)

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
    except KeyError:
        st.error("Unexpected response structure.")