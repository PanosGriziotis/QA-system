import streamlit as st
import json
import requests

# Streamlit App Setup
st.markdown("<h1 style='font-size:32px;'>Q&A App</h1>", unsafe_allow_html=True)

# Pipeline selection
st.markdown("<h3 style='font-size:24px;'>Select the QA Pipeline</h3>", unsafe_allow_html=True)
pipeline = st.radio("Select QA Pipeline", ('RAG', 'Extractive'), label_visibility="collapsed")

# JSON parameters input
st.markdown("<h3 style='font-size:24px;'>Query Parameters (in JSON)</h3>", unsafe_allow_html=True)
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
params_input = st.text_area("Query Parameters", json.dumps(default_params, indent=2), height=200, label_visibility="collapsed")

# Query input
st.markdown("<h3 style='font-size:24px;'>Enter your query</h3>", unsafe_allow_html=True)
query = st.text_input("Enter your query here", label_visibility="collapsed")

# Button to submit the query
if st.button("Submit Query"):
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
        response.raise_for_status()  
        json_response = response.json()
        
        # Display answer
        answer = json_response.get("answers", [{}])[0].get("answer", "No answer found.")
        st.markdown(f"<h3 style='font-size:24px;'>Answer:</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:20px;'>{answer}</p>", unsafe_allow_html=True)

        # Display retrieved documents
        documents = json_response.get("documents", [])
        main_doc = None
        
        # For Extractive pipeline, place the main document on top
        if pipeline == "Extractive" and documents:
            main_doc_id = json_response.get("answers", [{}])[0].get("document_ids", [None])[0]
            main_doc = next((doc for doc in documents if doc.get("id") == main_doc_id), None)
            if main_doc:
                documents.remove(main_doc) 
                documents.insert(0, main_doc)  # Insert main document at the top

        # Display Documents
        st.markdown("<h4 style='font-size:20px;'>Retrieved Documents</h4>", unsafe_allow_html=True)

        for i, doc in enumerate(documents, start=1):
            with st.expander(f"Document {i}"):
                st.write(doc.get("content", "No content available"))

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
    except KeyError:
        st.error("Unexpected response structure.")
