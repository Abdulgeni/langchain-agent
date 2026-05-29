import streamlit as st
import tempfile
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="LangChain RAG Agent", page_icon="🦜")
st.title("🦜 LangChain-Style RAG Agent with Memory")
st.markdown("Production-ready RAG agent with tools, memory, and reasoning.")

with st.sidebar:
    st.header("🧠 Agent Architecture")
    st.markdown("""
    **LangChain Agent Pattern:**
    1. Tool: Document Search
    2. Tool: Entity Extraction
    3. Memory: Conversation History
    4. Reasoning: Chain of Thought
    
    ---
    🆓 100% Free
    """)
    if st.button("🧹 Clear Memory"):
        st.session_state.memory = []
        st.rerun()

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Memory
if 'memory' not in st.session_state:
    st.session_state.memory = []

uploaded_files = st.file_uploader("Upload PDFs:", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        pdf_reader = PyPDF2.PdfReader(tmp_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        chunks = [c.strip() for c in text.split('\n\n') if len(c.strip()) > 100]
        all_chunks.extend(chunks)
    
    embeddings = model.encode(all_chunks)
    st.success(f"📦 {len(all_chunks)} chunks indexed")
    
    # Show memory
    if st.session_state.memory:
        with st.expander(f"🧠 Memory ({len(st.session_state.memory)} items)"):
            for m in st.session_state.memory[-5:]:
                st.markdown(f"- Q: {m['question'][:100]}...")
    
    question = st.text_input("Ask the agent:")
    
    if question:
        with st.spinner("🤔 Agent reasoning..."):
            # Tool 1: Search documents
            q_embedding = model.encode([question])[0]
            similarities = np.dot(embeddings, q_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_embedding)
            )
            top_idx = np.argmax(similarities)
            
            # Tool 2: Entity extraction
            import re
            entities = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', all_chunks[top_idx])
            
            # Reasoning trace
            st.subheader("🧠 Agent Reasoning Trace")
            st.markdown(f"**Step 1:** Searched {len(all_chunks)} chunks")
            st.markdown(f"**Step 2:** Best match score: {similarities[top_idx]:.2f}")
            st.markdown(f"**Step 3:** Found {len(set(entities))} unique entities")
            
            # Answer
            st.subheader("📋 Agent Response")
            st.write(all_chunks[top_idx][:800])
            
            # Save to memory
            st.session_state.memory.append({
                'question': question,
                'entities': list(set(entities))[:5],
                'score': float(similarities[top_idx])
            })