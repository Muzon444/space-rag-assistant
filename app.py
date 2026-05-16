import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# Page settings
# -----------------------------

st.set_page_config(
    page_title="Space Knowledge RAG Assistant",
    page_icon="🚀",
    layout="centered"
)

# -----------------------------
# Space Theme
# -----------------------------

st.markdown("""
<style>

.stApp{
background:
radial-gradient(circle at 20% 30%, white 1px, transparent 2px),
radial-gradient(circle at 80% 20%, white 1px, transparent 2px),
radial-gradient(circle at 30% 70%, white 2px, transparent 3px),
radial-gradient(circle at 60% 80%, white 1px, transparent 2px),
radial-gradient(circle at 90% 60%, white 2px, transparent 3px),
radial-gradient(circle at 50% 40%, white 1px, transparent 2px),
linear-gradient(to bottom,#050816,#0B1F3A);

background-size:
200px 200px,
250px 250px,
300px 300px,
180px 180px,
220px 220px,
150px 150px,
100% 100%;

color:white;
}

h1,h2,h3,p,label{
color:white !important;
}

.stTextInput input{
background-color:#0d1b2a !important;
color:#87CEFA !important;
font-size:20px !important;
border:2px solid #4da6ff !important;
border-radius:15px !important;
}

.stTextInput input::placeholder{
color:#d6d6d6 !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Read document
# -----------------------------

def load_document():

    with open(
        "documents/space_notes.txt",
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


# -----------------------------
# Chunking
# -----------------------------

def chunk_text(
    text,
    chunk_size=700,
    overlap=100
):

    chunks=[]

    start=0

    while start < len(text):

        end=start+chunk_size

        chunk=text[start:end]

        if chunk.strip():

            chunks.append(chunk)

        start+=chunk_size-overlap

    return chunks


# -----------------------------
# Build RAG
# -----------------------------

@st.cache_resource
def build_rag():

    text=load_document()

    chunks=chunk_text(text)

    model=SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings=model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension=embeddings.shape[1]

    index=faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return model,index,chunks


model,index,chunks=build_rag()


# -----------------------------
# Retrieval
# -----------------------------

def retrieve(query, top_k=2):

    query_embedding=model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores,indices=index.search(
        query_embedding,
        top_k
    )

    results=[]

    for score,idx in zip(
        scores[0],
        indices[0]
    ):

        results.append({

            "chunk":chunks[idx],
            "score":float(score)

        })

    return results


# -----------------------------
# Answer generation
# -----------------------------

def generate_answer(
    question,
    retrieved_chunk
):

    sentences=retrieved_chunk.split(".")

    question_words=[

        word.lower()
        .replace("?","")

        for word in question.split()

        if len(word)>3
    ]

    best_sentence=""

    max_matches=0

    for sentence in sentences:

        sentence_lower=sentence.lower()

        matches=sum(

            word in sentence_lower

            for word in question_words
        )

        if matches>max_matches:

            max_matches=matches

            best_sentence=sentence.strip()


    if best_sentence:

        return best_sentence+"."


    return sentences[0].strip()+"."


# -----------------------------
# UI
# -----------------------------

st.title(
"🚀 Space Knowledge RAG Assistant"
)

st.write(
"Ask questions about my local space document using embeddings and FAISS retrieval."
)

question=st.text_input(
"Ask a question"
)

if question:

    results=retrieve(question)

    answer=generate_answer(
        question,
        results[0]["chunk"]
    )

    st.subheader(
        "Generated Answer"
    )

    st.info(
        answer
    )

    st.subheader(
        "Retrieved Context & Sources"
    )

    for i,result in enumerate(
        results,
        start=1
    ):

        with st.container(
            border=True
        ):

            st.markdown(
                f"**Chunk {i}**"
            )

            st.write(
                result["chunk"]
            )

            st.caption(
                "📄 Source: space_notes.txt"
            )

            st.caption(
                f"Similarity Score: {result['score']:.2f}"
            )