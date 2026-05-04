from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from ddgs import DDGS
import pandas as pd
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load data & build RAG on startup ─────────────────────────────
@app.on_event("startup")
async def startup():
    global rag_chain, llm

    # LLM
    llm = ChatGroq(
        api_key=os.environ["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile"
)

    # Load Excel
    df = pd.read_excel("wordpress_ecoilk_data.xlsx")
    df = df[["title", "content", "date"]].dropna(subset=["content"])
    df["text"] = df.apply(
        lambda row: f"Title: {row['title']}\nDate: {row['date']}\nContent: {row['content']}",
        axis=1
    )

    # Embeddings & vector store
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 32, "normalize_embeddings": True}
    )
    docs = [
    Document(
        page_content=row["text"],
        metadata={"title": row["title"], "date": str(row["date"])}
    )
    for _, row in df.iterrows()
    ]
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join([d.page_content[:800] for d in docs])

    prompt = PromptTemplate.from_template("""
You are a helpful assistant for the ECOILK website.
Use the context below to answer the question as fully as possible.
If the answer is not in the context, say "I don't have that information".

Context:
{context}

Question:
{question}

Answer:
""")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# ── Web search ────────────────────────────────────────────────────
def web_search(query: str) -> str:
    try:
        results = DDGS().text(query, max_results=3)
        return "\n\n---\n\n".join([
            f"Title: {r.get('title')}\nSummary: {r.get('body')[:200]}"
            for r in results
        ])
    except Exception as e:
        return f"Web search failed: {str(e)}"

# ── Endpoints ─────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_endpoint(request: QuestionRequest):
    question = request.question
    rag_answer = rag_chain.invoke(question)

    unhelpful_phrases = [
        "i don't have that information",
        "not in the context",
        "no information",
        "cannot find",
        "not available"
    ]

    is_unhelpful = any(phrase in rag_answer.lower() for phrase in unhelpful_phrases)

    if is_unhelpful or len(rag_answer.strip()) < 20:
        web_answer = web_search(question)
        final = llm.invoke(f"""Based on these search results, answer: '{question}'
Search results:
{web_answer}
Give a clear, concise answer.""")
        return {"answer": final.content, "source": "web"}
    else:
        return {"answer": rag_answer, "source": "website_data"}

@app.get("/health")
async def health():
    return {"status": "running"}