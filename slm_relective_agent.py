import os
import json
import datetime
from llama_cpp import Llama
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Load Small Language Model (SLM)
llm = Llama(
    model_path="../downloaded_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",  # update local folder path as per user
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=0  # Set >0 if using GPU layers
)

# Load and Index Local PDFs
folder_path = "./Docs_Test/" # Place your PDFs .. Example: Some research papers related to Mistral

def load_all_pdfs_from_folder(folder_path):
    docs = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
    return docs

docs = load_all_pdfs_from_folder(folder_path)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(docs, embedding, persist_directory="./chroma_index")
vectordb.persist()

def truncate_context(context_docs, max_chars=3000):
    total = ""
    for doc in context_docs:
        if len(total) + len(doc.page_content) <= max_chars:
            total += "\n\n" + doc.page_content
        else:
            break
    return total.strip()

def query_pdf(query):
    context = vectordb.similarity_search(query, k=2)

    if not context:
        print("No relevant documents found in vector store.")
        return "Sorry, I couldn't find anything relevant in the documents to answer your question."

    print(f"Retrieved {len(context)} documents from vectordb.")
    joined = truncate_context(context, max_chars=3000)

    prompt = f"""<s>[INST] You are a research assistant.
Based on the following context, answer the question thoroughly and clearly.

Context:
{joined}

Question:
{query}

Please provide a complete and detailed answer. [/INST]"""

    response = llm(prompt, max_tokens=512, stop=[])
    answer = response['choices'][0]['text'].strip()

    if len(answer.split()) < 5:
        print("Generated answer appears too short. Returning fallback message.")
        return "The answer was incomplete. Please rephrase the question or check document relevance."

    print("\nAnswer:\n", answer)
    return answer

def reflect_on_answer(query, answer):
    prompt = f"""<s>[INST] You just answered this question:
"{query}"

Your answer was:
"{answer}"

Was this a good response? Should it be improved?
Reply in this format:
- Verdict: [Good/Improve]
- Reason:
- Improved Answer (if needed): [/INST]"""
    response = llm(prompt)
    reflection = response['choices'][0]['text'].strip()
    print("\nReflection:\n", reflection)
    return reflection

def save_episode(query, answer, feedback):
    episode = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "feedback": feedback
    }
    with open("logbook.jsonl", "a") as f:
        f.write(json.dumps(episode) + "\n")
    print("\nEpisode saved to logbook.jsonl")

def need_tool(query):
    prompt = f"""<s>[INST] Do you need to call a tool to answer this?
Question: {query}
Reply YES or NO and explain why. [/INST]"""
    response = llm(prompt)
    decision = response['choices'][0]['text'].strip()
    print("\nTool Decision:", decision)
    return decision

def agent_loop(query):
    if not query.strip():
        print("Empty query. Skipping.")
        return

    print("\nQuery:", query)
    decision = need_tool(query)
    if "NO" not in decision.upper():
        print("\nDecision: Tool required — skipping execution.")
        return

    answer = query_pdf(query)
    if "incomplete" in answer.lower():
        print("Skipping reflection due to weak or fallback answer.")
        return

    feedback = reflect_on_answer(query, answer)
