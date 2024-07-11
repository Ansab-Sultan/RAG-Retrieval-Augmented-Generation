import streamlit as st
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

CHROMA_PATH = "Database"
DATA_PATH = "Data"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key="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")
    response = llm.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    # Assuming response is an AIMessage object, extract content appropriately
    formatted_response = {
        "response": response.content,
        "sources": sources
    }
    return formatted_response

st.title("Query System")


query_text = st.text_input("Enter your query:")

if query_text and st.button("Query Database"):
    response_data = query_rag(query_text)
    response_text = response_data["response"]
    sources = response_data["sources"]
    
    # Use st.markdown for better formatting
    st.markdown("### Response")
    st.markdown(response_text)
    
    st.markdown("### Sources")
    st.markdown("\n".join(f"- {source}" for source in sources))
