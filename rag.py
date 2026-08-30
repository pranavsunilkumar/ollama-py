from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

from main import create_llm

def process_pdf(file_path:str="./c_programming_basics.pdf"):
    loader = PyPDFLoader(file_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks=splitter.split_documents(document)

    return chunks

embeddings=HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
    )

vector_store=Chroma(
    embedding_function=embeddings,
    collection_name="c_programming_basics",
    persist_directory="./vector_db"
)

def ingest_data():
    chunks=process_pdf()
    vector_store.add_documents(chunks)
    return "Data ingestion completed."

def rag_chain(query:str):
    llm=create_llm()
    retriever=vector_store.as_retriever(search_kwargs={"k":3})
    chain=RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    response = chain.invoke(query)
    return response['result']

if __name__ == "__main__":

    print("================================")
    print("       Simple RAG Agent")
    print("================================")

    print("\n1. Ingesting PDF...")
    print(ingest_data())

    print("\nRAG Agent is ready!")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        try:
            answer = rag_chain(user_query)

            print("\nAgent:", answer)
            print()
        except Exception as e:
            print("\nError:", e)
            print()