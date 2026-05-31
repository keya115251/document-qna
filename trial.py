from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load your document
loader = TextLoader("acme_policy.txt")
docs = loader.load()

# 2. Chunk it
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embed + store
print("Embedding your document... this may take a moment.")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 4. Build the prompt
prompt = PromptTemplate.from_template("""
Answer the question using only the context below.
If the answer isn't in the context, say "I don't know."

Context: {context}

Question: {question}
""")

# 5. Build the chain
llm = OllamaLLM(model="llama3.2")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Ask questions
print("\nReady! Ask anything about your document.")
print("Type 'quit' to exit.\n")

while True:
    question = input("You: ")
    if question.strip().lower() == "quit":
        break
    if not question.strip():
        continue
    answer = chain.invoke(question)
    print(f"\nAnswer: {answer}\n")