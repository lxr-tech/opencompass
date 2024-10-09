import os
import torch
from langchain_community.document_loaders import PyMuPDFLoader

root_dir = 'RAG/langchain'

loader = PyMuPDFLoader(os.path.join(root_dir, "Virtual_characters.pdf"))
PDF_data = loader.load()

print(f"{PDF_data=}")

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
all_splits = text_splitter.split_documents(PDF_data)

print(f"{all_splits=}")

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
model_path = "~/models/llama2-7b"
model_path = os.path.expanduser(model_path)
model_kwargs = {'device': 'cpu'}
embedding = HuggingFaceEmbeddings(model_name=model_path, model_kwargs=model_kwargs)
embedding.client.tokenizer.pad_token = embedding.client.tokenizer.eos_token

print(f"{embedding=}")

# Embed and store the texts
from langchain_community.vectorstores import Chroma
persist_directory = 'db'
vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory=persist_directory)

retriever = vectordb.as_retriever()

from langchain.prompts import ChatPromptTemplate

template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

print(f"{prompt=}")


# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms import OpenAI, VLLMOpenAI, VLLM

# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm = VLLM(
    model=model_path,
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=256,
    temperature=0.8,
    streaming=True,
    dtype='bfloat16',
    enforce_eager=True,
)

# llm = VLLMOpenAI(
#     openai_api_key="EMPTY",
#     openai_api_base="http://localhost:8000/v1",
#     model_name=model_path,
#     model_kwargs={"stop":["."]},
# )

print("\nBegin RAG Chain...\n")

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# query = "What did the president say about Justice Breyer"
# print(rag_chain.invoke(query))

query = "What is Alison's attitude towards her personal life?"
answer = rag_chain.invoke(query)
print(answer)
print(type(answer))

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
)

query = "What is Alison's attitude towards her personal life?"
answer = rag_chain.invoke(query)
print(answer)

messages=[HumanMessage(content='You are an assistant for question-answering tasks.\nUse the following pieces of retrieved context to answer the question.\nIf you don\'t know the answer, just say that you don\'t know.\nUse three sentences maximum and keep the answer concise.\nQuestion: What is Alison\'s attitude towards her personal life?\nContext: [Document(metadata={\'author\': \'琪婕 黃\', \'creationDate\': "D:20240121015900+08\'00\'", \'creator\': \'適用於 Microsoft 365 的 Microsoft® Word\', \'file_path\': \'RAG/langchain/Virtual_characters.pdf\', \'format\': \'PDF 1.7\', \'keywords\': \'\', \'modDate\': "D:20240121015900+08\'00\'", \'page\': 0, \'producer\': \'適用於 Microsoft 365 的 Microsoft® Word\', \'source\': \'RAG/langchain/Virtual_characters.pdf\', \'subject\': \'\', \'title\': \'\', \'total_pages\': 1, \'trapped\': \'\'}, page_content=\'figure of intrigue and admiration in her professional and personal circles.\'), Document(metadata={\'author\': \'琪婕 黃\', \'creationDate\': "D:20240121015900+08\'00\'", \'creator\': \'適用於 Microsoft 365 的 Microsoft® Word\', \'file_path\': \'RAG/langchain/Virtual_characters.pdf\', \'format\': \'PDF 1.7\', \'keywords\': \'\', \'modDate\': "D:20240121015900+08\'00\'", \'page\': 0, \'producer\': \'適用於 Microsoft 365 的 Microsoft® Word\', \'source\': \'RAG/langchain/Virtual_characters.pdf\', \'subject\': \'\', \'title\': \'\', \'total_pages\': 1, \'trapped\': \'\'}, page_content=\'figure of intrigue and admiration in her professional and personal circles.\'), Document(metadata={\'author\': \'琪婕 黃\', \'creationDate\': "D:20240121015900+08\'00\'", \'creator\': \'適用於 Microsoft 365 的 Microsoft® Word\', \'file_path\': \'RAG/langchain/Virtual_characters.pdf\', \'format\': \'PDF 1.7\', \'keywords\': \'\', \'modDate\': "D:20240121015900+08\'00\'", \'page\': 0, \'producer\': \'適用於 Microsoft 365 的 Microsoft® Word\', \'source\': \'RAG/langchain/Virtual_characters.pdf\', \'subject\': \'\', \'title\': \'\', \'total_pages\': 1, \'trapped\': \'\'}, page_content=\'figure of intrigue and admiration in her professional and personal circles.\'), Document(metadata={\'author\': \'琪婕 黃\', \'creationDate\': "D:20240121015900+08\'00\'", \'creator\': \'適用於 Microsoft 365 的 Microsoft® Word\', \'file_path\': \'RAG/langchain/Virtual_characters.pdf\', \'format\': \'PDF 1.7\', \'keywords\': \'\', \'modDate\': "D:20240121015900+08\'00\'", \'page\': 0, \'producer\': \'適用於 Microsoft 365 的 Microsoft® Word\', \'source\': \'RAG/langchain/Virtual_characters.pdf\', \'subject\': \'\', \'title\': \'\', \'total_pages\': 1, \'trapped\': \'\'}, page_content=\'figure of intrigue and admiration in her professional and personal circles.\')]\nAnswer:\n', additional_kwargs={}, response_metadata={})]