import os
import torch
from langchain_community.document_loaders import UnstructuredTSVLoader

# data_path = "RAG/data/collection.tsv"
def main():
    
    data_path = 'RAG/data/msmarco-test2019-queries.tsv'

    loader = UnstructuredTSVLoader(data_path)
    tsvdata = loader.load()

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    all_splits = text_splitter.split_documents(tsvdata)

    # print(f"{all_splits=}")

    from langchain_huggingface import HuggingFaceEmbeddings
    model_path = "~/models/llama2-7b"
    model_path = os.path.expanduser(model_path)
    model_kwargs = {'device': 'cuda', 'model_kwargs': {'torch_dtype': torch.bfloat16}}
    embedding = HuggingFaceEmbeddings(model_name=model_path,
                                    model_kwargs=model_kwargs,
                                    multi_process=True,
                                    show_progress=True,
                                    )
    embedding.client.tokenizer.pad_token = embedding.client.tokenizer.eos_token

    from langchain_chroma import Chroma

    persist_directory = 'RAG/data/collection_db'
    vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory=persist_directory)

    print(f"{vectordb=}")

if __name__ == '__main__':
    main()