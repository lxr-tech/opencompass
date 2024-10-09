from sentence_transformers import SentenceTransformer
from faiss import write_index, read_index
import faiss
import jieba
from typing import List, Dict
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import os
import json
import random
import logging
from tqdm import tqdm, trange
from joblib import Parallel, delayed

nltk.download('punkt')
nltk.download('punkt_tab')

def mixed_segment(text):
    # 使用jieba进行中文分词
    words = jieba.cut(text)
    # 将分词结果转换为列表
    words = list(words)
    
    # 创建一个存储分词结果的列表
    segmented_words = []
    
    for word in words:
        # 如果是英文或数字，使用nltk的word_tokenize进行分词
        if word == ' ':
            pass
        elif word.isalpha():
            segmented_words.extend(word_tokenize(word))
        else:
            segmented_words.append(word)
    
    return segmented_words

def build_faiss_index(embedding_model, data_path):
    datas = []
    print('Begin loading data:')
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            datas.append(line.split('\t', 1)[-1])
    print('Finish loading data!')

    print("length of datas:", len(datas))
    
    sample_num = 10000
    if sample_num < len(datas) and sample_num > 0:
        datas = datas[:sample_num]

    # bm25_retriever

    bm25_documents = []
    for data in datas:
        bm25_documents.append(data)
    
    # 分词并构建BM25模型
    print("Building BM25 model...")
    # tokenized_texts = [mixed_segment(text) for text in bm25_documents]
    tokenized_texts = []
    for text in tqdm(bm25_documents, desc="Tokenizing documents..."):
        tokenized_texts.append(mixed_segment(text))
    bm25 = BM25Okapi(tokenized_texts)
    print("BM25 model built.")

    save_dataidx_dir = os.path.dirname(data_path)
    save_dataidx_path = os.path.join(save_dataidx_dir, "all_data.index")
    
    print("Building faiss index...")
    if os.path.exists(save_dataidx_path):
        print("Loading existing index.")
        index = read_index(save_dataidx_path)
        print("Index loaded.")
    else:
        documents = []
        for data in datas:
            if len(data) > 8192:
                data = data[:8192]
            documents.append(data)
       # 计算文档的总长度
        len_documents = len(documents)
        print(f"Total number of documents: {len_documents}")

        # 初始化 faiss 索引，假设 768 是向量的维度
        index = faiss.IndexFlatL2(768)

        # 每次处理 10000 个文档
        batch_size = 10000
        num_batches = (len_documents + batch_size - 1) // batch_size  # 向上取整

        # 遍历批次，进行编码并添加到索引中
        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, len_documents)  # 确保不越界
            # print(f"Processing batch {i + 1}/{num_batches}: documents {start} to {end}")
            # 对当前批次的文档进行编码
            embeddings = embedding_model.encode(documents[start:end])
            # 将编码后的向量添加到索引中
            index.add(embeddings)
        print("The total index number is:", index.ntotal)
        print("Writing index to file system...")
        write_index(index, save_dataidx_path)
        print("Index file saved.")

    return index, datas, bm25
    # return index, datas

def load_embedding_model(model_path):
    embedding_model = SentenceTransformer(model_path,
                                          trust_remote_code=True,
                                        )
    return embedding_model

def load_faiss_index(embedding_model, data_path):
    index, datas, bm25= build_faiss_index(embedding_model=embedding_model, data_path=data_path)

    return index, datas, bm25

def get_query_embedding(embedding_model, query):
    # query_embedding = embedding_model.encode([query])[0]
    query_embedding = embedding_model.encode([query])
    return query_embedding

def get_relevant_contexts(query_embedding, query, datas, faiss_index, bm25, k: int = 3):
    score, faiss_indices = faiss_index.search(query_embedding, k)
    # print('------------------')
    # print('jina result is:', score, faiss_indices[0])
    faiss_results = [(idx, 1 / (i + 1)) for i, idx in enumerate(faiss_indices[0])]
    # for i in faiss_indices[0]:
    #     print(datas[i])
    if score[0][0] > 1.3:
        have_ans = False
    else:
        have_ans = True
    # bm25_scores = bm25.get_scores(query.split())
    bm25_scores = bm25.get_scores(mixed_segment(query))
    
    bm25_results = [(i, score) for i, score in enumerate(bm25_scores)]
    bm25_results = sorted(bm25_results, key=lambda x: x[1], reverse=True)[:0]


    # for idx, _ in bm25_results:
    #     print(datas[idx])

    print(f"{len(faiss_results)=}, {len(bm25_results)=}")

    # 倒数融合排序（RRF）
    fused_results = reciprocal_rank_fusion([faiss_results, bm25_results])
    
    # 返回倒数排序融合的前k个结果
    top_results = fused_results[:10]
    # print('-------')
    # print('final result is', [datas[idx] for idx, _ in top_results])
    return [datas[idx] for idx, _ in top_results], have_ans

def reciprocal_rank_fusion(results, k=60):
    rank_scores = {}
    for rank_list in results:
        for idx, (doc_id, score) in enumerate(rank_list):
            if doc_id not in rank_scores:
                rank_scores[doc_id] = 0
            rank_scores[doc_id] += 1 / (idx + k)

    sorted_scores = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:k]

def main(model_path, data_path, query):
    embedding_model = load_embedding_model(model_path)
    index, datas, bm25 = load_faiss_index(embedding_model, data_path)

    query_embedding = get_query_embedding(embedding_model, query)
    contexts, have_ans = get_relevant_contexts(query_embedding=query_embedding, query=query, datas=datas, faiss_index=index, bm25=bm25, k=10)
    print(f"{contexts=}")
    print(f"{have_ans=}")

if __name__ == "__main__":
    model_path = '~/models/jina-embeddings-v2-base-zh'
    model_path = os.path.expanduser(model_path)
    # data_path = 'RAG/data/msmarco-test2019-queries.tsv'
    data_path = 'RAG/data/collection.tsv'

    main(model_path, data_path, 'How can I learn PowerPoint Keyboard Shortcuts?')