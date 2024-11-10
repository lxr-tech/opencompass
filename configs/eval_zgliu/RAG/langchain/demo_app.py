import random
import json
import os.path
import requests
import numpy as np
from typing import Iterable, List, Tuple, Dict, Any
import streamlit as st
import jsonlines
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
from collections import defaultdict
from typing import Any
from openai import OpenAI
from tqdm import tqdm, trange

nltk.download('punkt')

logger = logging.getLogger(__name__)

model_path = 'jinaai/jina-embeddings-v2-base-zh'

def build_faiss_index(embedding_model, data_path):
    datas = []
    print('Begin loading data:')
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line)
            datas.append(json_data)
    print('Finish loading data!')


    # bm25_retriever

    bm25_documents = []
    for data in datas[:1000]:
        bm25_documents.append(data['article_info'])
    # 分词并构建BM25模型
    # tokenized_texts = [list(jieba.cut(text)) for text in bm25_documents]
    # tokenized_texts = [text.split() for text in bm25_documents]
    tokenized_texts = [mixed_segment(text) for text in bm25_documents]
    
    bm25 = BM25Okapi(tokenized_texts)
    
    if os.path.exists(os.path.join("data", "all_data.index")):
        print("Loading existing index.")
        index = read_index(os.path.join("data", "all_data.index"))
        print("Index loaded.")
        
    else:
        documents = []
        for data in datas:
            if len(data['article_info']) > 8192:
                data['article_info'] = data['article_info'][:8192]
            documents.append(data['article_info'])
        print('------------')
        index = faiss.IndexFlatL2(768)
        for i in range(13):
            print(i*10000,(i+1)*10000)
            embeddings = embedding_model.encode(documents[i*10000:(i+1)*10000])
            index.add(embeddings)
        print("The total index number is:", index.ntotal)
        print("Writing index to file system...")
        write_index(index, os.path.join("data", "all_data.index"))
        print("Index file saved.")

    return index, datas, bm25
    # return index, datas


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


@st.cache_resource
def load_embedding_model():
    with st.spinner('嵌入模型加载中...'):
        embedding_model = SentenceTransformer("jinaai/jina-embeddings-v2-base-zh", trust_remote_code=True)
    return embedding_model


@st.cache_resource
def load_faiss_index():
    with st.spinner('向量数据库加载中...'):
        embedding_model = load_embedding_model()
        index, datas, bm25= build_faiss_index(embedding_model=embedding_model, data_path="./data/all_data.jsonl")
        
    # st.success('向量数据库构建成功', icon='💫')
    return index, datas, bm25

embedding_distance = 0.55

def get_query_embedding(embedding_model, query):
    # query_embedding = embedding_model.encode([query])[0]
    query_embedding = embedding_model.encode([query])
    return query_embedding


def get_relevant_contexts(query_embedding, query,datas, faiss_index, bm25, k: int = 3):
    score, faiss_indices = faiss_index.search(query_embedding, k)
    # print('------------------')
    # print('jina result is:', score, faiss_indices[0])
    faiss_results = [(idx, 1 / (i + 1)) for i, idx in enumerate(faiss_indices[0])]
    for i in faiss_indices[0]:
        print(datas[i]['author_cn'], datas[i]['author_en'], datas[i]['title_cn'], datas[i]['title_en'])
    if score[0][0] > 1.3:
        have_ans = False
    else:
        have_ans = True
    # bm25_scores = bm25.get_scores(query.split())
    bm25_scores = bm25.get_scores(mixed_segment(query))
    
    bm25_results = [(i, score) for i, score in enumerate(bm25_scores)]
    bm25_results = sorted(bm25_results, key=lambda x: x[1], reverse=True)[:0]


    for idx, _ in bm25_results:
        print(datas[idx]['author_cn'], datas[idx]['author_en'], datas[idx]['title_cn'], datas[idx]['title_en'])

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


def response_each_query(query, embedding_model, index, datas, bm25):
    query_embedding = get_query_embedding(embedding_model, query)
    contexts, have_ans = get_relevant_contexts(query_embedding=query_embedding, query=query, datas=datas, faiss_index=index, bm25=bm25, k=10)
    if have_ans:
        response = generate_response(contexts)
    else:
        response = "抱歉你提问的问题我无法进行回答，我只能向你推荐你想要了解某个方向老师的相关信息，例如：“可以帮我推荐几个有关自然语言处理方向的老师吗？”或“我想要了解命名实体识别相关方向的内容，可以帮我推荐几个老师吗？”，请重新进行提问。"
    return response


def generate_response(context):
    response = "当然可以，以下是相关研究方向的教师的信息：\n"
    data = []
    for i in range(len(context)):
        res_one = ""
        one_data = {}
        # if context[i]['author_cn'] and context[i]['author_en'] and context[i]['author_email']:
        #     res_one = f"姓名：{context[i]['author_cn']} {context[i]['author_en']}，邮箱：{context[i]['author_email']}"
        # elif context[i]['author_en'] and context[i]['author_email']:
        #     res_one = f"姓名：{context[i]['author_en']}，邮箱：{context[i]['author_email']}"
        # elif context[i]['author_cn'] and context[i]['author_en']:
        #     res_one = f"姓名：{context[i]['author_cn']} {context[i]['author_en']}"
        # elif context[i]['author_cn']:
        #     res_one = f"姓名：{context[i]['author_cn']}"
        # else:
        #     res_one = f"姓名：{context[i]['author_en']}"
        if context[i]['author_cn'] and context[i]['author_en']:
            one_data['姓名'] = str(context[i]['author_cn']) + ' ' + str(context[i]['author_en'])
        elif context[i]['author_en']:
            one_data['姓名'] = str(context[i]['author_en'])
        else:
            one_data['姓名'] = str(context[i]['author_cn'])
        
        if context[i]['author_email']:
            one_data['邮箱'] = str(context[i]['author_email'])
        
        # # index = response.find(res_one)
        # if context[i]['title_cn'] and context[i]['title_en']:
        #     res_one += f"，相关论文：{context[i]['title_en']}（{context[i]['title_cn']}）"
        # elif context[i]['title_en']:
        #     res_one += f"，相关论文：{context[i]['title_en']}"
        # else:
        #     res_one += f"，相关论文：{context[i]['title_cn']}"
        if context[i]['title_cn'] and context[i]['title_en']:
            one_data['论文'] = str(context[i]['title_en']) + '(' + str(context[i]['title_cn']) + ')'
        elif context[i]['title_en']:
            one_data['论文'] = str(context[i]['title_en'])
        else:
            one_data['论文'] = str(context[i]['title_cn'])
        
        data.append(one_data)
        # if not response.find(res_one) == -1:
        #     continue
        # response += res_one + '\n'
    
    # 初始化一个空的字典来存储分组数据
    grouped_data = {}

    # 通过老师的姓名进行分组
    for entry in data:
        name = entry["姓名"]
        if name not in grouped_data:
            grouped_data[name] = { "邮箱": entry["邮箱"], "论文列表": []}
        grouped_data[name]["论文列表"].append(entry["论文"])

    # 输出分组后的结果
    for name, info in grouped_data.items():
        response += f"姓名：{name}，邮箱：{info['邮箱']} \n"
        response += f"相关论文：\n"
        for paper in info["论文列表"]:
            response += f" --- {paper}\n"
        response += '\n'
    return response



def call_gpt(client: Any,
             model: str,
             prompt: str,
             sys_prompt: str = "",
             n: int = 1,
             return_json: bool = False) -> str:
    """Perform a single api call with specified model and prompt."""
    if prompt is None or len(prompt) == 0:
        logger.warning("Prompt is empty!!!")
    if "gpt" in model or "deepseek" in model:
        if model in [
                "gpt-3.5-turbo", "gpt-4", "gpt-4o-2024-08-06", "gpt-4o-mini",
                "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-1106", "deepseek-chat"
        ]:
            if return_json:
                response = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": sys_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt
                        },
                    ],
                    max_tokens=480,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": sys_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt
                        },
                    ],
                    max_tokens=480,
                    top_p=1,
                    temperature=1,
                    n=n
                )
            if n == 1:
                msg = response.choices[0].message
                assert msg.role == "assistant", "Incorrect role returned."
                ans = msg.content
            else:
                ans = [msg.message.content for msg in response.choices]
        elif model in ["gpt-3.5-turbo-instruct"]:
            if len(sys_prompt) > 0:
                prompt = sys_prompt + prompt
            response = client.completions.create(model=model,
                                                 prompt=prompt,
                                                 max_tokens=480)
            ans = response.choices[0].text
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return ans




# def load_stopwords():
#     stopwords = set()
#     with open('stopwords.txt', 'r', encoding='utf-8') as f:
#         for line in f:
#             stopwords.add(line.strip())
#     return stopwords

# def chinese_tokenize(text, stopwords):
#     words = jieba.cut(text)
#     filtered_words = [word for word in words if word not in stopwords]
#     return filtered_words

# def cut_word(text):
#     return list(jieba.lcut(text))

# def bm25_search(query, k, index):
#     bm25_query = query
#     stopwords=load_stopwords()
#     query_delete_stopwords = chinese_tokenize(text=bm25_query, stopwords=stopwords)
#     bm25_query_dele_stopword = ''.join(query_delete_stopwords)
#     bm25_retriever = BM25Retriever.from_documents(index, preprocess_func=cut_word, k=k)
#     bm25_matching_documents_1 = bm25_retriever.get_relevant_documents(bm25_query_dele_stopword)
#     # bm25_data_1 = bm25_matching_documents_1[0].page_content
#     print(bm25_matching_documents_1)

# def search_faiss(query):
#     # search_results = retrieve_multiple_responses(query, k=3)
#     # output = reciprocal_rank_fusion(search_results_dict = search_results)
#     # print('-------------------')
#     # print(output)
#     result = ""
#     docs_and_scores = vector_db.similarity_search_with_relevance_scores(query=query, k=5)
#     for i in range(len(docs_and_scores)):
#         if docs_and_scores[i][0].metadata['author_cn'] and docs_and_scores[i][0].metadata['author_en'] and docs_and_scores[i][0].metadata['author_email']:
#             resstr = f"姓名：{docs_and_scores[i][0].metadata['author_cn']} {docs_and_scores[i][0].metadata['author_en']}， 邮箱：{docs_and_scores[i][0].metadata['author_email']}"
#         elif docs_and_scores[i][0].metadata['author_en'] and docs_and_scores[i][0].metadata['author_email']:   
#             resstr = f"姓名：{docs_and_scores[i][0].metadata['author_en']}， 邮箱：{docs_and_scores[i][0].metadata['author_email']}"
#         elif docs_and_scores[i][0].metadata['author_cn'] and docs_and_scores[i][0].metadata['author_en']:   
#             resstr = f"姓名：{docs_and_scores[i][0].metadata['author_cn']} {docs_and_scores[i][0].metadata['author_en']}"
#         else:
#             resstr = f"姓名：{docs_and_scores[i][0].metadata['author_en']}"
#         result = result + str(resstr) + '/n\n'
#     return result


deepseek_key = "sk-f3755f6e06dd445aa1fb494ff475a55a"
deepseek_url = "https://api.deepseek.com"

client = OpenAI(
    api_key=deepseek_key,
    base_url=deepseek_url,
    max_retries=100,
    timeout=20.0,
)


def main():
    st.set_page_config(page_title="旦融慧通/复联慧桥", page_icon="", layout="wide")
    st.header(":nazar_amulet: 旦融慧通/复联慧桥", divider='blue')
    st.caption(':blue[我们是一个跨学科交叉研究的连接平台，同学和老师们在这里可以轻松地问询到自己感兴趣的、与研究课题相关的多源学科的大咖信息。]')

    
    with st.sidebar:
        st.header("热点大咖")

    
    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"]=[]
    if "chat_answers_history" not in st.session_state:
        st.session_state["chat_answers_history"]=[]
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"]=[]
    embedding_model = load_embedding_model()
    index, datas, bm25 = load_faiss_index()
    
    prompt = st.chat_input("Enter your questions here")

    if prompt:
        with st.spinner("Generating......"):
            # PROMPT = """请你扮演一个查询改写器，将下面这段话提取出与学术研究相关的实体并将提取出的实体翻译成英文返回，请仅返回翻译后的英文：{} """
            PROMPT = """请你扮演一个翻译器，将下面这段话翻译成英文，请仅返回翻译后的内容：{}"""
            query = PROMPT.format(prompt)
            # "gpt-4o-2024-08-06", gpt-4o-mini, deepseek-chat
            response = call_gpt(client=client, model="deepseek-chat", prompt=query, n=1)
            # print('翻译后response为：',response)

            # Storing the questions, answers and chat history
            # ans = search_faiss(query=prompt)
            ans = response_each_query(response, embedding_model, index, datas, bm25)
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(ans)
            st.session_state["chat_history"].append((prompt,ans))


    # Displaying the chat history

    if st.session_state["chat_answers_history"]:
        for i, j in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
            message1 = st.chat_message("user")
            # message1.write(j)
            message1.write(j.replace('\n', '  \n'))
            message2 = st.chat_message("assistant")
            # message2.write(i)
            message2.write(i.replace('\n', '  \n'))
        

# python -m streamlit run demo_app.py --server.port 6006
if __name__ == "__main__":
    main()
    # ROBUST KERNEL DESIGN FOR SSD TRACKER SSD-based object
    
    # query = '复旦大学图书馆期刊回溯项目管理'
    # embedding_model = load_embedding_model()
    # index, datas = load_faiss_index()
    # ans = response_each_query(query, embedding_model, index, datas)
    # bm25_search(query=query, k=3, index= index)

