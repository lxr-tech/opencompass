from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import uvicorn
import json
import datetime
import torch
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import OpenAI, VLLMOpenAI, VLLM
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain.prompts import PromptTemplate
from moss_retriever import MossRetriever
import streamlit as st

# 设置设备参数
DEVICE = "cuda"  # 使用CUDA
DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息

# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片


Mossvllm = VLLM(
    model="/remote-home/share/models/internlm2-chat-20b/",
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=256,
    temperature=0.8,
    streaming=True,
)

custom_prompt_template = """

你是中国计算机学会青年计算机科技论坛（CCF YOCSEF 上海）的官方问答机器人。你很诚实，会根据搜索结果详细地解答用户的问题，且不会产生重复的，有攻击性或者有偏见的信息。默认使用中文回答问题。

{context}
聊天历史记录：{chat_history}
---

根据以上内容使用中文回答问题： {question}
"""


def document_data(query, chat_history):
    custom_prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question","chat_history"],
    )
    
    # retriever = load_embedding_model()
    retriever = MossRetriever()
    
   # ConversationalRetrievalChain 
    qa = ConversationalRetrievalChain.from_llm(
        llm = Mossvllm,
        retriever= retriever,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents = True
    )
    
    return qa({"question":query, "chat_history":chat_history})

# 创建FastAPI应用
app = FastAPI()

# 处理POST请求的端点
@app.post("/")
async def create_item(request: Request) :
    json_post_raw = await request.json()  # 获取POST请求的JSON数据
    json_post = json.dumps(json_post_raw)  # 将JSON数据转换为字符串
    json_post_list = json.loads(json_post)  # 将字符串转换为Python对象
    prompt = json_post_list.get('prompt')  # 获取请求中的提示
    history = json_post_list.get('history')  # 获取请求中的历史记录

    # 调用模型进行对话生成

    output=document_data(query=prompt, chat_history = history)
    
    now = datetime.datetime.now()  # 获取当前时间
    time = now.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间为字符串
    # 构建响应JSON
    answer = {
        "response": output['answer'],
        "history": history,
        "status": 200,
        "time": time
    }
    # 构建日志信息
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(output) + '"'
    print(log)  # 打印日志
    torch_gc()  # 执行GPU内存清理
    return answer  # 返回响应


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=30356, workers=1)
