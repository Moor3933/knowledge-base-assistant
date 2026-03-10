"""
个人知识库及问答小助理
Personal Knowledge Base & Q&A Assistant

基于 Streamlit + Supabase + 硅基流动 构建
"""

import streamlit as st
import fitz  # PyMuPDF
import base64
import io
import os
import json
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

# 第三方库
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import httpx

# ==================== 配置 ====================
st.set_page_config(
    page_title="个人知识库问答助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .source-card {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
    }
    .image-caption {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
    }
    .stButton>button {
        width: 100%;
    }
    .sidebar-section {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 初始化 ====================
@st.cache_resource
def init_supabase():
    """初始化Supabase客户端"""
    SUPABASE_URL = "https://qrocrjmmueonzgyoxgqk.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFyb2Nyam1tdWVvbnpneW94Z3FrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzMxNTY3MzIsImV4cCI6MjA4ODczMjczMn0.8-SnRrd5fowJIkzB0zApM0Wc8GrmdRCxXKXwGzGkXFE"
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def init_embedding_model():
    """初始化Embedding模型"""
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def init_siliconflow_client():
    """初始化硅基流动API客户端"""
    return httpx.Client(
        base_url="https://api.siliconflow.cn/v1",
        headers={
            "Authorization": "Bearer sk-wuzvlfqljczecidghokvjeiqjedgwwktyqfsmnxkjavvomln",
            "Content-Type": "application/json"
        },
        timeout=120.0
    )

# ==================== 工具函数 ====================
def get_embedding(text: str, model) -> List[float]:
    """获取文本的embedding向量"""
    return model.encode(text, convert_to_numpy=True).tolist()

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算余弦相似度"""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ==================== PDF处理 ====================
def extract_text_from_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """从PDF提取文本和图片"""
    chunks = []
    doc = fitz.open(stream=pdf_bytes, doc_type="pdf")

    for page_num in range(len(doc)):
        page = doc[page_num]

        # 提取文本
        text = page.get_text()
        if text.strip():
            chunks.append({
                "type": "text",
                "content": text,
                "page": page_num + 1,
                "metadata": {}
            })

        # 提取图片
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            chunks.append({
                "type": "image",
                "content": "",
                "page": page_num + 1,
                "metadata": {
                    "image_bytes": base64.b64encode(image_bytes).decode('utf-8'),
                    "width": base_image["width"],
                    "height": base_image["height"],
                    "xref": xref
                }
            })

    doc.close()
    return chunks

def extract_text_from_docx(docx_bytes: bytes) -> List[Dict[str, Any]]:
    """从DOCX提取文本"""
    return [{"type": "text", "content": "DOCX文件解析需要python-docx库", "page": 1, "metadata": {}}]

def extract_text_from_txt(txt_bytes: bytes) -> List[Dict[str, Any]]:
    """从TXT提取文本"""
    text = txt_bytes.decode('utf-8', errors='ignore')
    return [{"type": "text", "content": text, "page": 1, "metadata": {}}]

# ==================== AI功能 ====================
def understand_image_with_ai(image_base64: str, client: httpx.Client) -> str:
    """使用AI理解图片内容"""
    try:
        image_data = f"data:image/jpeg;base64,{image_base64}"
        payload = {
            "model": "Qwen/Qwen2-VL-72B-Instruct",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data}},
                    {"type": "text", "text": "请详细描述这张图片的内容。如果是图表，请解释其横纵坐标、数据趋势和核心结论。"}
                ]
            }],
            "max_tokens": 500
        }
        response = client.post("/chat/completions", json=payload)
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        return "图片理解失败"
    except Exception as e:
        return f"图片理解出错: {str(e)}"

def chat_with_ai(query: str, context: str, client: httpx.Client) -> str:
    """使用AI进行问答"""
    try:
        system_prompt = f"""你是一个知识渊博的助手，负责根据提供的知识库内容回答用户的问题。

要求：
1. 只根据提供的上下文信息回答
2. 如果上下文中没有相关信息，请如实说明
3. 回答要清晰、准确、简洁

参考知识库内容：
{context}

请根据以上知识库内容回答用户的问题。"""

        payload = {
            "model": "deepseek-ai/DeepSeek-R1",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "max_tokens": 2000,
            "temperature": 0.7
        }
        response = client.post("/chat/completions", json=payload)
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        return "生成回答失败"
    except Exception as e:
        return f"AI生成回答出错: {str(e)}"
# ==================== 数据库操作 ====================
def save_document_to_db(supabase: Client, filename: str, file_size: int) -> str:
    """保存文档信息到数据库"""
    result = supabase.table("documents").insert({
        "filename": filename,
        "file_size": file_size,
        "status": "processing"
    }).execute()
    if result.data and len(result.data) > 0:
        return result.data[0]["id"]
    return None

def update_document_status(supabase: Client, doc_id: str, status: str):
    """更新文档状态"""
    supabase.table("documents").update({"status": status}).eq("id", doc_id).execute()

def save_chunks_to_db(supabase: Client, doc_id: str, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
    """保存知识片段到数据库"""
    for chunk, embedding in zip(chunks, embeddings):
        supabase.table("document_chunks").insert({
            "document_id": doc_id,
            "content": chunk["content"],
            "embedding": embedding,
            "page_number": chunk["page"],
            "chunk_type": chunk["type"],
            "image_data": chunk.get("metadata", {}).get("image_bytes", ""),
            "original_caption": chunk.get("metadata", {}).get("caption", ""),
            "metadata": json.dumps(chunk.get("metadata", {}))
        }).execute()

def search_similar_chunks(supabase: Client, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """搜索相似的知识片段"""
    result = supabase.table("document_chunks").select("*").execute()
    if not result.data:
        return []
    
    similarities = []
    for item in result.data:
        if item["embedding"]:
            sim = cosine_similarity(query_embedding, item["embedding"])
            similarities.append((sim, item))
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [item for sim, item in similarities[:top_k]]

def get_all_documents(supabase: Client) -> List[Dict[str, Any]]:
    """获取所有文档"""
    result = supabase.table("documents").select("*").order("upload_date", desc=True).execute()
    return result.data if result.data else []

def delete_document(supabase: Client, doc_id: str):
    """删除文档"""
    supabase.table("document_chunks").delete().eq("document_id", doc_id).execute()
    supabase.table("documents").delete().eq("id", doc_id).execute()

# ==================== UI组件 ====================
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("📚 知识库助手")
        st.markdown("---")
        
        # 文件上传
        st.subheader("📤 上传文档")
        uploaded_file = st.file_uploader("选择PDF文件", type=["pdf"], help="支持PDF格式文档")
        
        if uploaded_file:
            if st.button("📊 处理文档", type="primary"):
                process_uploaded_file(uploaded_file)
        
        st.markdown("---")
        
        # 文档管理
        st.subheader("📁 我的文档")
        documents = get_all_documents(st.session_state.supabase)
        
        if documents:
            for doc in documents:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(doc["filename"][:20] + "..." if len(doc["filename"]) > 20 else doc["filename"])
                with col2:
                    if st.button("🗑️", key=f"del_{doc['id']}"):
                        delete_document(st.session_state.supabase, doc["id"])
                        st.rerun()
                
                status_color = "🟢" if doc["status"] == "completed" else "🟡"
                st.caption(f"{status_color} {doc['status']} | {doc['upload_date'][:10]}")
        else:
            st.info("暂无文档")
        
        st.markdown("---")
        
        # API信息
        st.subheader("⚙️ API状态")
        st.success("✅ 硅基流动: 已连接")
        st.success("✅ Supabase: 已连接")
        
        if st.button("🗑️ 清空对话历史"):
            st.session_state.messages = []
            st.rerun()

def process_uploaded_file(uploaded_file):
    """处理上传的文件"""
    with st.spinner("正在处理文档..."):
        try:
            pdf_bytes = uploaded_file.getvalue()
            chunks = extract_text_from_pdf(pdf_bytes)
            
            doc_id = save_document_to_db(st.session_state.supabase, uploaded_file.name, len(pdf_bytes))
            if not doc_id:
                st.error("保存文档信息失败")
                return
            
            sf_client = init_siliconflow_client()
            processed_chunks = []
            embeddings = []
            embedding_model = init_embedding_model()
            
            progress_bar = st.progress(0)
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks):
                if chunk["type"] == "image":
                    image_bytes = chunk.get("metadata", {}).get("image_bytes", "")
                    if image_bytes:
                        with st.spinner(f"正在理解第 {i+1}/{total_chunks} 个内容（图片）..."):
                            description = understand_image_with_ai(image_bytes, sf_client)
                            chunk["content"] = description
                
                if chunk["content"].strip():
                    embedding = get_embedding(chunk["content"], embedding_model)
                    processed_chunks.append(chunk)
                    embeddings.append(embedding)
                
                progress_bar.progress((i + 1) / total_chunks)
            
            if processed_chunks:
                save_chunks_to_db(st.session_state.supabase, doc_id, processed_chunks, embeddings)
                update_document_status(st.session_state.supabase, doc_id, "completed")
                st.success(f"✅ 文档处理完成！共提取 {len(processed_chunks)} 个知识片段")
            else:
                st.warning("⚠️ 未能提取到有效内容")
            
            sf_client.close()
            st.rerun()
        except Exception as e:
            st.error(f"处理文档时出错: {str(e)}")
def render_chat_interface():
    """渲染聊天界面"""
    st.title("💬 知识问答")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📚 参考来源"):
                        for idx, source in enumerate(message["sources"], 1):
                            if source["chunk_type"] == "image":
                                st.markdown(f"**来源 {idx}: 图片** (第{source['page_number']}页)")
                                if source.get("image_data"):
                                    try:
                                        img_bytes = base64.b64decode(source["image_data"])
                                        st.image(img_bytes, width=300)
                                    except:
                                        pass
                                st.caption(source["content"][:200] + "...")
                            else:
                                st.markdown(f"**来源 {idx}: 文本** (第{source['page_number']}页)")
                                st.caption(source["content"][:200] + "...")
    
    if query := st.chat_input("请输入你的问题..."):
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    embedding_model = init_embedding_model()
                    query_embedding = get_embedding(query, embedding_model)
                    
                    sources = search_similar_chunks(st.session_state.supabase, query_embedding, top_k=5)
                    
                    if not sources:
                        response = "抱歉，知识库中还没有找到相关信息。请先上传一些文档到知识库中。"
                        st.markdown(response)
                    else:
                        context = "\n\n".join([
                            f"[来源{i+1} - {s['chunk_type']} - 第{s['page_number']}页]\n{s['content'][:500]}"
                            for i, s in enumerate(sources)
                        ])
                        
                        sf_client = init_siliconflow_client()
                        response = chat_with_ai(query, context, sf_client)
                        sf_client.close()
                        
                        st.markdown(response)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": sources
                        })
                except Exception as e:
                    st.error(f"生成回答时出错: {str(e)}")

# ==================== 主程序 ====================
def main():
    """主程序入口"""
    if "supabase" not in st.session_state:
        st.session_state.supabase = init_supabase()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    render_sidebar()
    render_chat_interface()

if __name__ == "__main__":
    main()