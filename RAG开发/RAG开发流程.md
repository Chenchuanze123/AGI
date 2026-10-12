**RAG = Retrieval-Augmented Generation，检索增强生成**。  
一句话：**让大模型在回答前，先去你的知识库里“查资料”，再根据查到的内容回答**。类似开卷考试，而不是让模型凭记忆瞎编。

它主要解决：LLM 知识过期、幻觉、无法访问私有数据、答案不可溯源等问题。RAG 不是微调；RAG 负责“外挂知识”，微调更多负责“改变风格/能力”。

---

## 一、RAG 的核心流程

```text
离线索引：
文档 → 解析 → 清洗 → 切块 → Embedding → 存入向量库/索引

在线问答：
用户问题 → 查询改写 → 检索 → 重排 → 拼接 Prompt → LLM 生成 → 答案 + 引用
```

关键组件：

- **文档加载**：PDF、Word、网页、数据库、Markdown、代码等
- **切块 Chunking**：把长文档切成小段
- **Embedding**：把文本转成向量
- **向量数据库**：FAISS、Chroma、Milvus、Qdrant、pgvector 等
- **检索器**：向量检索、关键词检索、混合检索
- **重排 Reranker**：对召回结果精排
- **LLM**：根据上下文生成答案
- **Prompt**：约束模型“只根据上下文回答”

---

## 二、如何开发一个 RAG？

### 1. 明确需求

先确定：

- 面向什么场景：客服、法律、医疗、代码、企业知识库？
- 数据来源是什么：PDF、网页、数据库、API？
- 是否需要权限控制？
- 是否要求引用来源？
- 数据更新频率？
- 用云端模型还是本地模型？

### 2. 准备数据

把数据解析成纯文本，并保留元数据：

```json
{
  "text": "内容片段",
  "source": "员工手册.pdf",
  "page": 12,
  "title": "报销制度",
  "tenant_id": "公司A",
  "permission": "hr"
}
```

元数据很重要，后面可以做过滤、引用、权限控制。

### 3. 切块

常见策略：

- 固定长度：如 500~1000 tokens
- 递归切分：按段落、标题、句子
- 语义切分：按语义边界
- 父子块：检索小块，返回大块
- 表格、代码单独处理

建议：`chunk_size=500~1000`，`overlap=50~150`，具体要实验。

### 4. 向量化并存入向量库

选择 Embedding 模型：

- 云端：OpenAI `text-embedding-3-small/large`
- 开源：BGE、GTE、m3e、Jina Embeddings
- 多语言场景要选多语言模型

然后存入：

- 本地小项目：FAISS、Chroma
- 生产：Milvus、Qdrant、Weaviate、pgvector

### 5. 实现检索

最基础的是向量相似度检索：

```text
问题向量 → 在向量库找最相似的 top k 个 chunk
```

但生产级通常用**混合检索**：

- 向量检索：擅长语义
- BM25/关键词检索：擅长专有名词、编号、错误码
- 混合后融合排序

再加重排：

```text
先召回 top 20~100 → Reranker 精排 → 取 top 3~8 给 LLM
```

### 6. 拼接 Prompt 并生成

Prompt 示例：

```text
你是一个严谨的助手。请只根据下面提供的上下文回答问题。
如果上下文没有答案，就说“我不知道”，不要编造。

上下文：
{context}

问题：
{question}

请给出答案，并标注来源。
```

### 7. 评估与优化

构建测试集，至少 50~100 条问答，评估：

- 检索召回率：该找的文档找到了吗？
- 答案忠实度：是否基于上下文？
- 答案相关性：是否回答了问题？
- 引用准确率
- 延迟和成本

工具：RAGAS、TruLens、人工评估。

---

## 三、最小可用 RAG 代码示例

用 Python + LangChain + FAISS + OpenAI：

```bash
pip install langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu
```

```python
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["OPENAI_API_KEY"] = "sk-..."

# 1. 加载文档
docs = TextLoader("knowledge.txt", encoding="utf-8").load()

# 2. 切块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)

# 3. 向量化 + 存库
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. 检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 5. Prompt
prompt = ChatPromptTemplate.from_template("""
你是一个严谨的助手。只根据下面上下文回答；找不到答案就说“我不知道”。
上下文：
{context}
问题：
{question}
""")

def format_docs(docs):
    return "\n\n".join(
        f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)
    )

# 6. 组装 RAG 链
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. 提问
print(rag_chain.invoke("公司的报销流程是什么？"))
```

这就是一个最小 RAG。它已经能跑通“文档 → 检索 → 生成”。

---

## 四、生产级 RAG 还要加什么？

最小 RAG 很容易做，难的是效果稳定。生产级通常需要：

1. **混合检索**：向量 + BM25
2. **重排 Reranker**：BGE-reranker、Cohere Rerank
3. **查询改写**：多查询、HyDE、结合历史改写
4. **多路召回**：不同索引、不同粒度
5. **父子块 / 句子窗口**：检索小块，给大上下文
6. **元数据过滤**：时间、产品、租户、权限
7. **上下文压缩**：去掉无关内容，节省 token
8. **引用来源**：输出 `[1][2]` 并链接原文
9. **权限控制**：检索前过滤，防止越权泄露
10. **增量更新**：文档变更后重新索引、删除旧向量
11. **缓存**：Embedding 缓存、检索缓存、答案缓存
12. **监控**：延迟、成本、召回率、用户反馈
13. **评估体系**：RAGAS、人工标注、A/B 测试

---

## 五、技术选型参考

| 模块        | 常见选择                                |
| --------- | ----------------------------------- |
| 框架        | LangChain、LlamaIndex、Haystack       |
| 向量库       | FAISS、Chroma、Milvus、Qdrant、pgvector |
| Embedding | OpenAI、BGE、GTE、m3e、Jina             |
| LLM       | GPT、Claude、Qwen、DeepSeek、Llama      |
| 重排        | BGE-reranker、Cohere Rerank          |
| 评估        | RAGAS、TruLens                       |
| 部署        | FastAPI、Flask、Docker、K8s            |

---

## 六、常见坑

- 切块太大：噪声多，检索不准
- 切块太小：语义不完整
- 只用向量检索：专有名词、编号容易漏
- 不做重排：top k 里噪声太多
- Prompt 不约束：模型开始编造
- 不更新索引：答案过期
- 不做权限过滤：数据泄露
- 没有评估：不知道改好还是改坏

---

## 总结

RAG 的本质是：**搜索系统 + Prompt 工程 + LLM 生成**。  
开发顺序建议：

1. 先做最小 MVP：加载文档 → 切块 → 向量化 → 检索 → 生成
2. 构建测试集，评估检索和答案质量
3. 再加混合检索、重排、查询改写、权限、引用
4. 最后做部署、监控、增量更新

如果你告诉我你的具体场景，比如“用本地模型做企业 PDF 知识库”或“用 OpenAI 做客服机器人”，我可以给你一套具体的项目结构和技术方案。





# 开发流程

---

## 🖥️ 阶段一：环境搭建与数据准备（2-3天）

### 1. 安装 Ollama for Windows

Ollama 官方提供了原生 Windows 应用，安装非常简单：

1. **下载**：访问 Ollama 官网 [https://ollama.com/download](https://ollama.com/download)，点击下载 Windows 版安装程序 `OllamaSetup.exe`。
2. **安装**：双击运行 `OllamaSetup.exe`，一路“下一步”即可。它默认安装在你的用户目录下，**不需要管理员权限**。
3. **验证**：安装完成后，Ollama 会自动在后台运行，右下角系统托盘会出现一个羊驼图标。打开 **PowerShell** 或 **CMD**，输入以下命令验证：
   
   ```powershell
   ollama -v
   ```
   
   如果显示版本号，说明安装成功。

> **⚠️ 重要提示：更改模型存储位置**
> 模型文件可能高达数十 GB。如果你的 C 盘空间紧张，建议在安装前或安装后设置环境变量 `OLLAMA_MODELS` 来更改模型存储路径。
> 
> 1. 打开“系统属性” -> “环境变量”。
> 2. 在“用户变量”中，新建一个变量，名为 `OLLAMA_MODELS`，值设置为你的目标路径，例如 `D:\OllamaModels`。
> 3. 保存后，重启 Ollama（在系统托盘右键退出，再从开始菜单重新打开）。

### 2. 拉取模型

在 PowerShell 中执行以下命令，拉取 Qwen3 4B 和嵌入模型 BGE-M3：

```powershell
ollama pull qwen3:4b-instruct-2507
ollama pull bge-m3
```

### 3. 创建 Python 虚拟环境并安装依赖

在 PowerShell 中，进入你的项目目录，然后执行：

```powershell
# 创建虚拟环境
python -m venv rag_env

# 激活虚拟环境（Windows 下激活命令不同）
.\rag_env\Scripts\Activate.ps1

# 安装依赖
pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama llama-index-vector-stores-chroma chromadb sentence-transformers ragas fastapi uvicorn
```

### 4. 准备数据与解析

将你的 Word、Excel、代码文件放入 `./data` 目录。使用 `SimpleDirectoryReader` 统一加载：

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
```

**切片策略**（与平台无关，但很重要）：

- **Word**：按标题层级切分，每个标题下的段落作为一个 chunk。
- **Excel**：按 Sheet 切分，并确保每个 chunk 都包含表头信息。
- **代码**：按函数/类切分，保留文件路径和语言标记。

---

## 🔗 阶段二：核心 RAG 链路开发（3-5天）

### 5. 配置本地模型与嵌入

```python
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = Ollama(model="qwen3:4b-instruct-2507", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="bge-m3")
```

### 6. 构建向量索引并持久化（ChromaDB）

在 Windows 上，ChromaDB 以 Python 库的形式在本地运行，数据会持久化到指定目录。

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("tech_support")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
index.storage_context.persist(persist_dir="./storage")
```

### 7. 创建检索器并实现问答

```python
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
query_engine = RetrieverQueryEngine(retriever=retriever)

response = query_engine.query("设备报错 E-204 怎么处理？")
print(response)
```

此时你已经跑通了最小可用的 RAG 系统。

---

## 🎯 阶段三：效果调优（5-7天）

### 8. 加入重排（Reranker）提升精度

BGE-reranker-v2-m3 在 Windows 上有多种部署方式，最简单的是直接通过 Python 库调用（`FlagEmbeddingReranker` 会自动下载模型）。如果希望更高效，也可以考虑使用 ONNX 版本或通过 `llama.cpp` 运行 GGUF 量化版。

```python
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-v2-m3", top_n=5)
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[reranker]
)
```

### 9. 优化 Prompt 模板，强制引用来源

```python
from llama_index.core import PromptTemplate

qa_prompt = PromptTemplate(
    "你是一个技术支持助手。请严格根据以下上下文回答问题，并在末尾注明来源。\n"
    "如果上下文中没有答案，请明确说“根据现有资料无法回答”。\n"
    "上下文：{context_str}\n"
    "问题：{query_str}\n"
    "答案："
)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})
```

### 10. 构建测试集并用 RAGAS 评估

准备 50-100 个问答对，评估**忠实度（Faithfulness）** 和**答案相关性（Answer Relevancy）**。根据评估结果调整切片大小、检索数量、重排参数。

---

## 🚀 阶段四：工程化与部署（3-5天）

### 11. 用 FastAPI 封装 API

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    user_role: str  # "engineering" or "rd"

@app.post("/ask")
async def ask(req: QueryRequest):
    response = query_engine.query(req.question)
    return {"answer": str(response), "sources": [n.metadata for n in response.source_nodes]}
```

### 12. 实现权限控制

在数据入库时，为每个 chunk 的元数据打上 `permission` 标签。检索时通过元数据过滤实现权限隔离。

### 13. Docker Desktop 一键部署

在 Windows 上，你需要先安装 **Docker Desktop**。安装后，Docker 命令与 Linux 完全一致。

编写 `docker-compose.yml`，将 Ollama、Chroma 服务（可选择以服务模式运行）、FastAPI 服务编排在一起，实现一键启动。

---

## 💎 总结与 Windows 要点

| 项目               | Windows 上的关键差异                                     |
|:---------------- |:-------------------------------------------------- |
| **Ollama**       | 官方原生支持，安装简单，注意设置 `OLLAMA_MODELS` 环境变量              |
| **Python 虚拟环境**  | 激活命令是 `.\rag_env\Scripts\Activate.ps1`，不是 `source` |
| **ChromaDB**     | 以 Python 库形式运行，数据持久化到本地目录，无需额外服务                   |
| **BGE-reranker** | 可通过 Python 库直接调用，也可使用 ONNX/GGUF 优化版                |
| **Docker**       | 需要安装 Docker Desktop，之后命令与 Linux 一致                 |

你的技术选型在 Windows 上完全可行。先从最小 MVP 开始跑通，遇到具体报错随时可以再讨论。
