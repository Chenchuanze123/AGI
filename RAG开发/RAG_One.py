from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter ,CodeSplitter
import random
from llama_index.readers.file import DocxReader
from llama_index.core import Settings, VectorStoreIndex, StorageContext, PromptTemplate, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import os

all_nodes = []

# ---------- 1. C/C++ 代码 ----------
code_documents = SimpleDirectoryReader(
    input_dir="./test_data",
    required_exts=[".c", ".cpp", ".h"],
    recursive=True,
    encoding="gbk"
).load_data()
print(f"共加载 {len(code_documents)} 个代码文件")

cpp_splitter = CodeSplitter(
    language="cpp",
    chunk_lines=40,
    chunk_lines_overlap=15,
    max_chars=1500
)
code_nodes = cpp_splitter.get_nodes_from_documents(code_documents)
all_nodes.extend(code_nodes)
print(f"已经加载 {len(code_nodes)} 个代码切片")

# ---------- 2. Word 文档 ----------
docx_reader = DocxReader()

word_documents = SimpleDirectoryReader(
    input_dir="./test_data",
    required_exts=[".docx"],
    recursive=True,
    file_extractor={".docx": docx_reader}   # 显式指定解析器
).load_data()
print(f"共加载 {len(word_documents)} 个 Word 文件")

word_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
word_nodes = word_splitter.get_nodes_from_documents(word_documents)
all_nodes.extend(word_nodes)
print(f"已经加载 {len(word_nodes)} 个句子切片")

# ---------- 3. 纯文本文件 ----------
# 先尝试 UTF-8，失败再回退到 GBK
try:
    txt_documents = SimpleDirectoryReader(
        input_dir="./test_data",
        required_exts=[".txt"],
        recursive=True,
        encoding="utf-8"
    ).load_data()
except UnicodeDecodeError:
    print("UTF-8 解码失败，改用 GBK 重试...")
    txt_documents = SimpleDirectoryReader(
        input_dir="./test_data",
        required_exts=[".txt"],
        recursive=True,
        encoding="gbk"
    ).load_data()
print(f"共加载 {len(txt_documents)} 个 txt 文件")

# txt 和 Word 一样，用 SentenceSplitter 就行
txt_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
txt_nodes = txt_splitter.get_nodes_from_documents(txt_documents)
all_nodes.extend(txt_nodes)

# ---------- 汇总 ----------
print(f"共生成 {len(all_nodes)} 个知识切片")




# 随机抽查 3 个切片
for i in random.sample(range(len(all_nodes)), 3):
    node = all_nodes[i]
    print(f"===== 切片 {i} =====")
    print(f"来源: {node.metadata.get('file_name', '未知')}")
    print(f"内容: {node.text[:500]}...")  # 只打印前500字符
    print()

# 将切片内容写入文件
with open("slices_preview.txt", "w", encoding="utf-8") as f:
    for i, node in enumerate(all_nodes):
        f.write(f"===== 切片 {i} =====\n")
        f.write(f"来源: {node.metadata.get('file_name', '未知')}\n")
        f.write(f"内容:\n{node.text}\n\n")


# ---------- 1. 配套本地模型与嵌入 ----------
# 时间调大
Settings.llm = Ollama(
    model="qwen3:1.7b",
    request_timeout=900.0,
    additional_kwargs={"think": False}  # 在这里设置
)
Settings.embed_model = OllamaEmbedding(model_name="bge-m3")

# ---------- 2. 构建向量索引并持久化（ChromaDB） ----------
# 初始化 Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("tech_support_v1")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 判断是否已有索引
if os.path.exists("./storage") and chroma_collection.count() > 0:
    print("检测到已有向量库，直接加载索引，跳过向量化...")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir="./storage"
    )
    index = load_index_from_storage(storage_context)
else:
    print("未找到现有索引，开始向量化...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=all_nodes,
        storage_context=storage_context,
        show_progress=True
    )
    index.storage_context.persist(persist_dir="./storage")
    print("向量化完成！")

# ---------- 3. 定义防幻觉 Prompt 模板 ----------
qa_prompt = PromptTemplate(
    "你是一个技术支持助手。请严格根据以下上下文回答问题。\n"
    #"你必须直接给出最终答案，禁止输出任何思考过程、解释或内心独白。\n"
    "如果上下文中没有答案，请明确说“根据现有资料无法回答”，不要编造。\n"
    "请在回答末尾标注来源文件名。\n"
    "---------------------\n"
    "上下文：\n{context_str}\n"
    "---------------------\n"
    "问题：{query_str}\n"
    "答案："
)

# ---------- 4. 创建检索器并实现问答 ----------
retriever = VectorIndexRetriever(index=index, similarity_top_k=3) # 先降到3
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    text_qa_template=qa_prompt,
    response_mode="compact"
)

# ---------- 5. 测试提问 ----------
print("\n=== RAG 系统已就绪，输入 exit 退出 ===")
while True:
    question = input("\n请输入你的问题： ").strip()
    if question.lower() in ("exit", "quit", "q"):
        print("再见！")
        break
    if not question:
        continue

    try:
        response = query_engine.query(question)
        print("\n=== 回答 ===")
        print(response)
        print("\n=== 引用来源 ===")
        for node in response.source_nodes:
            print(f"- {node.metadata.get('file_name', '未知')} (相关度: {node.score:.4f})")
    except Exception as e:
        print(f"\n[出错] {e}")