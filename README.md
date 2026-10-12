# AGI

> 个人 AI 学习与实践仓库 —— 围绕「电力行业 + AI」沉淀的 RAG 原型、开发流程、踩坑记录与战略思考。

本仓库收录了两类内容：

1. **RAG 现场技术支持知识库原型**：一个基于 LlamaIndex + Chroma + 本地大模型的检索增强生成（RAG）最小可用系统，目标是把历史故障、规约文档、代码沉淀成可被现场技术支持人员快速检索、并带引用来源的知识库。
2. **AI + 电力网关 / AI 工程化转型 研究与思考**：关于 RTU 网关如何与 AI 融合的战略分析，以及个人 AI 工程化能力构建的行动指南。

---

## 📁 目录结构

```text
.
├── RAG开发/                          # ★ RAG 主项目（建议以此为准）
│   ├── RAG_One.py                    # 核心 RAG 流水线：加载 → 切片 → 向量化 → 检索 → 问答
│   ├── test.py                       # 简单的 .docx 文本提取测试脚本
│   ├── 需求文档.md                    # 需求定义、场景、技术选型
│   ├── RAG开发流程.md                 # 从环境搭建到部署的完整流程指南（含 Windows 要点）
│   ├── 问题记录.md                    # 开发踩坑与解决方案汇总
│   ├── slices_preview.txt             # 切片预览（运行 RAG_One.py 后生成）
│   ├── 364号主站_Modbus_2022规约使用说明.docx  # 样例知识文档（Word）
│   └── 工作台任务统计-2026-09-30 15_41_41.xlsx  # 样例统计数据（Excel）
│
├── 111/                              # RAG 项目的并行 / 较早副本，内容与 RAG开发 基本一致
│                                     # （维护时以 RAG开发/ 为权威源，避免两个副本漂移）
│
├── AI+网关/
│   └── 网关与AI的融合前景.md          # RTU 网关与 AI 融合的战略分析（政策 / 架构 / 场景 / 红线）
│
├── deepseek_markdown_20260930_b6f834.md  # 《AI 工程化转型与能力构建 1-2 年行动指南》
│
├── .gitignore
└── README.md
```

> ⚠️ **关于目录重复**：`111/` 与 `RAG开发/` 是同一 RAG 项目的两个副本。后续迭代请统一在 `RAG开发/` 进行，`111/` 视为历史快照，不必同步维护。

---

## 🧱 技术栈

| 模块         | 选型                                            | 说明                                  |
| ---------- | --------------------------------------------- | ----------------------------------- |
| 编排框架      | **LlamaIndex**                                 | 文档加载、切片、索引、检索、查询引擎一体化            |
| 向量数据库     | **Chroma**（本地持久化）                           | 以 Python 库形式运行，数据落盘到 `./chroma_db`   |
| 嵌入模型      | **BGE-M3**（经 Ollama 提供）                      | 多语言 Embedding，负责把文本转成向量              |
| 生成模型      | **Qwen3**（本地，Ollama）                         | 开发中用 `qwen3:1.7b`；选型目标为 `qwen3:4b`  |
| 重排 Reranker | **BGE-reranker-v2-m3**                          | 规划引入，用于对召回结果精排（尚未接入主流程）            |
| 评估         | **RAGAS**                                      | 规划用 50–100 条问答对评估忠实度 / 答案相关性        |
| 部署         | **FastAPI + Docker**                           | 规划中：封装 `/ask` API、做权限控制、容器化一键部署      |

> 完整的选型推导与阶段规划见 [`RAG开发/需求文档.md`](RAG开发/需求文档.md) 与 [`RAG开发/RAG开发流程.md`](RAG开发/RAG开发流程.md)。

---

## 🔧 RAG 流水线（`RAG开发/RAG_One.py`）

`RAG_One.py` 跑通了「文档 → 检索 → 生成」的最小闭环，核心步骤如下：

### 1. 数据加载与切片（`test_data/` 目录）
按文件类型分别处理，**切片策略因类型而异**：

| 数据类型 | 加载方式 | 切片器 | 关键参数 |
| ---- | ---- | ---- | ---- |
| C/C++ 代码 | `SimpleDirectoryReader(required_exts=[".c",".cpp",".h"], encoding="gbk")` | `CodeSplitter` | `language="cpp"`, `chunk_lines=40`, `overlap=15`, `max_chars=1500`（按 AST 结构切） |
| Word 文档 | `DocxReader` **显式**解析（避免乱码） | `SentenceSplitter` | `chunk_size=512`, `chunk_overlap=64` |
| 纯文本 | `SimpleDirectoryReader`，UTF-8 失败自动回退 GBK | `SentenceSplitter` | `chunk_size=512`, `chunk_overlap=64` |

所有节点汇总到 `all_nodes`，并随机抽取 3 个切片打印预览，同时把全部切片写入 `slices_preview.txt`。

### 2. 本地模型接入
```python
Settings.llm = Ollama(model="qwen3:1.7b", request_timeout=900.0,
                      additional_kwargs={"think": False})
Settings.embed_model = OllamaEmbedding(model_name="bge-m3")
```

### 3. 向量化与持久化
- 使用 `chromadb.PersistentClient(path="./chroma_db")`，集合名 `tech_support_v1`。
- 索引持久化到 `./storage`；**若已存在索引且向量库非空，则直接加载、跳过向量化**（避免重复耗时）。

### 4. 防幻觉 Prompt
约束模型「只依据上下文回答；无答案时明确说『根据现有资料无法回答』；并在回答末尾标注来源文件名」。

### 5. 检索与问答
```python
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever, text_qa_template=qa_prompt, response_mode="compact"
)
```
运行后进入交互式循环：输入问题 → 输出回答 + 引用来源（含相关度分数）；输入 `exit`/`quit`/`q` 退出。

### ✅ 已实现 vs 📌 规划中
| 能力 | 状态 |
| -- | -- |
| 多类型文档加载（代码 / Word / 文本） | ✅ 已实现 |
| Chroma 本地向量库 + 增量加载 | ✅ 已实现 |
| 防幻觉 / 引用来源 | ✅ 已实现 |
| 重排 Reranker（BGE-reranker-v2-m3） | 📌 规划中（流程文档已给出代码示例） |
| 权限控制（按 `permission` 元数据过滤） | 📌 规划中 |
| FastAPI 接口 + Docker 部署 | 📌 规划中 |

---

## 🚀 环境搭建（Windows）

详细步骤见 [`RAG开发/RAG开发流程.md`](RAG开发/RAG开发流程.md)，要点如下：

1. **安装 Ollama（Windows 原生版）**，并设置环境变量 `OLLAMA_MODELS` 把模型存到非 C 盘（模型可达数十 GB）。修改后需彻底退出并重启 Ollama。
2. **拉取模型**（注意用 Ollama 官方 tag）：
   ```powershell
   ollama pull qwen3:4b      # 或开发调试用 qwen3:1.7b
   ollama pull bge-m3
   ```
3. **创建并激活虚拟环境**：
   ```powershell
   python -m venv rag_env
   .\rag_env\Scripts\Activate.ps1
   ```
4. **安装依赖**：
   ```powershell
   pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama `
               llama-index-vector-stores-chroma chromadb sentence-transformers `
               llama-index-readers-file python-docx ragas fastapi uvicorn
   ```
5. **运行**：把知识文档放入 `test_data/`（或在 `RAG_One.py` 中改路径），然后 `python RAG_One.py`。

---

## 🧪 数据准备与切片策略

- **代码**：用 `CodeSplitter` 按语法结构切（不是按字符数），保留函数/类边界，并带文件路径与语言标记。
- **Word**：按 `DocxReader` 解析后，`SentenceSplitter` 切分；**务必显式指定 `file_extractor`**，否则 `.docx`（本质是 ZIP）会被当成二进制强行解码而乱码。
- **Excel / 表格**：规划中按 Sheet 切分，并确保每个 chunk 带表头。
- 经验值：`chunk_size=512`，`overlap=64`，需结合业务实测调优。

---

## 🐞 常见问题与踩坑

完整记录见 [`RAG开发/问题记录.md`](RAG开发/问题记录.md)，高频坑位速查：

| 现象 | 根因 | 解法 |
| -- | -- | -- |
| `ollama pull` 报 `file does not exist` / 网络超时 | 混用了 HF/ModelScope 命名；国内直连不稳 | 用官方 tag `qwen3:4b` / `bge-m3`；必要时走 ModelScope 镜像；清理残留 manifest |
| `CodeSplitter` 报 `No module named 'tree_sitter_language_pack'` | 缺依赖 | `pip install tree-sitter-language-pack` |
| `Could not get parser for language c++` | C++ 标识符是 `cpp` 不是 `c++` | `language="cpp"` |
| `DownloadError: Failed to download ... parsers-windows...tar.zst` | 国内访问 GitHub 动态下载解析器失败 | 改用 `tree_sitter` + `tree_sitter_cpp` 手动构造 parser 后传入 `CodeSplitter` |
| Word 切片严重乱码 | `.docx` 被回退成通用文本读取器解码 | 显式 `DocxReader` + `file_extractor={".docx": docx_reader}` |

---

## 🔭 AI + 电力网关 / AI 工程化（研究资料）

- [`AI+网关/网关与AI的融合前景.md`](AI+网关/网关与AI的融合前景.md)：一份面向 RTU 网关「AI 融合」的战略分析 —— 从政策推力（"人工智能+"能源系列文件）、网关四代形态跃迁（会说话 → 会看病 → 会学习 → 受控自治）、「确定性实时核 + AI 智能体」双核架构、七个高 ROI 场景，到不可逾越的安全红线（AI 不进控制闭环、数据不出场、全栈国产、可解释性一票否决）。
- [`deepseek_markdown_20260930_b6f834.md`](deepseek_markdown_20260930_b6f834.md)：《AI 工程化转型与能力构建 1-2 年行动指南》—— 围绕「AI 工程化能力 × 电力行业 Know-how」复合壁垒的阶段性学习计划。

---

## 📦 仓库约定（已排除内容）

为控制体积、避免泄露本地/专有数据，以下目录与文件已被 `.gitignore` 排除，**不入库**：

- `rag_env/`、`__pycache__/` —— Python 虚拟环境与缓存
- `chroma_db/`、`storage/` —— 向量库与索引（运行时生成）
- `test_data/` —— 本地知识文档（含待索引的原始材料）
- `.data/`、`.svn/` —— 本地其它工作副本与版本元数据
- `*.zip`、`*.exe`、`*.bin` —— 归档与二进制

> 即：仓库只保留**源码、脚本、需求/流程/问题文档、样例知识文档（docx/xlsx）与研究笔记**，方便复现与协作；模型权重、向量库、依赖环境均在本地准备。

---

## 📄 许可证与免责声明

- 本仓库为**个人学习与实践**用途，文档与代码按「现状」提供，不保证适用于任何生产环境。
- 涉及电力监控系统安全的规定（如发改委 27 号令）与行业红线，请务必以官方文件与所在单位合规要求为准；**任何 AI 能力不得接入控制闭环、不得下发遥控出口动作。**
- 仓库不含任何公司专有源码或涉密数据；如有误纳入，请联系维护者移除。
