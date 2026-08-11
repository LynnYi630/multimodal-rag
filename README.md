# FastAPI 多模态 RAG 检索服务

该服务提供面向 PDF、DOCX、PPTX 等文档的多模态入库与检索能力。文本与图片分别召回，经 RRF 融合和统一 Reranker 重排后返回；服务本身不生成最终问答内容。

## 核心能力

- FastAPI 文档上传、版本管理、任务查询、资源访问和检索接口
- PostgreSQL 保存文档、版本、节点、ACL 和任务状态
- MinIO 保存原始文件、抽取图片和派生结果
- Celery + Redis 异步执行文档入库
- Docling 解析 PDF、DOCX、PPTX
- RapidOCR 处理扫描页基础 OCR
- PaddleOCR-VL 和视觉语言模型进行图片增强理解
- Qwen3-VL Embedding 生成统一文本/图片向量
- Qdrant 执行文本节点和图片节点向量召回
- RRF 融合与 Qwen3-VL Reranker 统一重排
- Tenant、User、Role、Group 维度的检索权限过滤
- Mock Provider 支持无外部模型依赖的自动化测试

## 数据流

```text
上传文档
  -> MinIO 保存原文件
  -> PostgreSQL 创建文档版本和入库任务
  -> Celery Worker
  -> Docling + RapidOCR 解析版面、正文、表格和图片
  -> PaddleOCR / VLM 增强图片信息
  -> Embedding 生成节点向量
  -> Qdrant 写入向量和 ACL Payload

搜索请求
  -> Query Embedding
  -> 文本节点召回 + 图片节点召回
  -> RRF 融合
  -> ACL 与版本过滤
  -> Reranker 重排
  -> 返回文本片段或图片元数据
```

## 环境要求

- Python 3.11–3.13
- PostgreSQL 17 或兼容版本
- Redis 7 或兼容版本
- Qdrant
- MinIO
- Docling 模型文件
- 根据所选 Provider 准备 DashScope API Key、PaddleOCR Access Token 或本地模型服务

## 安装

创建虚拟环境后安装开发和文档解析依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,docling]"
```

PowerShell 激活命令：

```powershell
.\.venv\Scripts\Activate.ps1
```

可选依赖组：

- `dev`：pytest、pytest-asyncio、ruff
- `docling`：Docling、RapidOCR 和 Torch 文档解析组件
- `qwen-local`：进程内加载 Qwen Embedding/Reranker 时需要

## 配置

复制配置模板并填写真实连接信息和凭证：

```bash
cp .env.example .env
```

`.env` 包含密码和 API Key，已被 Git 忽略，不应提交到版本库。完整字段说明见 `.env.example`。

生产或共享环境至少需要确认以下配置：

```env
DATABASE_URL=postgresql+asyncpg://<user>:<password>@<postgres-host>:5432/<database>
REDIS_URL=redis://:<password>@<redis-host>:6379/0

VECTOR_PROVIDER=qdrant
QDRANT_URL=http://<qdrant-host>:6333
QDRANT_COLLECTION=knowledge_nodes_dashscope_2048

STORAGE_PROVIDER=minio
MINIO_ENDPOINT=<minio-host>:9000
MINIO_ACCESS_KEY=<application-access-key>
MINIO_SECRET_KEY=<application-secret-key>
MINIO_SECURE=false

CELERY_TASK_ALWAYS_EAGER=false
```

MinIO 使用三个私有 Bucket：

- `rag-originals`：上传的原始文件
- `rag-assets`：解析出的图片和页面渲染图
- `rag-derived`：Docling、OCR、VLM 等派生结果

应用账号应只获得上述 Bucket 所需的最小权限，不应使用 MinIO 管理员凭证。

## 数据库与 Qdrant 初始化

首次部署或升级版本时执行数据库迁移：

```bash
python -m alembic upgrade head
```

当 `AUTO_CREATE_SCHEMA=false` 时，应用不会代替 Alembic 创建业务表。

Qdrant Collection 由应用启动时检查并创建。Collection 的向量维度必须与 Embedding Provider 一致：

```env
QDRANT_COLLECTION=knowledge_nodes_dashscope_2048
EMBEDDING_DIMENSION=2048
```

不同模型或不同语义空间产生的向量不应写入同一个 Collection。更换 Embedding 模型、维度或语义空间时，应创建新的 Collection 并重新入库。

## Docling 与 RapidOCR

扫描版 PDF 需要布局、TableFormer V2 和 RapidOCR 模型：

```bash
python -m docling.cli.tools models download \
  layout tableformerv2 rapidocr \
  --output-dir "${DOCLING_MODEL_DIR}"
```

配置模型根目录和 OCR 引擎：

```env
PARSER_PROVIDER=docling
DOCLING_ARTIFACTS_PATH=/models/docling
DOCLING_OCR_ENGINE=rapidocr
DOCLING_RAPIDOCR_BACKEND=torch
DOCLING_OCR_LANGUAGES=chinese
```

模型根目录至少应包含：

- `docling-project--docling-layout-heron`
- `docling-project--TableFormerV2`
- `RapidOcr`

`torch` 后端可使用兼容的 CUDA PyTorch；CPU 部署可选择 `onnxruntime`。应用显式选择 RapidOCR，避免 Docling 自动探测因运行环境不同而切换 OCR 引擎。

Docling 内部 RapidOCR 与 `OCR_PROVIDER=paddleocr` 作用不同：RapidOCR 负责文档基础解析，PaddleOCR-VL 负责对抽取图片或扫描页进行增强识别，两者可以同时启用。

## Provider 配置

### DashScope 云端模式

```env
EMBEDDING_PROVIDER=dashscope
EMBEDDING_DIMENSION=2048
DASHSCOPE_EMBEDDING_URL=https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding
DASHSCOPE_EMBEDDING_MODEL=qwen3-vl-embedding
DASHSCOPE_EMBEDDING_CONCURRENCY=4

RERANKER_PROVIDER=dashscope
DASHSCOPE_RERANKER_URL=https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
DASHSCOPE_RERANKER_MODEL=qwen3-vl-rerank

VLM_PROVIDER=dashscope
DASHSCOPE_VLM_MODEL=qwen3.7-flash
DASHSCOPE_API_KEY=<dashscope-api-key>

OCR_PROVIDER=paddleocr
PADDLEOCR_MODEL=PaddleOCR-VL-1.6
PADDLEOCR_ACCESS_TOKEN=<paddleocr-access-token>
```

Embedding Provider 对每个节点启用多模态融合并输出与 Qdrant 一致的向量维度。图片节点在 Reranker 阶段优先发送原始图片；OCR、图片描述和关联正文参与前一阶段的融合 Embedding。

### 进程内 Qwen 模型

```env
EMBEDDING_PROVIDER=qwen_local
EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
EMBEDDING_REVISION=<model-revision>
QWEN_EMBEDDING_MODEL_PATH=/models/Qwen3-VL-Embedding-2B
QWEN_EMBEDDING_REPOSITORY_PATH=/models/Qwen3-VL-Embedding-2B/scripts

RERANKER_PROVIDER=qwen_local
RERANKER_MODEL=Qwen/Qwen3-VL-Reranker-2B
RERANKER_REVISION=<model-revision>
QWEN_RERANKER_MODEL_PATH=/models/Qwen3-VL-Reranker-2B
QWEN_RERANKER_REPOSITORY_PATH=/models/Qwen3-VL-Reranker-2B/scripts
```

进程内模型采用延迟加载，并在首次调用后常驻进程。多个 API 或 Worker 进程可能分别加载一份模型，需要按内存和显存容量规划进程数。

### vLLM Embedding 与 Reranker

Qwen3-VL Embedding 和 Reranker 分别启动独立 pooling 服务。示例中的路径、端口和并发需要按部署环境调整。

Embedding：

```bash
vllm serve /models/Qwen3-VL-Embedding-2B \
  --runner pooling \
  --host 0.0.0.0 \
  --port 8200 \
  --served-model-name qwen3-vl-embedding-2b \
  --api-key <embedding-api-key> \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image":1}'
```

Reranker：

```bash
vllm serve /models/Qwen3-VL-Reranker-2B \
  --runner pooling \
  --host 0.0.0.0 \
  --port 8300 \
  --served-model-name qwen3-vl-reranker-2b \
  --api-key <reranker-api-key> \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image":1}' \
  --hf-overrides '{"architectures":["Qwen3VLForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --chat-template /models/templates/qwen3_vl_reranker.jinja
```

应用配置：

```env
EMBEDDING_PROVIDER=vllm
EMBEDDING_VLLM_BASE_URL=http://<embedding-host>:8200/v1
EMBEDDING_VLLM_API_KEY=<embedding-api-key>
EMBEDDING_VLLM_MODEL=qwen3-vl-embedding-2b

RERANKER_PROVIDER=vllm
RERANKER_VLLM_BASE_URL=http://<reranker-host>:8300/v1
RERANKER_VLLM_API_KEY=<reranker-api-key>
RERANKER_VLLM_MODEL=qwen3-vl-reranker-2b
```

### OpenAI-compatible VLM

生成式 VLM 通过 Chat Completions 接口调用：

```bash
vllm serve /models/Qwen3-VL-8B-Instruct-FP8 \
  --host 0.0.0.0 \
  --port 8100 \
  --served-model-name qwen3-vl-8b-instruct-fp8 \
  --api-key <vlm-api-key> \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image":1}'
```

```env
VLM_PROVIDER=vllm
VLM_BASE_URL=http://<vlm-host>:8100/v1
VLM_API_KEY=<vlm-api-key>
VLM_MODEL=qwen3-vl-8b-instruct-fp8
VLM_MAX_TOKENS=800
VLM_TIMEOUT_SECONDS=120
```

修改解析器、OCR、VLM 或 Embedding 配置后，需要重启 FastAPI 和 Celery Worker。已经入库的文档不会自动重新处理，可使用 `force=true` 创建新版本。

## 启动服务

启动 API：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

开发环境需要热重载时增加 `--reload`。

健康检查：

- `GET /health/live`
- `GET /health/ready`
- OpenAPI：`GET /docs`

### Celery Worker

异步模式下需要独立启动 Worker：

```bash
celery -A app.workers.celery_app:celery_app worker --loglevel=INFO
```

省略 `--concurrency` 时，Celery 会根据运行环境和 Worker Pool 自动确定并发数；Linux 默认 prefork 通常以可见 CPU 核心数为基础。也可以显式设置：

```bash
celery -A app.workers.celery_app:celery_app worker \
  --loglevel=INFO \
  --concurrency=<worker-process-count>
```

Celery 并发数不应只按 CPU 核数决定。每个进程都可能独立加载 Docling、布局模型、TableFormer、RapidOCR 或进程内 Qwen 模型。建议根据以下上限制定并发：

```text
安全并发数 = min(
  可用 CPU 并发能力,
  可用内存 / 单任务峰值内存,
  可用显存 / 单进程峰值显存
)
```

对于单 GPU 或大模型进程内推理，应先通过压测确认模型是否会在每个 Worker 进程中重复加载；对于云端 Provider 和 CPU Docling，可在内存允许的情况下提高并发。容器部署还应确认 Celery 看到的 CPU 数量与容器实际 CPU 配额一致。

Windows 开发环境不支持 prefork 时可使用：

```powershell
python -m celery -A app.workers.celery_app:celery_app worker --loglevel=INFO --pool=solo
```

Worker 必须读取与 API 一致的 `.env`，并能够访问 PostgreSQL、Redis、Qdrant、MinIO 和已启用的外部 Provider。

## API 示例

请求通过以下 Header 传递身份和权限上下文：

```text
X-Tenant-ID: <tenant-uuid>
X-User-ID: <user-uuid>
X-Roles: <comma-separated-roles>
X-Groups: <comma-separated-groups>
```

上传文档：

```bash
curl -X POST http://<api-host>:8000/v1/documents \
  -H "X-Tenant-ID: <tenant-uuid>" \
  -H "X-User-ID: <user-uuid>" \
  -H "X-Groups: <group-name>" \
  -F "file=@sample.pdf;type=application/pdf" \
  -F "parser=docling" \
  -F "acl_subjects=group:<group-name>"
```

接口返回 HTTP 202，以及 `document_id`、`version_id` 和 `job_id`。

查询任务：

```bash
curl http://<api-host>:8000/v1/jobs/<job-id> \
  -H "X-Tenant-ID: <tenant-uuid>" \
  -H "X-User-ID: <user-uuid>"
```

执行混合检索：

```bash
curl -X POST http://<api-host>:8000/v1/search \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: <tenant-uuid>" \
  -H "X-User-ID: <user-uuid>" \
  -H "X-Groups: <group-name>" \
  -d '{
    "query": "查找系统架构图和相关说明",
    "top_k": 10,
    "text_candidate_k": 30,
    "image_candidate_k": 20,
    "include": {
      "snippets": true,
      "image_metadata": true
    }
  }'
```

## 测试与代码检查

```bash
pytest
ruff check app tests
```

自动化测试使用临时 SQLite、本地对象存储、内存向量库和 Mock Provider，不访问真实基础设施或外部 API。

Mock Provider 是用于测试的确定性替代实现：Embedding 生成哈希向量，Reranker 使用词项匹配。它用于验证接口、权限、版本和编排逻辑，不代表真实语义检索质量。
