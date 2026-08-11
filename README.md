# FastAPI 多模态 RAG 检索服务

这是 `FastAPI多模态RAG服务实现方案.md` 的第一版可运行实现。服务负责文档导入和多模态检索，不生成最终答案：文本与图片分别召回，经 RRF 融合，再由统一 Reranker 重排后返回。

## 当前本机配置

项目当前使用真实基础设施和本地 Qwen 模型，不再以 SQLite、内存向量库或 Mock 模型作为 `.env` 默认值。

| 组件 | 当前配置 |
| --- | --- |
| PostgreSQL | 容器 `postgres`，镜像 `postgres:17`，数据库 `appdb`，用户 `postgres`，端口 `5432` |
| Redis | 容器 `redis`，镜像 `redis:7.4`，端口 `6379`，密码 `123456` |
| Qdrant | 容器 `qdrant`，镜像 `qdrant/qdrant:v1.18.3`，HTTP 端口 `6333` |
| MinIO | 容器 `minio`，镜像 `minio/minio:RELEASE.2025-04-22T22-12-26Z`，API 端口 `9000`，控制台端口 `9001` |
| Embedding | `C:/Models/Qwen/Qwen3-VL-Embedding-2B`，向量维度 `2048` |
| Reranker | `C:/Models/Qwen/Qwen3-VL-Reranker-2B` |

MinIO 应用账号为 `rag-service`，密码为 `service123`，使用以下三个私有 Bucket：

- `rag-originals`：保存上传的原始文件
- `rag-assets`：保存文档解析出的图片等资源
- `rag-derived`：保存解析、OCR、描述等派生结果

`.env` 保存本机密码和 API Key，已加入 `.gitignore`，不应提交到版本库。`.env.example` 是带详细说明的配置模板。

## 环境要求

- Python 3.11–3.13
- Docker Desktop
- 已下载上述两个 Qwen 模型
- CPU 可以运行真实模型，但首次加载和推理较慢，并且需要较多内存；当前本机没有 NVIDIA GPU

创建虚拟环境并一次性安装核心依赖及全部可选依赖：

```powershell
cd C:\Development\multimodal-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,docling,qwen-local]"
```

三个可选依赖组分别是：

- `dev`：pytest、ruff 等开发和测试工具
- `docling`：文档解析组件
- `qwen-local`：PyTorch、Transformers、Qwen VL 工具和本地模型运行依赖

## 启动基础设施

现有容器全部启动：

```powershell
docker start postgres redis qdrant minio
docker ps
```

当前 `CELERY_TASK_ALWAYS_EAGER=true`，所以 Redis 在同步开发模式下不是必需的；这里仍启动 Redis，便于随后切换到真正的异步 Worker。

验证 PostgreSQL：

```powershell
docker exec postgres pg_isready -U postgres -d appdb
docker exec postgres psql -U postgres -d appdb -c "SELECT current_database(), current_user;"
```

验证 Redis：

```powershell
docker exec redis redis-cli -a 123456 ping
```

验证 Qdrant：

```powershell
curl.exe http://127.0.0.1:6333/collections
```

MinIO 控制台地址为 `http://127.0.0.1:9001`。管理员账号仅用于管理：

```text
用户名：admin
密码：admin123
```

应用自身使用 `.env` 中的 `rag-service/service123`，不要在应用里使用管理员账号。

## 初始化数据库和 Qdrant

PostgreSQL 容器通过以下初始化参数创建了数据库，因此不需要再次手动执行 `CREATE DATABASE`：

```text
POSTGRES_USER=postgres
POSTGRES_PASSWORD=123456
POSTGRES_DB=appdb
```

首次使用或迁移版本发生变化时执行：

```powershell
python -m alembic upgrade head
```

当前数据库迁移版本为 `0002`。该版本将 `documents.source_type` 扩展为
`VARCHAR(128)`，以容纳 DOCX、PPTX 等 Office 文件的标准 MIME 类型。`.env` 中设置了：

```env
DATABASE_URL=postgresql+asyncpg://postgres:123456@127.0.0.1:5432/appdb
AUTO_CREATE_SCHEMA=false
```

Qdrant Collection 不需要提前手动创建。应用启动时会检查并自动创建：

```env
VECTOR_PROVIDER=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=knowledge_nodes
EMBEDDING_DIMENSION=2048
```

`knowledge_nodes` 使用 2048 维向量和 Cosine 距离，并自动创建检索所需的 Payload 索引。Collection 创建后不能直接修改向量维度；更换为不同维度的 Embedding 模型时，应新建或重建 Collection。

## Docling PDF 模型

扫描版 PDF 需要提前准备 Docling 的布局、表格和 OCR 模型。Windows 推荐显式下载到不依赖符号链接的本地目录：

```powershell
& ".\.venv\Scripts\python.exe" -m docling.cli.tools models download `
  layout tableformer rapidocr `
  --output-dir "C:\Models\Docling"
```

然后在 `.env` 中把模型根目录交给应用：

```env
PARSER_PROVIDER=docling
DOCLING_ARTIFACTS_PATH=C:/Models/Docling
```

该目录至少应包含 `docling-project--docling-layout-heron`、`docling-project--docling-models` 和 `RapidOcr`。配置的是三者共同的父目录，不是某个模型子目录。显式路径可以避免首次解析 PDF 时访问 Hugging Face 全局缓存，也规避普通 Windows 账户无法创建缓存符号链接的问题。

## 本地 Qwen 模型

当前配置分别加载两个模型快照自带的 Python 脚本：

```env
EMBEDDING_PROVIDER=qwen_local
EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
EMBEDDING_REVISION=local
EMBEDDING_DIMENSION=2048
QWEN_EMBEDDING_MODULE=qwen3_vl_embedding
QWEN_EMBEDDING_CLASS=Qwen3VLEmbedder
QWEN_EMBEDDING_MODEL_PATH=C:/Models/Qwen/Qwen3-VL-Embedding-2B
QWEN_EMBEDDING_REPOSITORY_PATH=C:/Models/Qwen/Qwen3-VL-Embedding-2B/scripts

RERANKER_PROVIDER=qwen_local
RERANKER_MODEL=Qwen/Qwen3-VL-Reranker-2B
RERANKER_REVISION=local
QWEN_RERANKER_MODULE=qwen3_vl_reranker
QWEN_RERANKER_CLASS=Qwen3VLReranker
QWEN_RERANKER_MODEL_PATH=C:/Models/Qwen/Qwen3-VL-Reranker-2B
QWEN_RERANKER_REPOSITORY_PATH=C:/Models/Qwen/Qwen3-VL-Reranker-2B/scripts
```

本地 Embedding 和 Reranker 不需要 API Key。模型采用延迟加载并在首次调用后常驻进程；生产环境应使用 GPU，并根据显存规划 API/Worker 的进程数，避免每个进程重复加载一份模型。

### 使用 vLLM 部署 Embedding 和 Reranker

Qwen3-VL-Embedding/Reranker 的 vLLM 支持要求 `vllm>=0.14.0`。两个模型必须分别启动为独立 pooling 服务；不能与生成式 VLM 共用同一个 vLLM 进程。

Embedding 服务：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve \
  /home/wangyi/models/Qwen/Qwen3-VL-Embedding-2B \
  --runner pooling \
  --host 127.0.0.1 \
  --port 8200 \
  --served-model-name qwen3-vl-embedding-2b \
  --api-key local-embedding-key \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.22 \
  --limit-mm-per-prompt '{"image":1}'
```

Reranker 必须使用 vLLM 官方 Qwen3-VL 模板。先下载模板：

```bash
mkdir -p ~/models/Qwen/vllm-templates
curl -fL \
  https://raw.githubusercontent.com/vllm-project/vllm/main/examples/pooling/score/template/qwen3_vl_reranker.jinja \
  -o ~/models/Qwen/vllm-templates/qwen3_vl_reranker.jinja
```

再启动 Reranker 服务：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve \
  /home/wangyi/models/Qwen/Qwen3-VL-Reranker-2B \
  --runner pooling \
  --host 127.0.0.1 \
  --port 8300 \
  --served-model-name qwen3-vl-reranker-2b \
  --api-key local-reranker-key \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.22 \
  --limit-mm-per-prompt '{"image":1}' \
  --hf-overrides '{"architectures":["Qwen3VLForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}' \
  --chat-template ~/models/Qwen/vllm-templates/qwen3_vl_reranker.jinja
```

项目 `.env` 配置：

```env
EMBEDDING_PROVIDER=vllm
EMBEDDING_VLLM_BASE_URL=http://127.0.0.1:8200/v1
EMBEDDING_VLLM_API_KEY=local-embedding-key
EMBEDDING_VLLM_MODEL=qwen3-vl-embedding-2b
EMBEDDING_VLLM_BATCH_SIZE=8
EMBEDDING_VLLM_TIMEOUT_SECONDS=120

RERANKER_PROVIDER=vllm
RERANKER_VLLM_BASE_URL=http://127.0.0.1:8300/v1
RERANKER_VLLM_API_KEY=local-reranker-key
RERANKER_VLLM_MODEL=qwen3-vl-reranker-2b
RERANKER_VLLM_TIMEOUT_SECONDS=120
```

上述显存比例是 RTX 5090 同时运行生成式 VLM、Embedding 和 Reranker 时的起始参考值，不是固定要求。三个 vLLM 实例的显存预算总和应留出 CUDA、Docling 和其他进程所需余量。

## OCR 和 VLM

OCR 与图片描述 VLM 是彼此独立的增强能力。未配置时可以保持禁用：

```env
OCR_PROVIDER=disabled
PADDLEOCR_ACCESS_TOKEN=

VLM_PROVIDER=disabled
```

启用 PaddleOCR 云 API 或阿里云百炼 VLM 时：

1. 填写 `PADDLEOCR_ACCESS_TOKEN`，并将 `OCR_PROVIDER` 改为 `paddleocr`。
2. 填写 `DASHSCOPE_API_KEY`，并将 `VLM_PROVIDER` 改为 `dashscope`。
3. 重启 FastAPI；若使用 Celery，也要重启 Worker。

项目也支持通过 OpenAI-compatible Chat Completions API 调用本地 VLM。
`VLM_PROVIDER=vllm` 与 `VLM_PROVIDER=openai_compatible` 使用同一套配置，前者用于明确标识部署方式。

### 在服务器上部署 Qwen3-VL vLLM 服务

建议让 vLLM 使用独立 Python 环境和独立进程，不要安装到本项目虚拟环境中。在服务器终端执行：

```bash
python3 -m venv ~/venvs/qwen-vllm
source ~/venvs/qwen-vllm/bin/activate
python -m pip install --upgrade pip
pip install accelerate
pip install "qwen-vl-utils==0.0.14"
pip install --upgrade "vllm>=0.14.0"

CUDA_VISIBLE_DEVICES=0 vllm serve \
  /home/wangyi/models/Qwen/Qwen3-VL-8B-Instruct-FP8 \
  --host 127.0.0.1 \
  --port 8100 \
  --served-model-name qwen3-vl-8b-instruct-fp8 \
  --api-key local-vlm-key \
  --max-model-len 8192 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.42 \
  --limit-mm-per-prompt '{"image":1}'
```

RTX 5090 同时运行 VLM、Embedding 和 Reranker 时，先从 README 中三个服务合计约 `0.86` 的 vLLM 显存预算和单个 Celery Worker 开始；确认显存余量后再逐步提高并发。若这三个模型分配到不同 GPU，可以分别提高各服务的显存预算。

另开一个服务器终端验证 vLLM：

```bash
curl -sS http://127.0.0.1:8100/v1/models \
  -H "Authorization: Bearer local-vlm-key"
```

然后在项目服务器的 `.env` 中配置：

```env
VLM_PROVIDER=vllm
VLM_BASE_URL=http://127.0.0.1:8100/v1
VLM_API_KEY=local-vlm-key
VLM_MODEL=qwen3-vl-8b-instruct-fp8
VLM_MAX_TOKENS=800
VLM_TIMEOUT_SECONDS=120
```

`VLM_MODEL` 必须与 `--served-model-name` 一致。若 vLLM 未使用 `--api-key`，`VLM_API_KEY` 可留空；若 FastAPI 或 Celery 在 Docker 容器内运行，则 `127.0.0.1` 要改为容器能够访问的 vLLM 地址。

VLM 在文档入库阶段处理解析出的图片，描述结果会写入派生对象并参与 Embedding。修改 VLM 配置后必须同时重启 FastAPI 和 Celery Worker；已经入库的文档不会自动补充描述，需要使用 `force=true` 创建新版本并重新入库。

## 启动 API

确认 PostgreSQL、Qdrant 和 MinIO 已启动且数据库迁移完成，然后运行：

```powershell
cd C:\Development\multimodal-rag
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

常用地址：

- OpenAPI：`http://127.0.0.1:8000/docs`
- 存活检查：`http://127.0.0.1:8000/health/live`
- 就绪检查：`http://127.0.0.1:8000/health/ready`

开发身份可以通过请求头覆盖：

```text
X-Tenant-ID: tenant UUID
X-User-ID: user UUID
X-Roles: engineering,reviewer
X-Groups: group-a
```

未传入时使用开发环境的默认 Tenant 和 User。

## 上传、查看任务和搜索

上传文档：

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/documents `
  -F "file=@sample.pdf" `
  -F "parser=docling"
```

接口返回 HTTP 202，以及 `document_id`、`version_id` 和 `job_id`。使用返回的 `job_id` 查看导入进度：

```powershell
curl.exe http://127.0.0.1:8000/v1/jobs/<job_id>
```

文档状态变为 `search_ready` 后执行搜索：

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/search `
  -H "Content-Type: application/json" `
  -d '{"query":"查找系统架构图和相关说明","top_k":10}'
```

## Celery 执行模式

当前设置：

```env
CELERY_TASK_ALWAYS_EAGER=true
```

在此模式下，导入任务由 FastAPI 进程执行，适合本机单进程调试，不要求单独启动 Celery Worker。

需要测试真正的异步队列时，将其改为：

```env
CELERY_TASK_ALWAYS_EAGER=false
REDIS_URL=redis://:123456@127.0.0.1:6379/0
VECTOR_PROVIDER=qdrant
```

然后重启 API，并在另一个 PowerShell 窗口启动 Worker。Windows 本机建议使用 `solo` 池：

```powershell
cd C:\Development\multimodal-rag
.\.venv\Scripts\Activate.ps1
celery -A app.workers.celery_app:celery_app worker --loglevel=INFO --pool=solo
```

Worker 必须读取与 API 相同的 `.env`，并能访问 PostgreSQL、Redis、Qdrant、MinIO 和本地模型目录。无 GPU 环境不要随意启动多个 Worker，因为每个进程可能各自加载模型并占用大量内存。

## 测试与代码检查

```powershell
pytest
ruff check .
```

自动化测试使用临时 SQLite、本地对象存储、内存向量库和 Mock Provider，不会加载真实 Qwen 模型，也不会访问 PostgreSQL、Redis、Qdrant、MinIO 或外部 API。

Mock 表示用于测试的轻量替代实现：Embedding 生成确定性的哈希向量，Reranker 使用简单词项匹配。它只能验证程序链路，不能代表真实语义检索效果。
