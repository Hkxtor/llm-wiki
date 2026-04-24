# LLM Wiki

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

A CLI tool for managing a personal knowledge base using an LLM, backed by OceanBase vector search.

### Features
- **Vector Search**: Utilizes OceanBase's vector database capabilities for fast and accurate semantic search.
- **LLM Integration**: Designed to be used alongside LLM agents (like Claude) to easily ingest, search, and synthesize personal notes.
- **Markdown Support**: Naturally works with Markdown files for documentation and notes.

### Prerequisites
- Python >= 3.10
- An OceanBase database instance
- OpenAI-compatible API (for generating embeddings)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llm-wiki.git
   cd llm-wiki
   ```

2. Install dependencies using `uv` (or pip):
   ```bash
   uv pip install -e .
   ```

3. Configuration:
   Copy `.env.example` to `.env` and fill in your database and API credentials:
   ```bash
   cp .env.example .env
   ```

### Usage

1. **Initialization**: Initialize the database tables.
   ```bash
   llm-wiki init
   ```

2. **Sync**: Ingest documents from your `wiki/` directory into the vector database.
   ```bash
   llm-wiki sync
   ```

3. **Search**: Perform a semantic search query against your knowledge base.
   ```bash
   llm-wiki search "your query here"
   ```

### License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

<a name="chinese"></a>
## 中文

LLM Wiki 是一个基于命令行的个人知识库管理工具，结合大语言模型（LLM），并使用 OceanBase 向量数据库提供底层搜索支持。

### 主要特性
- **向量检索**：利用 OceanBase 的向量数据库功能，实现快速准确的语义搜索。
- **LLM 深度集成**：专为与 LLM Agent（如 Claude）配合使用而设计，方便摄入、搜索和综合个人笔记内容。
- **Markdown 支持**：原生支持 Markdown 格式的文档和笔记。

### 环境要求
- Python >= 3.10
- OceanBase 数据库实例
- 兼容 OpenAI 格式的 API（用于生成向量 Embedding）

### 安装指南

1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/llm-wiki.git
   cd llm-wiki
   ```

2. 使用 `uv`（或 pip）安装依赖：
   ```bash
   uv pip install -e .
   ```

3. 配置环境：
   将 `.env.example` 复制为 `.env`，并填入你的数据库连接信息和 API 密钥：
   ```bash
   cp .env.example .env
   ```

### 使用说明

1. **初始化**：初始化数据库表结构。
   ```bash
   llm-wiki init
   ```

2. **同步数据**：将 `wiki/` 目录下的文档提取并写入向量数据库中。
   ```bash
   llm-wiki sync
   ```

3. **搜索**：对你的知识库进行语义搜索。
   ```bash
   llm-wiki search "你的搜索关键词"
   ```

### 开源协议
本项目采用 Apache License 2.0 协议开源。详情请参阅 [LICENSE](LICENSE) 文件。
