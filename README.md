# 中药鉴定学 · 刘春生老师 AI 助教

模拟北京中医药大学中药学院刘春生教授授课风格的中药鉴定学 AI 助教，基于 FastAPI + DeepSeek。

## 功能特点

- 🎓 **人设定制**：刘春生老师教学风格，严谨学术、条理清晰
- 📚 **知识框架**：来源 → 性状 → 显微 → 理化 → 伪品鉴别
- 💬 **多轮对话**：前端保留历史，支持连续追问
- ⚡ **流式输出**：SSE 实时显示，体验流畅
- 🎨 **中式 UI**：仿古典配色，契合中医药主题

## 快速开始

```bash
cd ~/Desktop/tcm-liuchunsheng

# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key（https://platform.deepseek.com/）

# 3. 启动服务
uvicorn main:app --reload --port 8000
```

打开浏览器访问 http://localhost:8000

## 项目结构

```
tcm-liuchunsheng/
├── main.py            # FastAPI 入口 + /api/chat 接口
├── llm_client.py      # LLM 封装 + 刘春生老师 system prompt
├── requirements.txt
├── .env.example
├── static/
│   └── index.html     # 单页聊天界面（多轮对话）
└── README.md
```

## 切换大模型

编辑 `.env`：

| 提供商 | base_url | 模型 |
|--------|----------|------|
| DeepSeek（默认） | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| Kimi | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-plus` |

## 定制建议

- **人设调整**：修改 `llm_client.py` 中的 `LIU_CHUNSHENG_SYSTEM_PROMPT`
- **知识增强**：可接入向量检索（如 ChromaDB）存储《中药鉴定学》教材片段做 RAG
- **功能扩展**：
  - 上传药材图片 → 多模态识别（接入 Qwen-VL）
  - 题库模式：执业药师 / 考研 / 期末考点
  - 对比模式：并排展示正品 vs 伪品鉴别要点
- **持久化**：用 SQLite 保存用户会话历史

## 示例提问

- "人参的性状鉴别要点有哪些？"
- "黄连有几种来源？如何区分？"
- "大黄伪品鉴别怎么做？"
- "讨论一下川贝母与浙贝母的区别"
- "中药鉴定的依据有哪些？"（总论类）
