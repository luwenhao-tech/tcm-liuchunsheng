"""FastAPI 入口：中药鉴定学 - 刘春生教授 AI 助教。
支持多轮对话历史，流式 SSE 输出。
"""
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llm_client import generate_stream, generate

app = FastAPI(title="中药鉴定学 - 刘春生教授 AI 助教")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 日志数据库 ============
DB_PATH = Path(__file__).parent / "chat_logs.db"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            ip TEXT,
            ua TEXT,
            prompt TEXT,
            answer TEXT,
            think INTEGER DEFAULT 0,
            duration_ms INTEGER
        )
    """)
    conn.commit()
    conn.close()


init_db()


def log_chat(ip: str, ua: str, prompt: str, answer: str, think: bool, duration_ms: int):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO chat_log (ts, ip, ua, prompt, answer, think, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), ip, ua, prompt, answer, 1 if think else 0, duration_ms),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[log_chat error] {e}")


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    prompt: str
    history: Optional[List[Message]] = None
    temperature: float = 0.6
    stream: bool = True
    think: bool = False
    image: Optional[str] = None  # base64 data URL: "data:image/jpeg;base64,..."


@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request):
    if not req.prompt.strip() and not req.image:
        raise HTTPException(400, "问题不能为空")

    history_dicts: List[Dict[str, str]] = (
        [m.model_dump() for m in req.history] if req.history else []
    )

    client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "")
    client_ip = client_ip.split(",")[0].strip()
    user_agent = request.headers.get("user-agent", "")
    started = time.time()

    # 日志里的 prompt 标记是否带图
    prompt_for_log = req.prompt + ("  [📷 含图片]" if req.image else "")

    if not req.stream:
        text = await generate(
            req.prompt, history=history_dicts,
            temperature=req.temperature, think=req.think,
            image_data=req.image,
        )
        log_chat(client_ip, user_agent, prompt_for_log, text, req.think, int((time.time() - started) * 1000))
        return {"content": text}

    async def event_stream():
        full_answer = ""
        try:
            async for token in generate_stream(
                req.prompt, history=history_dicts,
                temperature=req.temperature, think=req.think,
                image_data=req.image,
            ):
                full_answer += token
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {err}\n\n"
        finally:
            log_chat(client_ip, user_agent, prompt_for_log, full_answer, req.think, int((time.time() - started) * 1000))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ============ 管理后台 ============
def check_admin(password: Optional[str]) -> bool:
    return password == ADMIN_PASSWORD


@app.get("/api/admin/logs")
async def admin_logs(password: str = "", limit: int = 200, offset: int = 0):
    if not check_admin(password):
        raise HTTPException(401, "密码错误")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM chat_log ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM chat_log").fetchone()[0]
    conn.close()
    return {"total": total, "logs": [dict(r) for r in rows]}


@app.get("/api/admin/stats")
async def admin_stats(password: str = ""):
    if not check_admin(password):
        raise HTTPException(401, "密码错误")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    total = cur.execute("SELECT COUNT(*) FROM chat_log").fetchone()[0]
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = cur.execute("SELECT COUNT(*) FROM chat_log WHERE ts LIKE ?", (today + "%",)).fetchone()[0]
    unique_ips = cur.execute("SELECT COUNT(DISTINCT ip) FROM chat_log").fetchone()[0]
    today_ips = cur.execute("SELECT COUNT(DISTINCT ip) FROM chat_log WHERE ts LIKE ?", (today + "%",)).fetchone()[0]
    think_count = cur.execute("SELECT COUNT(*) FROM chat_log WHERE think=1").fetchone()[0]
    think_rate = round(think_count / total * 100, 1) if total else 0

    # 最近 7 天每日问答数
    daily = []
    for i in range(6, -1, -1):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        c = cur.execute("SELECT COUNT(*) FROM chat_log WHERE ts LIKE ?", (d + "%",)).fetchone()[0]
        daily.append({"date": d[5:], "count": c})

    # 热门关键词（按 prompt 出现的常见药材/术语统计）
    hot_keywords = ["人参", "黄连", "大黄", "甘草", "当归", "何首乌", "三七", "天麻",
                    "川芎", "白芍", "黄芪", "防风", "党参", "川贝母", "浙贝母",
                    "菊花心", "朱砂点", "起霜", "车轮纹", "鹦哥嘴", "过桥",
                    "狮子盘头", "怀中抱月", "云锦花纹", "金井玉栏", "蚯蚓头", "星点", "珍珠疙瘩"]
    hot = []
    for kw in hot_keywords:
        c = cur.execute("SELECT COUNT(*) FROM chat_log WHERE prompt LIKE ?", (f"%{kw}%",)).fetchone()[0]
        if c > 0:
            hot.append({"keyword": kw, "count": c})
    hot.sort(key=lambda x: -x["count"])

    # 热门 IP
    top_ips = cur.execute(
        "SELECT ip, COUNT(*) AS c FROM chat_log GROUP BY ip ORDER BY c DESC LIMIT 10"
    ).fetchall()

    conn.close()
    return {
        "total": total,
        "today_count": today_count,
        "unique_ips": unique_ips,
        "today_ips": today_ips,
        "think_count": think_count,
        "think_rate": think_rate,
        "daily": daily,
        "hot_keywords": hot[:20],
        "top_ips": [{"ip": r[0], "count": r[1]} for r in top_ips],
    }


# 静态资源
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/admin")
    async def admin_page():
        return FileResponse(static_dir / "admin.html")
