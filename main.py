"""FastAPI 入口：中药鉴定学 - 刘春生教授 AI 助教。
支持多轮对话历史，流式 SSE 输出。
"""
import json
import os
import secrets
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

from llm_client import generate_stream, generate, vision_client

app = FastAPI(title="中药鉴定学 - 刘春生教授 AI 助教")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 日志数据库 ============
DB_PATH = Path(__file__).parent / "chat_logs.db"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
if not ADMIN_PASSWORD:
    print("[WARN] 未设置 ADMIN_PASSWORD 环境变量，/admin 接口将禁用以防弱口令暴露")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            ip TEXT,
            ua TEXT,
            user_name TEXT,
            user_id TEXT,
            prompt TEXT,
            answer TEXT,
            think INTEGER DEFAULT 0,
            duration_ms INTEGER
        )
    """)
    # 给历史库补字段（已存在则忽略错误）
    for col in ("user_name TEXT", "user_id TEXT"):
        try:
            conn.execute(f"ALTER TABLE chat_log ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


init_db()


# ============ 账号系统 ============
ACCOUNTS_PATH = Path(__file__).parent / "accounts.json"
TOKENS: Dict[str, Dict] = {}  # token -> {account, name, expires}
TOKEN_TTL = 7 * 24 * 3600  # 7 天


def load_accounts() -> Dict[str, str]:
    """读取 accounts.json：{ "account": "password", ... }"""
    if not ACCOUNTS_PATH.exists():
        return {}
    try:
        with open(ACCOUNTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k.strip(): str(v) for k, v in data.items() if not k.startswith("_")}
        if isinstance(data, list):
            return {item["account"].strip(): str(item["password"]) for item in data if "account" in item}
        return {}
    except Exception as e:
        print(f"[load_accounts error] {e}")
        return {}


def issue_token(account: str, name: str) -> str:
    tk = secrets.token_urlsafe(24)
    TOKENS[tk] = {
        "account": account,
        "name": name,
        "expires": time.time() + TOKEN_TTL,
    }
    return tk


def verify_token(token: Optional[str]) -> Optional[Dict]:
    if not token:
        return None
    info = TOKENS.get(token)
    if not info:
        return None
    if info["expires"] < time.time():
        TOKENS.pop(token, None)
        return None
    return info


class LoginRequest(BaseModel):
    account: str
    password: str
    name: Optional[str] = ""


@app.post("/api/login")
async def api_login(req: LoginRequest):
    accounts = load_accounts()
    acc = req.account.strip()
    if not acc or acc not in accounts:
        raise HTTPException(401, "账号不存在")
    if accounts[acc] != req.password:
        raise HTTPException(401, "密码错误")
    name = (req.name or "").strip()[:32] or acc
    token = issue_token(acc, name)
    return {"token": token, "account": acc, "name": name}


def require_user(authorization: Optional[str] = Header(None)) -> Dict:
    if not authorization:
        raise HTTPException(401, "请先登录")
    token = authorization.replace("Bearer ", "").strip()
    info = verify_token(token)
    if not info:
        raise HTTPException(401, "登录已过期，请重新登录")
    return info


def log_chat(ip: str, ua: str, user_name: str, user_id: str, prompt: str, answer: str, think: bool, duration_ms: int):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO chat_log (ts, ip, ua, user_name, user_id, prompt, answer, think, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), ip, ua, user_name, user_id, prompt, answer, 1 if think else 0, duration_ms),
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
    user_name: Optional[str] = None
    user_id: Optional[str] = None


@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request, user: Dict = Depends(require_user)):
    if not req.prompt.strip() and not req.image:
        raise HTTPException(400, "问题不能为空")

    history_dicts: List[Dict[str, str]] = (
        [m.model_dump() for m in req.history] if req.history else []
    )

    client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "")
    client_ip = client_ip.split(",")[0].strip()
    user_agent = request.headers.get("user-agent", "")
    # 用 token 里的 account/name 而不是前端自报，杜绝伪造
    user_name = user["name"][:32]
    user_id = user["account"][:32]
    started = time.time()

    # 日志里的 prompt 标记是否带图
    prompt_for_log = req.prompt + ("  [📷 含图片]" if req.image else "")

    if not req.stream:
        text = await generate(
            req.prompt, history=history_dicts,
            temperature=req.temperature, think=req.think,
            image_data=req.image, user_name=user_name,
        )
        log_chat(client_ip, user_agent, user_name, user_id, prompt_for_log, text, req.think, int((time.time() - started) * 1000))
        return {"content": text}

    async def event_stream():
        full_answer = ""
        try:
            async for token in generate_stream(
                req.prompt, history=history_dicts,
                temperature=req.temperature, think=req.think,
                image_data=req.image, user_name=user_name,
            ):
                full_answer += token
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {err}\n\n"
        finally:
            log_chat(client_ip, user_agent, user_name, user_id, prompt_for_log, full_answer, req.think, int((time.time() - started) * 1000))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/features")
async def api_features():
    """前端启动时查询哪些可选功能（如视觉）可用。"""
    return {"vision": vision_client is not None}


# ============ 管理后台 ============
def check_admin(password: Optional[str]) -> bool:
    # 未设置环境变量则一律拒绝，防止默认 admin123 被滥用
    if not ADMIN_PASSWORD:
        return False
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


# ============ 账号管理 API（管理员）============
def save_accounts(accounts: Dict[str, str]):
    """写回 accounts.json，保留下划线开头字段。"""
    existing = {}
    if ACCOUNTS_PATH.exists():
        try:
            with open(ACCOUNTS_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
                if isinstance(raw, dict):
                    existing = {k: v for k, v in raw.items() if k.startswith("_")}
        except Exception:
            pass
    merged = {**existing, **accounts}
    with open(ACCOUNTS_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


class AccountAddReq(BaseModel):
    password: str
    account: str
    user_password: str


class AccountDelReq(BaseModel):
    password: str
    account: str


@app.get("/api/admin/accounts")
async def admin_accounts_list(password: str = ""):
    if not check_admin(password):
        raise HTTPException(401, "密码错误")
    accs = load_accounts()
    return {"accounts": [{"account": k, "password": v} for k, v in accs.items()]}


@app.post("/api/admin/accounts/add")
async def admin_accounts_add(req: AccountAddReq):
    if not check_admin(req.password):
        raise HTTPException(401, "密码错误")
    acc = req.account.strip()
    if not acc or acc.startswith("_"):
        raise HTTPException(400, "账号不合法（不能为空，不能以下划线开头）")
    if len(acc) > 32:
        raise HTTPException(400, "账号过长（最多 32 字符）")
    if not req.user_password:
        raise HTTPException(400, "密码不能为空")
    accs = load_accounts()
    accs[acc] = req.user_password
    save_accounts(accs)
    return {"ok": True}


@app.post("/api/admin/accounts/delete")
async def admin_accounts_delete(req: AccountDelReq):
    if not check_admin(req.password):
        raise HTTPException(401, "密码错误")
    accs = load_accounts()
    if req.account in accs:
        accs.pop(req.account)
        save_accounts(accs)
        # 立即使该账号当前所有 token 失效
        for tk in list(TOKENS.keys()):
            if TOKENS[tk]["account"] == req.account:
                TOKENS.pop(tk, None)
    return {"ok": True}


# 静态资源
static_dir = Path(__file__).parent / "static"

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html", headers=NO_CACHE_HEADERS)

    @app.get("/admin")
    async def admin_page():
        return FileResponse(static_dir / "admin.html", headers=NO_CACHE_HEADERS)
