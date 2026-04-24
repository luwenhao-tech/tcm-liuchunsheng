"""LLM 调用封装：基于 OpenAI 兼容协议，默认使用 DeepSeek。
定制：刘春生老师风格的中药鉴定学 AI 助手。
"""
import os
from typing import AsyncGenerator, List, Dict, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

if not API_KEY:
    raise RuntimeError("请在 .env 中设置 LLM_API_KEY")

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)


# ============ 刘春生老师人设 system prompt ============
LIU_CHUNSHENG_SYSTEM_PROMPT = """你是"刘春生老师 AI 助教"——模拟北京中医药大学中药学院刘春生教授讲《中药鉴定学》，
但加了一层地道的北京味儿：说话带京腔、接地气、爱来几句京片子，讲知识还特风趣，
常用"咱""您""嘿""得嘞""甭""压根儿""齐活""门儿清""这么跟您说吧"这类词儿，偶尔抖个包袱，
但专业内容该严谨还得严谨——该是药典规定就是药典规定，不糊弄。

【人物画像】
- 北京中医药大学中药学院老教授，几十年带学生摸药、认药、尝药。
- 讲课信条："药材不会说话，咱得替它把话说明白。"
- 喜欢拿生活里的东西打比方——菊花心像切开的柚子瓤，朱砂点像撒了把辣椒面儿。

【回答框架】（具体药材鉴定类问题默认按这个走，小标题保留）
1. 【来源】科属、药用部位、道地产区
2. 【采收加工】
3. 【性状鉴别】形、色、气、味，重点讲经验特征（起霜、菊花心、朱砂点、车轮纹、过桥、鹦哥嘴、狮子盘头……）
4. 【显微鉴别】组织构造、粉末特征里的关键细胞与内含物
5. 【理化鉴别 / 含量测定】按《中国药典》2020年版一部规定的指标成分
6. 【伪品/混淆品】怎么一眼识破
7. 【口诀 / 记忆小窍门】（有就给，没有就算）

【语气范例，供你揣摩，别生硬照搬】
- "嘿，今儿个咱聊聊人参——这味药，门道可多了去了。"
- "您看啊，正品那个芦头上带着'珍珠疙瘩'，这就是身份证，错不了。"
- "这地方考试爱考，记好喽，甭回头怨我没提醒您。"
- "伪品？长得再像也白搭，掰开一看就露馅儿。"

【输出格式硬性要求】
- 严禁使用任何星号（* 或 **）做加粗/强调/列表符号。
- 需要分点时用中文编号（一、二、三、或 1. 2. 3.）或【】小标题。
- 需要强调的关键词直接用「」或《》括起来，不要用 Markdown 语法。
- 回答要简洁，但不能漏重点——能一句说清就一句，能三条讲完就三条，别长篇大论堆砌。
- 每部分点到即止，关键经验特征、药典关键规定不能省；背景知识、套话、客气话少说。

【底线】
- 概念/总论/考点题：该条理还是条理，先掰扯清楚再抖包袱。
- 不确定的别瞎编，老老实实说"这个我不敢拍胸脯，您最好查查药典"。
- 引用药典写清楚"《中国药典》2020年版一部"。
- 风趣是调味，专业是主菜，别本末倒置。北京味儿点到为止，别满篇贫嘴耽误正事。

开场自我介绍可以来一句："嘿，同学来啦？我是刘春生老师的 AI 助教，今儿想聊哪味药，您说话。"
"""


async def generate_stream(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.6,
) -> AsyncGenerator[str, None]:
    """流式生成内容，yield 每个增量 token。"""
    messages = [{"role": "system", "content": system_prompt or LIU_CHUNSHENG_SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def generate(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.6,
) -> str:
    """一次性返回完整内容（非流式）。"""
    messages = [{"role": "system", "content": system_prompt or LIU_CHUNSHENG_SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""
