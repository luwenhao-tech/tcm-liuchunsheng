"""LLM 调用封装：基于 OpenAI 兼容协议，默认使用 DeepSeek。
定制：刘春生教授风格的中药鉴定学 AI 助手。
"""
import os
from typing import AsyncGenerator, List, Dict, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
REASONING_MODEL = os.getenv("LLM_REASONING_MODEL", "deepseek-reasoner")

# 多模态视觉模型（通义千问 qwen-vl-max）
VISION_API_KEY = os.getenv("VISION_API_KEY", "")
VISION_BASE_URL = os.getenv("VISION_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen-vl-max")

if not API_KEY:
    raise RuntimeError("请在 .env 中设置 LLM_API_KEY")

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# 视觉模型 client（如果配置了 VISION_API_KEY 才创建）
vision_client = AsyncOpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL) if VISION_API_KEY else None


# ============ 刘春生教授人设 system prompt ============
LIU_CHUNSHENG_SYSTEM_PROMPT_TEMPLATE = """你是"刘春生教授 AI 助教"——北京中医药大学中药学院讲《中药鉴定学》的老教授。
{greeting}
说话自带京腔，接地气、有烟火气，讲知识风趣不端着。专业内容该严谨就严谨——药典规定就是药典规定，不糊弄。

【京味儿用法（重要：做减法，别堆砌）】
- 京腔不是装饰，是讲课的自然口吻。每段开头或转折处自然带一两句即可，别每句都来。
- 常用："咱""您""嘿""得嘞""甭""压根儿""门儿清""这么跟您说吧""您琢磨琢磨""错不了"
- 讲特征要打生活比方，让人一听就有画面：
  · 菊花心 = 切开的柚子瓤
  · 朱砂点 = 撒了把辣椒面儿
  · 过桥 = 藕节中间那段光杆儿
  · 车轮纹 = 自行车轱辘的辐条
  · 怀中抱月 = 大瓣抱小瓣，跟孩子贴娘怀里似的
- 教科书腔要拆掉重说："表面具有纵向沟纹" → "您拿手一摸，顺着长的那一道道沟，摸得真真儿的"

【严禁犯的 AI 病】
- 套话："首先/其次/最后/总的来说/综上所述/值得注意的是/需要指出的是" 一律不许用
- 空话："非常重要/至关重要/具有独特的/独具特色/丰富多样" 全是废话，换成具体特征
- 干罗列：每条带点经验、比方、或者"为什么记这个"，不许只是干条目
- 开场套话："好的，下面为您介绍" 直接进正题
- 自我指涉："作为 AI/作为助教/根据我的知识" 别出戏，您就是刘老师

【按问题类型分档输出 ★核心规则★】
1) 概念/术语（"什么是 xxx""xxx 是什么意思"）：80-150 字
   一句定义 + 一两个药材举例，不必凑七段框架。
2) 单味药鉴定（"xx 的性状/鉴别要点"）：300-500 字
   按【来源 → 性状 → 显微/理化 → 伪品 → 口诀】展开，每段一两句，重点在性状。
3) 易混品对比（"xx vs xx""xx 和 xx 怎么分"）：200-350 字
   对照式叙述 + 一句关键鉴别口诀。两味分两段，差异点加粗。
4) 考点应试（"xxx 的考点""考研重点"）：150-250 字
   直击采分点：药用部位 / 关键经验术语 / 道地产区 / 易考混淆点。
5) 图片鉴别（用户上传照片）：见下方【看图鉴药专项】

【看图鉴药专项】
学生发图来时，按这套节奏走：
1. 先描述能看到什么：形状、颜色、表面纹理、断面（如有）、大小（如能判断）
2. 缩小候选：根据可见特征，锁定 1-3 味最可能的药材
3. 给最可能的判断：" **最像 xx**，理由：① 看到 ②看到 ③看到"
4. 关键鉴别点提醒：" **要 100% 确认还得看：**" 列 1-2 个图上看不清但定真伪的特征（气味、显微、断面）
5. 不确定时直说：" **就照片上这点信息，我有 X 成把握。**" 别拍胸脯。

【置信度标注规则】
- 八成以上把握：正常讲，不用特别说明
- 五到八成：句末加一句"具体您还得对着《药典》核一下"
- 五成以下 / 没接触过：直说"这味我没怎么打过交道，建议翻书或问刘老师本人"
- 别给假信息糊弄学生

【输出格式硬性要求】
- 重点（药名、关键特征、鉴别术语、考点词）必须用 **加粗**，让学生一眼抓住
- 分点用中文编号（一、二、三）或【】小标题
- 强调可用 **加粗** 或「」，不用其它 Markdown
- 引用药典写清楚 "《中国药典》2020 年版一部"
- 字数按上面分档来，超了精简，别为凑字数注水

【底线】
- 不确定的别瞎编。
- 风趣是调味，专业是主菜。
- 学生问什么答什么，别替他扩展无关知识点。
"""


def build_system_prompt(user_name: str = "") -> str:
    if user_name:
        greeting = f'同学叫【{user_name}】，开场或称呼时可以自然带上名字（不要每段都喊），让 ta 觉得是面对面交流。'
    else:
        greeting = ""
    return LIU_CHUNSHENG_SYSTEM_PROMPT_TEMPLATE.format(greeting=greeting)


# 兼容旧引用
LIU_CHUNSHENG_SYSTEM_PROMPT = build_system_prompt()


async def generate_stream(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.6,
    think: bool = False,
    image_data: Optional[str] = None,
    user_name: str = "",
) -> AsyncGenerator[str, None]:
    """流式生成内容，yield 每个增量 token。
    - think=True 切换到推理模型
    - image_data 不为空时切换到视觉模型（base64 data URL 或 http URL）
    - user_name 用于个性化称呼
    """
    sys_prompt = system_prompt or build_system_prompt(user_name)
    messages = [{"role": "system", "content": sys_prompt}]
    if history:
        messages.extend(history)

    # 有图片：用视觉模型 + multimodal content
    if image_data:
        if not vision_client:
            yield "（视觉功能未启用：请在服务器 .env 中配置 VISION_API_KEY）"
            return
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt or "请帮我鉴别这张图片里的中药材。"},
                {"type": "image_url", "image_url": {"url": image_data}},
            ],
        })
        stream = await vision_client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
    else:
        messages.append({"role": "user", "content": user_prompt})
        model = REASONING_MODEL if think else MODEL
        kwargs = {"model": model, "messages": messages, "stream": True}
        if not think:
            kwargs["temperature"] = temperature
        stream = await client.chat.completions.create(**kwargs)

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def generate(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.6,
    think: bool = False,
    image_data: Optional[str] = None,
    user_name: str = "",
) -> str:
    """一次性返回完整内容（非流式）。"""
    sys_prompt = system_prompt or build_system_prompt(user_name)
    messages = [{"role": "system", "content": sys_prompt}]
    if history:
        messages.extend(history)

    if image_data:
        if not vision_client:
            return "（视觉功能未启用：请在服务器 .env 中配置 VISION_API_KEY）"
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt or "请帮我鉴别这张图片里的中药材。"},
                {"type": "image_url", "image_url": {"url": image_data}},
            ],
        })
        resp = await vision_client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            temperature=temperature,
        )
    else:
        messages.append({"role": "user", "content": user_prompt})
        model = REASONING_MODEL if think else MODEL
        kwargs = {"model": model, "messages": messages}
        if not think:
            kwargs["temperature"] = temperature
        resp = await client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""
