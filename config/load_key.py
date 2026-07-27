import os
from dotenv import load_dotenv


def load_key(key_name: str = "DEEPSEEK_API_KEY") -> str:
    """加载 API key，优先读环境变量，其次读 .env 文件"""
    load_dotenv()
    key = os.getenv(key_name)
    if not key:
        raise ValueError(f"未找到 {key_name}，请在 .env 文件中设置或设置环境变量")
    return key
