from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent
from langchain.messages import HumanMessage
import os
import sqlite3

# 1.加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 2.web搜索工具，使用tavily作为web搜索工具
web_search = TavilySearch(
    max_result = 5,
    topic = "general"
)

# 3.多模态模型
model = init_chat_model(
    model="qwen3.7-plus",
    model_provider="openai",
    base_url = os.getenv("DASHSCOPE_BASE_URL"),
    api_key = os.getenv("DASHSCOPE_API_KEY")
)

# 4.初始化checkpointer
# 连接sqlite
connection = sqlite3.connect("../db/personal_chief.db", check_same_thread=False)
# 初始化checkpointer
checkpointer = SqliteSaver(connection)
# 自动建表
checkpointer.setup()

# 4.Agent系统提示词
system_prompt = """
你是一名私人厨师。收到用户提供的食材照片或清单后，请按以下流程操作[reference:6][reference:7]：

1.  **识别和评估食材**：若用户提供照片，首先辨识所有可见食材[reference:8]。基于食材的外观状态，评估其新鲜度与可用量，整理出一份“当前可用食材清单”[reference:9][reference:10]。
2.  **智能食谱检索**：**优先调用 `web_search` 工具**，以“可用食材清单”为核心关键词，查找可行菜谱[reference:11][reference:12]。
3.  **多维度评估与排序**：从营养价值和制作难度两个维度对检索到的候选食谱进行量化打分，并根据得分排序，让制作简单且营养丰富的食谱排名靠前[reference:13][reference:14]。
4.  **结构化方案输出**：把排序后的食谱整理为一份结构清晰的建议报告，要包含食谱信息、得分、推荐理由、食谱的参考图片，帮助用户快速做出决策[reference:15][reference:16]。

请**严格按照流程**，**优先调用 `web_search` 工具**搜索食谱，搜索不到的情况下才能自己发挥[reference:17][reference:18]。
"""


# 5.创建Agent
agent = create_agent(
    model=model,
    tools=[web_search],
    checkpointer=checkpointer,  # 记忆
    system_prompt=system_prompt,
    # 基于LangSmith是不需要checkpoint
)

'''
- LangGraph会自动托管Agent的记忆，因此代码中不用自己添加checkpointer！
- LangGraph自带Restful的API接口，我们只要定义好Agent就可以，其它不用管
'''


# langgraph.json
# {
#     "dependencies": ["."],
#     "graphs": {
#         "chief_agent": "./app/agents/personal_chief.py:agent"
#     },
#     "env": ".env"
# }
# [Agent文件路径]:[Agent变量名]

# - ./app/agents/personal_chief.py：就是文件路径
# - agent：就是文件中第5步定义的Agent名字