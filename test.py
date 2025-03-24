from openai import OpenAI
import os

# 设置 API 配置
os.environ["OPENAI_API_KEY"] = "sk-FfO0CEe94GPThYEruFjppQJsAbomAf9SlnXWvRRlhXlgfr2c"
os.environ["OPENAI_API_BASE"] = "https://xiaoai.plus/v1"

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_API_BASE"],
)

# 测试嵌入模型
try:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="Hello, world!"
    )
    print("嵌入模型可用！")
    # print(response.data[0].embedding)
except Exception as e:
    print(f"嵌入模型不可用，错误信息: {e}")