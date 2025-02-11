import json
import base64
from openai import OpenAI

def encode_image(image_path):
    """
    将本地图片文件编码为 Base64 字符串
    :param image_path: 本地图片文件的路径
    :return: 图片的 Base64 编码字符串
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"错误：未找到图片文件 {image_path}")
        return None
    except Exception as e:
        print(f"错误：读取图片文件时出现异常 - {e}")
        return None

def call_vision_language_api(image_path, prompt, api_key, base_url, model="deepseek-ai/deepseek-vl2"):
    """
    调用视觉语言模型 API
    :param image_path: 本地图片文件的路径
    :param prompt: 文本提示信息
    :param api_key: 硅基流动的 API 密钥
    :param base_url: API 的基础 URL
    :param model: 要使用的模型名称，默认为 "deepseek-ai/deepseek-vl2"
    :return: 无，直接打印模型的响应结果
    """
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # 对图片进行编码
    image_base64 = encode_image(image_path)
    if image_base64 is None:
        return

    # 判断图片格式，生成正确的 data URL
    if image_path.lower().endswith('.png'):
        data_url = f"data:image/png;base64,{image_base64}"
    else:
        data_url = f"data:image/jpeg;base64,{image_base64}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            stream=True
        )

        for chunk in response:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message:
                print(chunk_message, end='', flush=True)
        print()  # 打印换行符，使输出更清晰
    except Exception as e:
        print(f"错误：调用 API 时出现异常 - {e}")

if __name__ == "__main__":
    # 替换为你从硅基流动平台获取的 API 密钥
    api_key = "sk-eqoxfispktkhpdunfatwbvterqnitnptnmgqotqtiyomryqo"
    # 替换为硅基流动提供的 API 的基础 URL
    base_url = "https://api.siliconflow.cn/v1"
    # 替换为你要使用的本地图片文件的实际路径
    image_path = "/data1/lfwj/PIVOT_R/PIVOT-R/images/pipeline.png"
    # 输入你要发送给模型的文本提示信息
    prompt = "Describe the image."

    call_vision_language_api(image_path, prompt, api_key, base_url)

