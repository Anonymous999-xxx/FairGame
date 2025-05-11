"""Response Function of LLM and MLLM"""

from openai import OpenAI, AsyncOpenAI
import os
import json
from typing import Callable, Dict, Optional, Union, List, Tuple, Set, Any
import logging

def filter_config(config_list, filter_dict):
    """Filter the config list by provider and model.

    Args:
        config_list (list): The config list.
        filter_dict (dict, optional): The filter dict with keys corresponding to a field in each config,
            and values corresponding to lists of acceptable values for each key.

    Returns:
        list: The filtered config list.
    """
    if filter_dict:
        config_list = [
            config for config in config_list if all(config.get(key) in value for key, value in filter_dict.items())
        ]
    return config_list

def config_list_from_json(
    env_or_file: str,
    file_location: Optional[str] = "",
    filter_dict: Optional[Dict[str, Union[List[Union[str, None]], Set[Union[str, None]]]]] = None,
) -> List[Dict]:
    """Get a list of configs from a json parsed from an env variable or a file.

    Args:
        env_or_file (str): The env variable name or file name.
        file_location (str, optional): The file location.
        filter_dict (dict, optional): The filter dict with keys corresponding to a field in each config,
            and values corresponding to lists of acceptable values for each key.
            e.g.,
    ```python
    filter_dict = {
        "api_type": ["open_ai", None],  # None means a missing key is acceptable
        "model": ["gpt-3.5-turbo", "gpt-4"],
    }
    ```

    Returns:
        list: A list of configs for openai api calls.
    """
    json_str = os.environ.get(env_or_file)
    if json_str:
        config_list = json.loads(json_str)
    else:
        config_list_path = os.path.join(file_location, env_or_file)
        try:
            with open(config_list_path) as json_file:
                config_list = json.load(json_file)
        except FileNotFoundError:
            logging.warning(f"The specified config_list file '{config_list_path}' does not exist.")
            return []
    return filter_config(config_list, filter_dict)

def get_response(
    messages: List[Dict],
    llm_config: Dict, 
    model_type: str = "llm",
    temperature: float = 1.0,
    max_tokens: int = 4096,
    top_p: float = 0.7,
    top_k: int = 50
) -> str:
    client = OpenAI(
        api_key=llm_config["api_key"],
        base_url=llm_config["base_url"],
    )
    if model_type == "llm":
        response = client.chat.completions.create(
            model=llm_config["model"],
            messages=messages,
            stream=False,
            temperature = temperature, # 1.0
            max_tokens = max_tokens, # 4096
            top_p = top_p, # 0.05,
            top_k = top_k, # 40
        )
    elif model_type == "default":
        response = client.chat.completions.create(
            model=llm_config["model"],
            messages=messages,
            stream=False,
        )
    else:
        response = client.chat.completions.create(
            model=llm_config["model"],
            messages=messages,
            stream=False,
            temperature = temperature, # 1.0
            max_tokens = max_tokens, # 4096 65536
            top_p = top_p, # 1,
            top_k = top_k, # 40
        )
    #print("==========================")
    #print("prompt_tokens: {}".format(response.usage.prompt_tokens))
    #print("completion_tokens: {}".format(response.usage.completion_tokens))
    #print("total_tokens: {}".format(response.usage.total_tokens))
    #print("==========================")
    #print(response.choices[0].message.content)
    #print("==========================")
    return response.choices[0].message.content

async def get_response_async(
    messages: List[Dict],
    llm_config: Dict, 
    model_type: str = "llm",
    max_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 0.7,
    # top_k: int = 50
) -> str:
    """
    异步版本的 get_response 函数，用于发送异步请求到远程 API。
    """
    if "base_url" in llm_config:
        client = AsyncOpenAI(
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
        )
    else:
        client = AsyncOpenAI(
            api_key=llm_config["api_key"]
        )

    try:
        if model_type == "llm":
            response = await client.chat.completions.create(
                model=llm_config["model"],
                messages=messages,
                stream=False,
                temperature = temperature, # 1.0
                max_tokens = max_tokens, # 4096 65536
                top_p = top_p, # 1,
                # top_k = top_k, # 40
            )
        elif model_type == "default":
            response = client.chat.completions.create(
                model=llm_config["model"],
                messages=messages,
                stream=False,
            )
        else:
            response = await client.chat.completions.create(
                model=llm_config["model"],
                messages=messages,
                stream=False,
                temperature = temperature, # 1.0
                max_tokens = max_tokens, # 4096 65536
                top_p = top_p, # 1,
                # top_k = top_k, # 40
            )
        #print("==========================")
        #print("prompt_tokens: {}".format(response.usage.prompt_tokens))
        #print("completion_tokens: {}".format(response.usage.completion_tokens))

        #print("total_tokens: {}".format(response.usage.total_tokens))
        #print("==========================")

        #print(response.choices[0].message.content)
        #print("==========================")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during API request: {e}")
        return "ERROR"
    
if __name__ == '__main__':
    model = "deepseek-chat" # "qwen-vl-max" or "deepseek-chat"
    model_type = "llm" # "mllm" or "llm"
    current_dir = os.getcwd() # 获取当前目录
    data_set = os.path.join(current_dir, 'data_set') # 指定原始数据集文件夹路径

    filter_dict = {"model": model}
    llm_config = config_list_from_json(env_or_file="./configure_list_20241014.json", filter_dict=filter_dict)[0]
    print("llm_config: \n{}".format(llm_config))
    
    if model_type == "llm":
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Who are you?"},
        ]
    else:
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "这些是什么"},
                    {"type": "image_url", "image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}},
                    {"type": "image_url", "image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/tiger.png"}}
                ]
            }
        ]

    output = get_response(messages, llm_config, model_type=model_type)
    