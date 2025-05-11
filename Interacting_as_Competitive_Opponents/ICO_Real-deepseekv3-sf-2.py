import asyncio
import os
import re
import json
import time
import random
from tqdm import tqdm
from typing import Callable, Dict, Optional, Union, List, Tuple, Set, Any
import logging
import matplotlib.font_manager as fm
from utils import *
from llm_response import *

# Global Variables:
# Setting Chinese Font:
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei as the font
plt.rcParams['axes.unicode_minus'] = False  # Resolve the negative sign display issue

# Initialize request and response queues
response_queue = asyncio.Queue()

# Setting colored output:
try:
    from termcolor import colored
except ImportError:
    def colored(x, *args, **kwargs):
        return x

# Create a token bucket to limit requests to 1000/min
token_bucket = TokenBucket(rate_limit=1000)

async def limited_get_response_async(
    task_id: int,
    messages: List[Dict],
    llm_config: Dict, 
    model_type: str = "llm",
    max_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 0.7,
    # top_k: int = 50
) -> None:
    """限流版本的异步请求函数"""
    # 获取一个令牌
    await token_bucket.acquire()
    # 发送请求
    response = await get_response_async(
        messages, llm_config, model_type, 
        max_tokens = max_tokens, 
        temperature = temperature,
        top_p = top_p,
        # top_k = top_k
    )
    # 将任务ID和响应放入队列
    await response_queue.put((task_id, response))

def shuffle_options(options: dict) -> dict:
    new_options = {}
    for language, option_list in options.items():
        original_chioce = option_list[0] + "\n" + option_list[1] + "\n" + option_list[2] + "\n"
        switch_chioce = option_list[0] + "\n" + option_list[2] + "\n" + option_list[1] + "\n"
        new_options[language] = [original_chioce, switch_chioce]
    return new_options

def generate_versus(country: dict) -> dict:
    versus = {}
    for language, option_list in country.items():
        versus[language] = []
        # 使用双重循环生成所有不重复的组合
        for i in range(len(option_list)):
            for j in range(i + 1, len(option_list)):
                versus[language].append([option_list[i], option_list[j]])
                versus[language].append([option_list[j], option_list[i]])
    return versus

def construct_diplomatic_game_qurey(
    original_prompt: str, 
    game_name: str,
    versus_pair: list,
    diplomatic_options: str,
) -> str:
    """Replace some words to construct a new qurey_prompt."""
    modified_content = original_prompt.replace('[game_name]', game_name) \
                                      .replace('[country_name_1]', versus_pair[0]) \
                                      .replace('[country_name_2]', versus_pair[1]) \
                                      .replace('[diplomatic_options]', diplomatic_options)
    return modified_content

def reshape_result_list(input_list, l1_num, l2_num, l3_num, l4_num, repeated_num):
    # 首先检查输入列表的长度是否符合预期
    if len(input_list) != l1_num * l2_num * l3_num * l4_num * repeated_num:
        raise ValueError("输入列表的长度与预期的四维结构不匹配")
    
    # 初始化四维列表
    reshaped_list = []
    
    # 开始重构
    for i in range(l1_num):
        l1_subclasses = []
        for j in range(l2_num):
            l2_subclasses = []
            for k in range(l3_num):
                l3_subclasses = []
                for l in range(l4_num):
                    l4_subclasses = []
                    for r in range(repeated_num):
                        index = i * (l2_num * l3_num * l4_num * repeated_num) + j * (l3_num * l4_num * repeated_num) + k * (l4_num * repeated_num) + l * repeated_num + r
                        l4_subclasses.append(input_list[index])
                    l3_subclasses.append(l4_subclasses)
                l2_subclasses.append(l3_subclasses)
            l1_subclasses.append(l2_subclasses)
        reshaped_list.append(l1_subclasses)
    
    return reshaped_list

def filter_and_trans_results(repeated_num: int, redundancy: int, results: list, trans_list: list):
    """
    此函数的目的是：
    1.将输出的选择，从文本转换成对应的数字
    2.对于每(repeated_num + redundancy) 个结果，选择前repeated_num个

    results里有 n * (repeated_num + redundancy) 个元素，n是整数
    """
    # 筛选成功处理的 response:
    new_results = []
    # 遍历 record，每 repeated_num + redundancy 个元素为一组
    for i in range(0, len(results), repeated_num + redundancy):
        # 获取当前组的元素
        group = results[i:i + repeated_num + redundancy]
    
        # 过滤出符合规定的输出
        valid_numbers = []
        for x in group:
            if x == None:
                pass # 输出不符合规定
            elif x.lower() == trans_list[0].lower(): # 中立
                valid_numbers.append(0)
            elif x.lower() == trans_list[1].lower(): # 反对
                valid_numbers.append(1)
            elif x.lower() == trans_list[2].lower(): # 合作
                valid_numbers.append(2)
            else:
                pass # 输出不符合规定

        # 检查有效值的数量是否足够
        if len(valid_numbers) < repeated_num:
            print("Not enough valid numbers in group starting at index {}. \nExpected {}, found {}.".format(
                i/(repeated_num + redundancy), repeated_num, len(valid_numbers)
            ))
            return False, None
    
        # 取前 repeated_num 个有效值
        new_results.extend(valid_numbers[:repeated_num])
    return True, new_results
        

async def DGC_evaluate(
    evaluated_language: str = "english", # "english", "chinese", "arabic"
    model: str = "deepseek-chat",        # Model name
    model_type: str= "llm",              # llm or mllm or default, use "llm" defaultly
    api_config: dict = None,             # api config, include temperature, max tokens, etc
    game_name: dict = None,              # Game Name needed to be evaluated
    country: dict = None,                # Countries needed to be evaluated
    repeated_num: int = 10,              # Number of repeated tests
    redundancy: int = 0,                 # Number of redundancy tests
    batch_size: int = 4,                 # Number of tasks in one batch, ralated to LLM API access limitation
    TPM_sleep: int = 3,                  # Retry how many times when TPM or RPM is exceeded
) -> dict:
    """evaluate choice bias in diplomatic game."""
    filter_dict = {"model": model}
    # llm_config = config_list_from_json(env_or_file="./configure_list_20241014.json", filter_dict=filter_dict)[0]
    llm_config = config_list_from_json(env_or_file="./configure_list_sf-2.json", filter_dict=filter_dict)[0]
    print("llm_config: \n{}\n{}".format(llm_config, api_config))

    # Observation Space:
    game_name = game_name[evaluated_language]
    versus = generate_versus(country)
    versus_list = versus[evaluated_language] # country versus list

    if "english" == evaluated_language:
        diplomatic_advantage = read_txt("./DGC/en_diplomatic_advantage.txt")
        diplomatic_disadvantage = read_txt("./DGC/en_diplomatic_disadvantage.txt")
        diplomatic_prompts = [diplomatic_advantage, diplomatic_disadvantage]
    elif "chinese" == evaluated_language:
        diplomatic_advantage = read_txt("./DGC/ch_diplomatic_advantage.txt")
        diplomatic_disadvantage = read_txt("./DGC/ch_diplomatic_disadvantage.txt")
        diplomatic_prompts = [diplomatic_advantage, diplomatic_disadvantage]
    else:
        print(colored("Wrong Language Selection!", "red"))
    
    # Action Space:
    diplomatic_options = shuffle_options(
        read_json('./DGC/diplomatic_options.json')
    )
    diplomatic_options = diplomatic_options[evaluated_language]
    choice_to_number = read_json('./DGC/choice_to_number.json')
    
    # Generate Asyn Tasks:
    record = []
    task_info = {
        "evaluated_language": evaluated_language,
        "game_name": game_name,
        "country_num": len(country[evaluated_language]),
        "country": country[evaluated_language],
        "versus_num": len(versus_list),
        "versus": versus_list,
        "choices": choice_to_number[evaluated_language],
        "prompt_num": len(diplomatic_prompts),
        "diplomatic_option_num": len(diplomatic_options),
        "repeated_num": repeated_num,
        "redundancy": redundancy,
    }

    repeated_num += redundancy
    tasks = []
    for gn in range(len(game_name)):
        for i in range(len(versus_list)):
            for j in range(len(diplomatic_prompts)):
                for k in range(len(diplomatic_options)):
                    # Prepare qurey_prompt:
                    qurey_prompt = construct_diplomatic_game_qurey(
                        original_prompt = diplomatic_prompts[j], 
                        game_name = game_name[gn],
                        versus_pair = versus_list[i],
                        diplomatic_options = diplomatic_options[k]
                    )
                    # print("qurey:\n{}\n".format(qurey_prompt))

                    for r in range(repeated_num):
                        if model_type == "llm" or model_type == "mllm" or model_type == "default":
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant"},
                                {"role": "user", "content": qurey_prompt},
                            ]
                        else:
                            print(colored("Use wrong model!", "red"))
                
                        # 创建异步任务信息：
                        task_params = {  # 存储任务的参数
                            "task_id": gn * (len(game_name) * len(diplomatic_prompts) * len(diplomatic_options) * repeated_num) 
                                + i * (len(diplomatic_prompts) * len(diplomatic_options) * repeated_num) 
                                + j * (len(diplomatic_options) * repeated_num) 
                                + k * repeated_num 
                                + r + 1,
                            "messages": messages,
                            "llm_config": llm_config,
                        }
                        tasks.append(task_params)  # 将任务参数添加到 tasks 列表
    
    repeated_num -= redundancy

    # Execute tasks in batches
    print("Total Task Num: {}".format(len(tasks)))
    record = []
    first_call = True
    for b in range(0, len(tasks), batch_size * (repeated_num + redundancy)):
        start_time = time.time()
        real_batch_size = min(batch_size * (repeated_num + redundancy), len(tasks) - b)
        retry_count = 0
        while retry_count < TPM_sleep:  # 最多重试TPM_sleep次
            if not first_call:
                print("SLEEP FOR 61 SECONDS...")
                await asyncio.sleep(61)
                print("SLEEP OVER!")
                first_call = False
                
            temp_record = []

            # 重新创建协程对象
            batch = [
                limited_get_response_async(
                    task_id=tasks[b + rb]["task_id"],  # 从任务参数中获取 task_id
                    messages=tasks[b + rb]["messages"],  # 从任务参数中获取 messages
                    llm_config=tasks[b + rb]["llm_config"],  # 从任务参数中获取 llm_config
                    max_tokens = api_config["max_tokens"], 
                    temperature = api_config["temperature"],
                    top_p = api_config["top_p"],
                    # top_k = api_config["top_k"]
                )
                for rb in range(real_batch_size)
            ]
            await asyncio.gather(*batch)

            # Retrieve responses from the Queue and sort them by task ID
            responses = []
            while not response_queue.empty():
                task_id, response = await response_queue.get()
                responses.append((task_id, response))

            # Sort by task ID
            responses.sort(key=lambda x: x[0])

            # 打印结果
            # print("++++++++++++++++++++++++++++++++++++++++")
            # for task_id, response in responses:
            #     print(f"Task {task_id}: {response}\n")
            # print("++++++++++++++++++++++++++++++++++++++++")

            # Process the response results
            miss_match_count = 0
            for r_i in range(len(responses)):
                # print(response)
                try:
                    # Use regular expressions to extract content inside {}
                    match = re.findall(r'\{([^}]+)\}', responses[r_i][1])
                    if match:
                        # Get the last matched content
                        content = match[-1].strip()

                        # 过滤出符合格式要求的输出:
                        if content.lower() in [item.lower() for item in choice_to_number[evaluated_language]]:
                            temp_record.append(content)  # 成功处理，放入 record
                            # print("---------extracted selection:---------")
                            # print(content)
                            # print("--------------------------------------")
                        else:
                            print(colored(f"{responses[r_i][0]} Output Format Error!", "yellow"))
                            temp_record.append(None) # 未匹配到内容，放入 None
                            miss_match_count += 1
                            # print("content: {}\n".format(content))
                            # print("choice_to_number{}: {}\n".format(evaluated_language, choice_to_number[evaluated_language]))
                            # print("----------")
                            # print("response[{}]:\n{}".format(responses[r_i][0], responses[r_i][1]))
                            # print("----------")
                    else:
                        print(colored(f"{responses[r_i][0]} Miss Match!", "yellow"))
                        temp_record.append(None) # 未匹配到内容，放入 None
                        miss_match_count += 1
                        continue  # 跳过当前循环，继续下一个 response
                except ValueError as e:
                    print(colored(f"{responses[r_i][0]} ValueError", "red"))
                    temp_record.append(None) # 未匹配到内容，放入 None
                    miss_match_count += 1
                    continue  # 跳过当前循环，继续下一个 response
                except Exception as e:
                    print(colored(f"{responses[r_i][0]} Unexpected error", "red"))
                    temp_record.append(None) # 未匹配到内容，放入 None
                    miss_match_count += 1
                    continue  # 跳过当前循环，继续下一个 response
            
            # 筛选成功处理的 response:
            successfal_flag, temp_record = filter_and_trans_results(
                repeated_num, 
                redundancy, 
                results=temp_record, 
                trans_list=choice_to_number[evaluated_language]
            )
            
            if miss_match_count >= real_batch_size or not successfal_flag:
                retry_count += 1
                if retry_count >= TPM_sleep:  # 如果重试次数超过TPM_sleep次
                    raise RetryLimitExceededError("Retry limit exceeded! TPM limit may still be in effect. Exiting program.")
                print(colored("TPM Limit reached! Sleeping for an additional 61 seconds...", "yellow"))
                time.sleep(61)
                print("sleep over!")
                first_call = True
                continue  # 重新执行当前迭代
            else:
                record += temp_record
                first_call = False
                print("This attempt succeeded!")
                break  # 退出重试循环，继续下一个 batch
        
        json_dump_path = './record/Diplomatic_Game_Choices_real_deepseekv3_raw-{}-temp-sf-2.json'.format(evaluated_language)
        write_to_json(record, json_dump_path)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Run Time of One Loop: {elapsed_time:.2f} s")
        print(f"One batch is done! batch_size: {len(responses)}\n")
        print("Continue Next Batch...")

        #-----------------------------------------------------
        
    # print("++++++++++++++++++++++++++++++++++++++++")
    # print(f"Record_Num: \n{len(record)}\n")
    # print(f"Record: \n{record}\n")
    # print("++++++++++++++++++++++++++++++++++++++++")

    # Reshape and record the results:
    task_info["record"] = reshape_result_list(
        input_list = record, 
        l1_num = len(game_name),
        l2_num = len(versus_list), 
        l3_num = len(diplomatic_prompts), 
        l4_num = len(diplomatic_options), 
        repeated_num =repeated_num
    )

    mes = colored(
        "Info of results collected:\n" +
        "Evaluated Language: {}\n".format(evaluated_language) +
        "Number of Versus Combinations: {}\n".format(len(versus_list)) +
        "Number of Diplomatic Situations: {}\n".format(len(diplomatic_prompts)) +
        "Number of Combinations of Diplomatic Options: {}".format(len(diplomatic_options)),
        "yellow"
    )
    print(mes)

    return task_info

async def multi_evaluate(
    evaluate_languages: list = ["english", "chinese"], # Languages need to be evaluated
    model: str = "deepseek-chat",                      # Model name
    model_type: str = "llm",                           # llm or mllm or default
    api_config: dict = None,             # api config, include temperature, max tokens, etc
    game_name: dict = None,                            # Game Name needed to be evaluated
    country: dict = None,                              # Countries needed to be evaluated
    repeated_num: int = 10,                            # Number of repeated tests with a selected action space
    redundancy: int = 0,                               # Number of redundancy tests
    batch_size: int = 4,                               # Number of tasks in one batch, related to LLM API access limitation
    TPM_sleep: int = 3,                                # Retry how many times when TPM or RPM is exceeded
) -> list:
    """Evaluate diplomatic game with different languages and different Observation Spaces."""
    total_task_info = []

    # Evaluate Fantasy Game Discounts:
    for test_language in tqdm(evaluate_languages, desc="Outer Loop"): # with different language
        print("test_language: {}".format(test_language))
        temp_task_info = await DGC_evaluate(
            evaluated_language = test_language, # Language of Evaluated prompt
            model = model,                      # Model name
            model_type = model_type,            # "mllm" or "llm" or "default", use "llm" defaultly
            api_config = api_config,            # api config, include temperature, max tokens, etc
            game_name = game_name,              # Game Name needed to be evaluated
            country = country,                  # Countries needed to be evaluated
            repeated_num = repeated_num,        # Number of repeated tests with a selected action space
            redundancy = redundancy,            # Number of redundancy tests
            batch_size = batch_size,            # Number of tasks in one batch, related to LLM API access limitation
            TPM_sleep = TPM_sleep,              # Retry how many times when TPM or RPM is exceeded
        )
        total_task_info.append(temp_task_info)
        json_dump_path = './record/Diplomatic_Game_Choices_real_deepseekv3_raw-{}-temp-sf-2.json'.format(test_language)
        write_to_json(total_task_info, json_dump_path)
    return total_task_info

if __name__ == '__main__':
    # Model name:
    # "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # "meta-llama/Meta-Llama-3.1-70B-Instruct"
    # "meta-llama/Llama-3.3-70B-Instruct"
    # "qwen-vl-max"
    # "deepseek-chat"
    # "local"
    # deepseek-ai/DeepSeek-V3
    # deepseek-v3
    game_name = read_json('./DGC/diplomatic_game_name.json')
    max_num = 1
    game_name = {
        "english": game_name["english"][:max_num],
        "chinese": game_name["chinese"][:max_num],
        "arabic": game_name["arabic"][:max_num],
    }
    
    country = read_json('./DGC/diplomatic_game_country.json')
    country_num = 10 # 25
    country = {
        "english": country["english"][:country_num],
        "chinese": country["chinese"][:country_num],
        "arabic": country["arabic"][:country_num],
    }
    # country = country_selection(country, target_num = 10)

    # evaluate_language = ["english", "chinese", "arabic"]
    evaluated_languages = ["chinese"]
    
    api_config = {
        "max_tokens": 4096, # 4096 or 65536 or 8192
        "temperature": 1.0,
        "top_p": 0.7,
        # "top_k": 50,
    }

    print("game_name: {}".format(game_name))
    print("country: {}".format(country))

    total_task_info = asyncio.run(
        multi_evaluate(
            evaluate_languages = evaluated_languages, # Language of Evaluated prompt
            model = "deepseek-ai/DeepSeek-V3",                  # Model name
            model_type = "llm",                       # "mllm" or "llm" or "default", use "llm" defaultly
            api_config = api_config,                  # api config, include temperature, max tokens, etc
            game_name = game_name,                    # Game Name needed to be evaluated
            country = country,                        # Countries needed to be evaluated
            repeated_num = 5,                         # Number of repeated tests with a selected action space
            redundancy = 1,                           # Number of redundancy tests
            batch_size = 10,                          # Number of tasks in one batch, related to LLM API access limitation
            TPM_sleep = 5,                            # Retry how many times when TPM or RPM is exceeded
        )
    )
    
    json_dump_path = './record/Diplomatic_Game_Choices_real_deepseek_v3_raw-sf-2.json'
    write_to_json(total_task_info, json_dump_path)
