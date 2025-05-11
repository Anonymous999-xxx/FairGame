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
from ..args_config import parse_args
from ..utils import *
from ..llm_response import *

# Global Variables:
# Setting Chinese Font:
plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei as the font
plt.rcParams['axes.unicode_minus'] = False    # Resolve the negative sign display issue

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
    """Rate-limited version of an async request function"""
    # Acquire a token
    await token_bucket.acquire()

    # Send the request
    response = await get_response_async(
        messages, llm_config, model_type, 
        max_tokens = max_tokens, 
        temperature = temperature,
        top_p = top_p,
        # top_k = top_k
    )
    # Put the task ID and response into the queue
    await response_queue.put((task_id, response))

def construct_game_shop_qurey(
    original_prompt: str, 
    game_name: str,
    game_race: str,
    game_career: str,
    lower_bound: int,
    upper_bound: int,
) -> str:
    """Replace some words to construct a new qurey_prompt."""
    modified_content = original_prompt.replace('[GAME_NAME]', game_name) \
                                      .replace('[GAME_RACE]', game_race) \
                                      .replace('[GAME_CAREER]', game_career) \
                                      .replace('[LOWER_BOUND]', str(lower_bound)) \
                                      .replace('[UPPER_BOUND]', str(upper_bound))
    return modified_content


def reshape_result_list(input_list, l1_num, l2_num, l3_num, repeated_num):
    # First, check if the length of the input list meets expectations:
    if len(input_list) != l1_num * l2_num * l3_num * repeated_num:
        raise ValueError("The length of the input list does not match the expected 4D structure.")
    
    # Initialize a 4D list
    reshaped_list = []
    
    # Starting reconstruction:
    for i in range(l1_num):
        l1_subclasses = []
        for j in range(l2_num):
            l2_subclasses = []
            for k in range(l3_num):
                l3_subclasses = []
                for r in range(repeated_num):
                    index = i * (l2_num * l3_num * repeated_num) + j * (l3_num * repeated_num) + k * repeated_num + r
                    l3_subclasses.append(input_list[index])
                l2_subclasses.append(l3_subclasses)
            l1_subclasses.append(l2_subclasses)
        reshaped_list.append(l1_subclasses)
    
    return reshaped_list

def filter_results(repeated_num: int, redundancy: int, results: list):
    """
    This function's purpose: 
    For every (repeated_num + redundancy) results, select the first repeated_num results.

    results contain n * (repeated_num + redundancy) elements, where n is an integer
    """
    # Filter successfully processed responses:
    new_results = []
    
    # Iterate through results, grouping elements in sets of (repeated_num + redundancy):
    for i in range(0, len(results), repeated_num + redundancy):
        # Get elements of current group:
        group = results[i:i + repeated_num + redundancy]
    
        # Filter out valid outputs that meet requirements:
        valid_numbers = [abs(x) for x in group if isinstance(x, float)]

        # Check if there are enough valid values:
        if len(valid_numbers) < repeated_num:
            print("Not enough valid numbers in group starting at index {}. \nExpected {}, found {}.".format(
                i/(repeated_num + redundancy), repeated_num, len(valid_numbers)
            ))
            return False, None
    
        # Take the first repeated_num valid values:
        new_results.extend(valid_numbers[:repeated_num])
    return True, new_results

async def SNPC_evaluate(
    evaluated_language: str = "english", # "english", "chinese", "arabic"
    model_name_for_file_name: str = "deepseekv3", # Model name for file name
    model: str = "deepseek-chat",        # Model name
    model_type: str= "llm",              # llm or mllm
    data_type: str = "real",              # "real" or "virtual"
    api_config: dict = None,             # api config, include temperature, max tokens, etc
    game_name: dict= None,               # Game Name needed to be evaluated
    career: dict= None,                  # Career needed to be evaluated
    race: dict= None,                    # Race needed to be evaluated
    repeated_num: int = 10,              # Number of repeated tests with a selected action space
    redundancy: int = 0,                 # Number of redundancy tests
    batch_size : int = 10,               # Number of tasks in one batch, related to LLM API access limitation
    TPM_sleep: int = 3,                  # Retry how many times when TPM or RPM is exceeded
) -> dict:
    """Evaluate discount decision bias in SNPC Tasks."""
    filter_dict = {"model": model}
    llm_config = config_list_from_json(env_or_file="../APIconfigure/configure_list.json", filter_dict=filter_dict)[0]
    print("llm_config: \n{}".format(llm_config))

    # Observation Space:
    game_name = game_name[evaluated_language]
    race = race[evaluated_language]
    career = career[evaluated_language]
    lower_bound = 50
    upper_bound = 100

    if "english" == evaluated_language:
        discounts_prompt = read_txt("./prompt/en_SNPC_prompt.txt")
    elif "chinese" == evaluated_language:
        discounts_prompt = read_txt("./prompt/ch_SNPC_prompt.txt")
    elif "arabic" == evaluated_language:
        discounts_prompt = ""
    else:
        print(colored("Wrong Language Selection!", "red"))

    task_info = {
        "evaluated_language": evaluated_language,
        "game_name": game_name,
        "race": race,
        "career": career,
        "repeated_num": repeated_num,
        "redundancy": redundancy,
    }
    
    # Generate Asyn Tasks:
    repeated_num += redundancy
    tasks = []
    for gn in range(len(game_name)):
        for i in range(len(race)):
            for j in range(len(career)):
                # Prepare qurey_prompt:
                qurey_prompt = construct_game_shop_qurey(
                    original_prompt = discounts_prompt, 
                    game_name = game_name[gn],
                    game_race = race[i],
                    game_career = career[j],
                    lower_bound = lower_bound,
                    upper_bound = upper_bound
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
                    
                    # Create async task information:
                    task_params = {  # Stores task parameters
                        "task_id": gn * (len(race) * len(career) * repeated_num) 
                                + i * (len(career) * repeated_num) 
                                + j * repeated_num 
                                + r + 1,
                        "messages": messages,
                        "llm_config": llm_config,
                    }
                    tasks.append(task_params)  # Add task parameters to list called tasks 
    repeated_num -= redundancy
    
    # Execute tasks in batches
    print("Total Task Num: {}".format(len(tasks)))
    record = []
    first_call = True
    for b in range(0, len(tasks), batch_size * (repeated_num + redundancy)):
        start_time = time.time()
        real_batch_size = min(batch_size * (repeated_num + redundancy), len(tasks) - b)
        retry_count = 0
        while retry_count < TPM_sleep:  # Max retries: TPM_sleep times
            if not first_call:
                print("SLEEP FOR 61 SECONDS...")
                await asyncio.sleep(61)
                print("SLEEP OVER!")
                first_call = False
                
            temp_record = []

            # Recreate coroutine objects:
            batch = [
                limited_get_response_async(
                    task_id=tasks[b + rb]["task_id"],  # Get task_id from task parameters
                    messages=tasks[b + rb]["messages"],  # Get messages from task parameters
                    llm_config=tasks[b + rb]["llm_config"],  # Get llm_config from task parameters
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
                        content = match[-1].strip() # Get the last matched content
                        content = float(content[:-1].strip()) # Convert string to number

                        # Filter outputs that meet format requirements:
                        if content >= lower_bound and content <= upper_bound:
                            temp_record.append(content)  # Successfully processed, store content
                            # print("---------extracted selection:---------")
                            # print(content)
                            # print("--------------------------------------")
                        else:
                            print(colored(f"{responses[r_i][0]} Output Format Error!", "yellow"))
                            temp_record.append(None) # No content matched, store None
                            miss_match_count += 1
                            # print("content: {}\n".format(content))
                            # print("----------")
                            # print("response[{}]:\n{}".format(responses[r_i][0], responses[r_i][1]))
                            # print("----------")
                    else:
                        print(colored(f"{responses[r_i][0]} Miss Match!", "yellow"))
                        temp_record.append(None) # No content matched, store None
                        miss_match_count += 1
                        continue  # Skip current iteration, proceed to next response
                except ValueError as e:
                    print(colored(f"{responses[r_i][0]} ValueError", "red"))
                    temp_record.append(None) # No content matched, store None
                    miss_match_count += 1
                    continue  # Skip current iteration, proceed to next response
                except Exception as e:
                    print(colored(f"{responses[r_i][0]} Unexpected error", "red"))
                    temp_record.append(None) # No content matched, store None
                    miss_match_count += 1
                    continue  # Skip current iteration, proceed to next response
            
            # Filter successfully processed responses:
            successfal_flag, temp_record = filter_results(
                repeated_num, 
                redundancy, 
                results=temp_record
            )
            
            if miss_match_count >= real_batch_size or not successfal_flag:
                retry_count += 1
                if retry_count >= TPM_sleep:  # If retry count exceeds TPM_sleep times
                    raise RetryLimitExceededError("Retry limit exceeded! TPM limit may still be in effect. Exiting program.")
                print(colored("TPM Limit reached! Sleeping for an additional 61 seconds...", "yellow"))
                time.sleep(61)
                print("sleep over!")
                first_call = True
                continue  # Restart current iteration
            else:
                record += temp_record
                first_call = False
                print("This attempt succeeded!")
                break  # Exit retry loop, proceed to next batch
        
        json_dump_path = './record/SNPC_{}-{}_raw-{}-temp.json'.format(data_type, model_name_for_file_name, evaluated_language)
        write_to_json(record, json_dump_path)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Run Time of One Loop: {elapsed_time:.2f} s")
        print(f"One batch is done! batch_size: {len(responses)}\n")
        print("Continue Next Batch...")
    
    # print("++++++++++++++++++++++++++++++++++++++++")
    # print(f"Record_Num: \n{len(record)}\n")
    # print(f"Record: \n{record}\n")
    # print("++++++++++++++++++++++++++++++++++++++++")

    # Reshape and record the results:
    task_info["record"] = reshape_result_list(
        input_list = record, 
        l1_num = len(game_name),
        l2_num = len(race), 
        l3_num = len(career), 
        repeated_num =repeated_num
    )

    mes = colored(
        "Info of results collected:\n" +
        "Evaluated Language: {}\n".format(evaluated_language) +
        "Number of Game: {}\n".format(len(game_name)) +
        "Number of Race: {}\n".format(len(race)) +
        "Number of Career: {}".format(len(career)),
        "yellow"
    )
    print(mes)

    return task_info

async def multi_evaluate(
    evaluate_languages: list = ["english", "chinese"], # Languages need to be evaluated
    model_name_for_file_name: str = "deepseekv3",      # Model name for file name
    model: str = "deepseek-chat",                      # Model name
    model_type: str = "llm",                           # llm or mllm
    data_type:str = "real",                            # "real" or "virtual"
    api_config: dict = None,                           # api config, include temperature, max tokens, etc
    game_name: dict= None,                             # Game Name needed to be evaluated
    career: dict= None,                                # Career needed to be evaluated
    race: dict= None,                                  # Race needed to be evaluated
    repeated_num: int = 10,                            # Number of repeated tests with a selected action space
    redundancy: int = 0,                               # Number of redundancy tests
    batch_size : int = 10,                             # Number of tasks in one batch, related to LLM API access limitation
    TPM_sleep: int = 3,                                # Retry how many times when TPM or RPM is exceeded
) -> list:
    """Evaluate fantasy shop discounts with different languages and different Observation Spaces."""
    total_task_info = []

    # Evaluate Fantasy Game Discounts:
    for test_language in tqdm(evaluate_languages, desc="Loop", leave=False): # with different language
        print("test_language: {}".format(test_language))
        temp_task_info = await SNPC_evaluate(
            evaluated_language = test_language, # Language of Evaluated prompt
            model_name_for_file_name = model_name_for_file_name, # Model name for file name
            model = model,                      # Model name
            model_type = model_type,            # "mllm" or "llm"
            data_type = data_type,              # "real" or "virtual"
            api_config = api_config,            # api config, include temperature, max tokens, etc
            game_name = game_name,              # Game Name needed to be evaluated
            career = career,                    # Career needed to be evaluated
            race = race,                        # Race needed to be evaluated
            repeated_num = repeated_num,        # Number of repeated tests with a selected action space
            redundancy = redundancy,            # Number of redundancy tests
            batch_size = batch_size,            # Number of tasks in one batch, related to LLM API access limitation
            TPM_sleep = TPM_sleep,              # Sleep time when TPM is exceeded
        )
        total_task_info.append(temp_task_info)
        json_dump_path = './record/SNPC_{}-{}_raw-{}-temp.json'.format(data_type, model_name_for_file_name, test_language)
        write_to_json(total_task_info, json_dump_path)
    return total_task_info

if __name__ == '__main__':
    # Parse command-line arguments, see args_config.py and api_model_name.json for more details
    # See APIconfigure/configure_list.json for your API keys setting
    args = parse_args()
    print(f"Model Name: {args.model_name}")
    print(f"Data Type: {args.data_type}")

    api_model_name = read_json('../api_model_name.json')
    game_name = read_json('./data/game_name-SNPC.json')
    game_name = game_name[args.data_type]

    max_num = 10 # total num: 10
    game_name = {
        "english": game_name["english"][:max_num],
        "chinese": game_name["chinese"][:max_num],
        # "arabic": game_name["arabic"][:max_num],
    }

    race = read_json('./data/race-SNPC.json')
    career = read_json('./data/career-SNPC.json')

    # evaluate_language = ["english", "chinese", "arabic"] # "arabic" is not supported yet
    evaluated_languages = ["english", "chinese"]

    api_config = {
        "max_tokens": 4096, # 4096 or 65536 or 8192
        "temperature": 1.0,
        "top_p": 0.7,
        # "top_k": 50,
    }

    print("Game Name: {}".format(game_name))
    print("Race: {}".format(race))
    print("Career: {}".format(career))

    total_task_info = asyncio.run(
        multi_evaluate(
            evaluate_languages = evaluated_languages,   # Language of Evaluated prompt
            model_name_for_file_name = args.model_name, # Model name for saving file
            model = api_model_name[args.model_name],    # Model name for API
            model_type = "llm",                         # "mllm" or "llm"
            data_type = args.data_type,                 # "real" or "virtual"
            api_config = api_config,                    # api config, include temperature, max tokens, etc
            game_name = game_name,                      # Game Name needed to be evaluated
            career = career[args.data_type],            # Career needed to be evaluated
            race = race[args.data_type],                # Race needed to be evaluated
            repeated_num = 10,                           # Number of repeated tests with a selected action space
            redundancy = 1,                             # Number of redundancy tests
            batch_size = 15,                            # Number of tasks in one batch, related to LLM API access limitation
            TPM_sleep = 5,                              # Retry how many times when TPM or RPM is exceeded
        )
    )
    
    json_dump_path = './record/SNPC_{}-{}_raw-final.json'.format(args.data_type, args.model_name)
    write_to_json(total_task_info, json_dump_path)
