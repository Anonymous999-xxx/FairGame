import asyncio
import os
import re
import json
import time
from tqdm import tqdm
import random
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

def select_elements_randomly(elements_list: list, ratio: float) -> list:
    """
    Select elements from multiple lists based on a given ratio, while keeping the elements synchronized.
    
    Args:
        elements_list (list): A list of lists, where each sublist contains elements to be shuffled.
        ratio (float): The ratio of elements to select from each list.
    
    Returns:
        list: A list of lists, where each sublist contains the selected elements in the same order.
    """
    # Check if all sublists have the same length
    if not all(len(sublist) == len(elements_list[0]) for sublist in elements_list):
        '''
        for sublist in elements_list:
            if len(sublist) != len(elements_list[0]):
                print("================================")
                print(sublist)
                print("================================")
                print(elements_list[0])
                print("================================")
                break
        '''
        raise ValueError("All sublists must have the same length.")
    
    # Calculate the number of elements to select
    num_elements = len(elements_list[0])
    num_selected = int(num_elements * ratio)
    
    # Shuffle the indices
    shuffled_indices = random.sample(range(num_elements), num_selected)
    
    # Select elements from each sublist based on the shuffled indices
    selected_elements = []
    for sublist in elements_list:
        selected_elements.append([sublist[i] for i in shuffled_indices])
    
    return selected_elements

def select_elements_with_truncation(elements: list, truncation: int) -> list:
    """
    Select elements from the top of each sublist.
    
    Args:
        elements (list): A list of lists, where each sublist contains elements to be truncated.
        truncation (int): The number of elements to select from the top of each sublist.
    
    Returns:
        list: A list of lists, where each sublist contains the selected elements from the top.
    """
    selected_elements = []
    for sublist in elements:
        selected_elements.append(sublist[:truncation])
    
    return selected_elements

def shuffle_elements(elements_list: list) -> list:
    """
    Shuffle elements in multiple lists while keeping the elements synchronized.
    
    Args:
        elements_list (list): A list of lists, where each sublist contains elements to be shuffled.
    
    Returns:
        list: A list of lists, where each sublist contains the shuffled elements in the same order.
    """
    # Check if all sublists have the same length
    if not all(len(sublist) == len(elements_list[0]) for sublist in elements_list):
        raise ValueError("All sublists must have the same length.")
    
    # Shuffle the indices
    num_elements = len(elements_list[0])
    shuffled_indices = list(range(num_elements))
    random.shuffle(shuffled_indices)
    
    # Shuffle elements in each sublist based on the shuffled indices
    shuffled_elements = []
    for sublist in elements_list:
        shuffled_elements.append([sublist[i] for i in shuffled_indices])
    
    return shuffled_elements


def get_selected_action_space(action_space: dict, para: (float|int), mode: str = "all", lang: str = None) -> dict:
    """
    Get the action space by selecting elements based on a ratio and optionally shuffling the result.
    
    Args:
        action_space (dict): The original action space dictionary.
        para (float): The ratio of elements to select (for "randomly" mode) or the truncation number (for "truncation" mode).
        mode (str): The selection mode, either "randomly" or "truncation" or "all".
        shuffle (bool): Whether to shuffle the selected elements after selection.
    
    Returns:
        dict: A dictionary containing the selected and optionally shuffled action space.
    """
    # Select elements based on the mode
    if mode == "randomly":
        [selected_item] = select_elements_randomly([action_space[lang]], ratio=para)
    elif mode == "truncation":
        [selected_item] = select_elements_with_truncation([action_space[lang]], truncation=para)
    elif mode == "all":
        selected_item = action_space[lang]
    else:
        raise ValueError("Invalid mode. Must be 'randomly' or 'truncation'.")
    
    # Update the selected action space
    selected_action_space = {}
    selected_action_space["label"] = len(selected_item) * [0]
    selected_action_space[lang] = selected_item
    return selected_action_space

def shuffle_action_space(selected_action_space: dict, shuffle: bool = True, lang: str = None) -> dict:
    # Optionally shuffle the selected elements
    label = selected_action_space["label"]
    item = selected_action_space[lang]
    if shuffle:
        temp_label, temp_item = shuffle_elements([label, item])
    
    # Return selected elements with shuffled elements
    shuffled_action_space = {
        "label": temp_label,
        lang: '\n'.join(temp_item),
    }
    return shuffled_action_space
    
def construct_bar_shop_qurey(
    original_prompt: str, 
    game_name: str,
    flavor: str,
    shop_type: str,
    location: str,
    item_type: str,
    item_list: str,
    item_num: int,
) -> str:
    """Replace some words to construct a new qurey_prompt."""
    item_list += '\n'

    modified_content = original_prompt.replace('[game_name]', game_name) \
                                      .replace('[flavor]', flavor) \
                                      .replace('[shop_type]', shop_type) \
                                      .replace('[location]', location) \
                                      .replace('[item_type]', item_type) \
                                      .replace('[item_list]', item_list) \
                                      .replace('[item_num]', str(item_num))

    return modified_content

def reshape_result_list(input_list, l1_num, l2_num, l3_num, l4_num, repeated_num):
    # First, check if the length of the input list meets expectations:
    if len(input_list) != l1_num * l2_num * l3_num * l4_num * repeated_num:
        raise ValueError("The input list's length is incompatible with the expected 4D shape.")
    
    # Initialize a 5D list
    reshaped_list = []
    
    # Starting reconstruction:
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

def check_extracted(extracted_items: list, check_list: list, index: int):
    check_list_lower = [t.lower() for t in check_list]
    # Filter out the outputs that comply with the regulations:
    target_length = len(extracted_items)
    temp = []
    for y in extracted_items:
        if y.lower() in check_list_lower:
            temp.append(y)
    if len(temp) == target_length:
        return temp
    else: # The output does not comply with the regulations
        print(colored(f"An output didn't follow the instruction. Expect {target_length} items, get {len(temp)} items.", "yellow"))
        print("check[evaluated_language]:{}".format(check_list))
        print("extracted_items: {}".format(extracted_items))
        print("index: {}".format(index))
        return None

def filter_results(repeated_num: int, redundancy: int, results: list):
    """
    This function's purpose: 
    For every (repeated_num + redundancy) results, select the first repeated_num results.

    results contain n * (repeated_num + redundancy) elements, where n is an integer
    """
    # Filter successfully processed responses:
    # repeated_num -= redundancy
    new_results = []
    # Iterate through results, grouping elements in sets of (repeated_num + redundancy):
    for i in range(0, len(results), repeated_num + redundancy):
        # Get elements of current group:
        valid_results = []
        group = results[i:i + repeated_num + redundancy]
        for e in group:
            if e != None:
                valid_results.append(e)
        # Check if there are enough valid values:
        if len(valid_results) < repeated_num:
            print("Not enough valid numbers in group starting at sample {}.\nExpected {}, found {}.".format(
                i/(repeated_num + redundancy), repeated_num, len(valid_results)
            ))
            return False, None
        else:
            # Take the first repeated_num valid values:
            new_results.extend(valid_results[:repeated_num])
    return True, new_results

def filter_action_num(record: list = None, minnum: int = 1, maxnum: int = 10, item_pool: list = None):
    # Initialize a 5D list
    reshaped_list = []
    print("l1_num: {}".format(len(record)))
    print("l2_num: {}".format(len(record[0])))
    print("l3_num: {}".format(len(record[0][0])))
    print("l4_num: {}".format(len(record[0][0][0])))
    print("repeated_num: {}".format(len(record[0][0][0][0])))
    print("Item Num: {}".format(len(record[0][0][0][0][0])))
    
    # Starting reconstruction:
    for i in range(len(record)):
        l1_subclasses = []
        for j in range(len(record[i])):
            l2_subclasses = []
            for k in range(len(record[i][j])):
                l3_subclasses = []
                for l in range(len(record[i][j][k])):
                    l4_subclasses = []
                    # print("action num: {}".format(l+1))
                    for r in range(len(record[i][j][k][l])):
                        diff_num = (l+1) - len(record[i][j][k][l][r])
                        if len(record[i][j][k][l][r]) >= (l+1):
                            l4_subclasses.append(record[i][j][k][l][r][:l+1])
                        elif diff_num > 0:
                            # print(f"diff = {diff_num}")
                            temp = record[i][j][k][l][r] + random.sample(item_pool, diff_num)
                            # print("expect {}, actual {}, get {}".format((l+1), len(record[i][j][k][l][r]), len(temp)))
                            l4_subclasses.append(temp)
                            # print(temp)
                        else:
                            l4_subclasses.append(record[i][j][k][l][r][:l+1])
                    l3_subclasses.append(l4_subclasses)
                l2_subclasses.append(l3_subclasses)
            l1_subclasses.append(l2_subclasses)
        reshaped_list.append(l1_subclasses)
    
    return reshaped_list

async def GGS_evaluate(
    evaluated_language: str = "english", # Language of Evaluated prompt
    model_name_for_file_name: str = "deepseekv3", # Model name for file name
    model: str = "deepseek-chat",        # Model name
    model_type: str = "llm",             # llm or mllm or default, use "llm" defaultly
    data_type: str = "virtual",          # "real" or "virtual"
    api_config: dict = None,             # api config, include temperature, max tokens, etc
    game_name: dict = None,              # Game Name needed to be evaluated
    shop_type: str = None,               # Shop Type needed to be evaluated
    shop_type_in_text: dict = None,      # Shop Type in text
    flavor: dict = None,                 # Flavor needed to be evaluated
    location: dict = None,               # Location needed to be evaluated
    eval_selection: str = None,          # Evaluate what, "flavor" or "location" or "flavor_N_location"
    item_type: str = None,               # Item Type needed to be evaluated
    item: dict = None,                   # Items needed to be evaluated
    select_mode: str = "all",            # How to select action space, "randomly" or "truncation" or "all"
    select_ratio: float = 1.0,           # Ratio of selected actions, "randomly" mode only
    select_truncation: int = 10,         # Num of selected actions , "truncation" mode only
    min_item_num: int = 10,              # Min num of selected actions
    max_item_num: int = 20,              # Max num of selected actions
    repeated_num: int = 10,              # Number of repeated tests of a selected action space
    redundancy: int = 0,                 # Number of redundancy tests
    batch_size: int = 4,                 # Number of tasks in one batch, related to LLM API access limitation
    TPM_sleep: int = 3,                  # Sleep time when TPM is exceeded
) -> str:
    """evaluate game shop allocation bias."""
    filter_dict = {"model": model}
    llm_config = config_list_from_json(env_or_file="../APIconfigure/configure_list.json", filter_dict=filter_dict)[0]
    print("llm_config: \n{}\n{}".format(llm_config, api_config))
    
    # Observation Space:
    if "english" == evaluated_language:
        game_shop_allocation = read_txt("./prompt/en_GGS_prompt.txt")
        shop_type_in_text = shop_type_in_text["english"][0]
        game_name = game_name["english"]
        # flavor = flavor["english"]
        # location = location["english"]
    elif "chinese" == evaluated_language:
        game_shop_allocation = read_txt("./prompt/ch_GGS_prompt.txt")
        shop_type_in_text = shop_type_in_text["chinese"][0]
        game_name = game_name["chinese"]
        # flavor = flavor["chinese"]
        # location = location["chinese"]
    elif "arabic" == evaluated_language:
        game_shop_allocation = ""
        shop_type_in_text = shop_type_in_text["arabic"][0]
        game_name = game_name["arabic"]
        # flavor = flavor["arabic"]
        # location = location["arabic"]
    else:
        print(colored("Wrong Language Selection!", "red"))
    
    flavor = flavor[""]
    location = location[""]

    # Action Space:
    selected_action_space = get_selected_action_space(
        item, 
        para = select_truncation,   # select_ratio or truncation
        mode = select_mode,         # "randomly" or "truncation"
        lang = evaluated_language   # "english" or "chinese" or "arabic"
    ) # Select a specified number of items according to certain rules, without shuffling
    # print("Selected Action Space:\n{}".format(selected_action_space))

    check = selected_action_space.copy()
    check[evaluated_language] = [t.split(" - ")[0] for t in check[evaluated_language]]
    # print("check: \n{}".format(check))

    # Generate Asyn Tasks:
    task_info = {
        "evaluated_language": evaluated_language,
        "game_name": game_name,
        "shop_type": shop_type,
        "shop_type_in_text": shop_type_in_text,
        "flavor": flavor,
        "location": location,
        "item_type": item_type,
        "select_mode": select_mode,
        "select_ratio": select_ratio,
        "select_truncation": select_truncation,
        "min_item_num": min_item_num,
        "max_item_num": max_item_num,
        "repeated_num": repeated_num,
        "redundancy": redundancy,
        "selected_action_space": selected_action_space, 
    }
    repeated_num += redundancy

    tasks = []
    for i in range(len(game_name)):
        for j in range(len(location)):
            for k in range(len(flavor)):
                for l in range(min_item_num, max_item_num + 1):
                    # Shuffle the selected action space and evaluate
                    for r in range(repeated_num):
                        shuffled_action_space = shuffle_action_space(
                            selected_action_space, shuffle = True, lang = evaluated_language
                        )
                        # print("Shuffled Action Space:\n{}".format(shuffled_action_space))
        
                        # Prepare qurey_prompt:
                        qurey_prompt = construct_bar_shop_qurey(
                            original_prompt = game_shop_allocation, 
                            game_name = game_name[i],
                            flavor = "",
                            shop_type = shop_type_in_text,
                            location = "",
                            item_type = item_type,
                            item_list = shuffled_action_space[evaluated_language],
                            item_num = l,
                        )
                        # print("qurey:\n{}\n".format(qurey_prompt))

                        if model_type == "llm" or model_type == "mllm" or model_type == "default":
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant"},
                                {"role": "user", "content": qurey_prompt},
                            ]
                        else:
                            # print(colored("\nLOOK FOR RELEVANT MEMOS, AS DEMAND-RESPONSE PAIRS", "light_yellow"))
                            print(colored("Use wrong model!", "red"))
                        
                        # Create async task information:
                        task_params = {  # Stores task parameters
                            "task_id": i * (len(location) * len(flavor) * (max_item_num - min_item_num + 1) * repeated_num)
                                + j * (len(flavor) * (max_item_num - min_item_num + 1) * repeated_num)
                                + k * ((max_item_num - min_item_num + 1) * repeated_num)
                                + (l-1) * repeated_num
                                + (r+1),
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
                        # Get the last matched content
                        content =  match[-1].strip()
                        # Split the content into a list by commas:
                        if evaluated_language == "english":
                            extracted = [item.strip() for item in content.split(',')]
                        elif evaluated_language == "chinese":
                            extracted = [item.strip() for item in content.split('，')]
                        elif evaluated_language == "arabic":
                            extracted = [item.strip() for item in content.split('،')]
                        else:
                            extracted = None
                            print(colored("Wrong Language Selection!", "red"))
                        
                        # Filter outputs that meet format requirements:
                        extracted = check_extracted(extracted, check[evaluated_language], r_i)
                        temp_record.append(extracted)
                        # print("---------extracted selection:---------")
                        # print(record)
                        # print("--------------------------------------")
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
            successfal_flag, temp_record = filter_results(repeated_num, redundancy, results=temp_record)

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
        
        json_dump_path = './record/GGS_{}-bar-{}_raw-{}-temp.json'.format(data_type, model_name_for_file_name, evaluated_language)
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
    temp = reshape_result_list(
        input_list = record, 
        l1_num = len(game_name), 
        l2_num = len(location), 
        l3_num = len(flavor), 
        l4_num = (max_item_num - min_item_num + 1), 
        repeated_num =repeated_num
    )
    task_info["record"] = filter_action_num(temp, min_item_num, max_item_num, check[evaluated_language])

    # Print the results:
    print(colored(
        "Info of results collected:\n"
        "Evaluated Language: {}\n"
        "Game Name: {}\n"
        "Shop Types: {}\n"
        "Flavor: {}\n"
        "Location: {}\n"
        "Item Type: {}\n"
        "Length of selected_action_space: {}\n".format(
            evaluated_language, 
            game_name,
            shop_type, 
            flavor, 
            location,
            item_type,
            len(task_info["selected_action_space"][evaluated_language])
        ), 
        "yellow"
    ))

    return task_info

async def multi_evaluate(
    evaluate_languages: list = ["english", "chinese"], # Language of Evaluated prompt
    model_name_for_file_name: str = "deepseekv3",      # Model name for file name
    model: str = "deepseek-chat",                      # Model name
    data_type: str = "virtual",                        # "real" or "virtual"
    model_type: str = "llm",                           # "mllm" or "llm" or "default", use "llm" defaultly
    api_config: dict = None,                           # api config, include temperature, max tokens, etc
    game_name: dict = None,                            # Game Name needed to be evaluated
    shop_type: str = None,                             # Shop Type needed to be evaluated
    shop_type_in_text: dict = None,                    # Shop Type in text
    flavor: dict = None,                               # Flavor needed to be evaluated
    location: dict = None,                             # Location needed to be evaluated
    item_type: str = None,                             # Item Type needed to be evaluated
    item: dict = None,                                 # Items needed to be evaluated
    select_mode: str = "all",                          # How to select action space, "randomly" or "truncation" or "all"
    select_ratio: float = 1.0,                         # Ratio of selected actions, "randomly" mode only
    select_truncation: int = 5,                        # Num of selected actions , "truncation" mode only
    min_item_num: int = 10,                            # Min num of selected actions
    max_item_num: int = 20,                            # Max num of selected actions
    repeated_num: int = 10,                            # Number of repeated tests of a selected action space
    redundancy: int = 0,                               # Number of redundancy tests
    batch_size: int = 100,                             # Number of tasks in one batch, related to LLM API access limitation
    TPM_sleep: int = 3,                                # Sleep time when TPM is exceeded
) -> list:
    """Evaluate game shop allocation with different languages and different Observation Spaces."""
    total_task_info = []

    # Evaluate Fantasy Game Discounts:
    for test_language in tqdm(evaluate_languages, desc="Loop", leave=False): # with different language
        print("test_language: {}".format(test_language))
        temp_task_info = await GGS_evaluate(
            evaluated_language = test_language,    # Language of Evaluated prompt
            model_name_for_file_name = model_name_for_file_name, # Model name for file name
            model = model,                         # Model name
            model_type = model_type,               # "mllm" or "llm" or "default", use "llm" defaultly
            data_type = data_type,                 # "real" or "virtual"
            api_config = api_config,               # api config, include temperature, max tokens, etc
            game_name = game_name,                 # Game Name needed to be evaluated
            shop_type = shop_type,                 # Shop Type needed to be evaluated
            shop_type_in_text = shop_type_in_text, # Shop Type in text
            flavor = flavor,                       # Flavor needed to be evaluated
            location = location,                   # Location needed to be evaluated
            item_type = item_type,                 # Item Type needed to be evaluated
            item = item,                           # Items needed to be evaluated
            select_mode = select_mode,             # How to select action space, "randomly" or "truncation" or "all"
            select_ratio = select_ratio,           # Ratio of selected actions, "randomly" mode only
            select_truncation = select_truncation, # Num of selected actions, "truncation" mode only
            min_item_num = min_item_num,           # Min num of selected actions
            max_item_num = max_item_num,           # Max num of selected actions
            repeated_num = repeated_num,           # Number of repeated tests of a selected action space
            redundancy = redundancy,               # Number of redundancy tests
            batch_size = batch_size,               # Number of tasks in one batch, related to LLM API access limitation
            TPM_sleep = TPM_sleep,                 # Sleep time when TPM is exceeded
        )
        total_task_info.append(temp_task_info)
        json_dump_path = './record/GGS_{}-bar-{}_raw-{}-temp.json'.format(data_type, model_name_for_file_name, test_language)
        write_to_json(total_task_info, json_dump_path)
    return total_task_info

if __name__ == '__main__':
    # Parse command-line arguments, see args_config.py and api_model_name.json for more details
    # See APIconfigure/configure_list.json for your API keys setting
    args = parse_args()
    print(f"Model Name: {args.model_name}")
    print(f"Data Type: {args.data_type}")

    api_model_name = read_json('../api_model_name.json')
    game_name = read_json('./data/game_name-GGS.json')
    game_name = game_name[args.data_type]
    
    max_num = 10 # total num: 10
    game_name = {
        "english": game_name["english"][:max_num],
        "chinese": game_name["chinese"][:max_num],
        # "arabic": game_name["arabic"][:max_num],
    }
    
    shop_type = "bar"
    shop_type_all = read_json('./data/shop_type-GGS.json')
    shop_type_in_text = shop_type_all[shop_type]

    if args.data_type == "virtual":
        item = read_json('./data/item_GOF2-Virtual.json')
        item_type = "wine"
        item = {
            "english": item[item_type]["english"][:-1],
            "chinese": item[item_type]["chinese"][:-1],
        } # remove last wine (task item)
    
        flavor = {
            "english": [""],
            "chinese": [""],
        }

        location = {
            "english": [""],
            "chinese": [""],
        }
    else:
        raise ValueError("This program is designed for virtual data only. You should change the data type to 'virtual'.")  

    # evaluate_language = ["english", "chinese", "arabic"] # "arabic" is not supported yet
    evaluated_languages = ["english", "chinese"]

    api_config = {
        "max_tokens": 4096, # 4096 or 65536 or 8192
        "temperature": 1.0,
        "top_p": 0.7,
        # "top_k": 50,
    }
    
    print("game_name: {}".format(game_name))
    print("flavor: {}".format(flavor))
    print("location: {}".format(location))
    
    total_task_info = asyncio.run(
        multi_evaluate(
            evaluate_languages = evaluated_languages,   # Language of Evaluated prompt
            model_name_for_file_name = args.model_name, # Model name for saving file
            model = api_model_name[args.model_name],    # Model name for API
            model_type = "llm",                         # "mllm" or "llm" or "default", use "llm" defaultly
            data_type = args.data_type,                 # "real" or "virtual"
            api_config = api_config,                    # api config, include temperature, max tokens, etc
            game_name = game_name,                      # Game Name needed to be evaluated
            shop_type = shop_type,                      # Shop Type needed to be evaluated
            shop_type_in_text = shop_type_in_text,      # Shop Type in text
            flavor = flavor,                            # Flavor needed to be evaluated
            location = location,                        # Location needed to be evaluated
            item_type = item_type,                      # Item Type needed to be evaluated
            item = item,                                # Items needed to be evaluated
            select_mode = "all",                        # How to select action space, "randomly" or "truncation" or "all"
            select_ratio = 0.5,                         # Ratio of selected actions, "randomly" mode only
            select_truncation = 10,                     # Num of selected actions , "truncation" mode only
            min_item_num = 1,                           # Min num of selected actions
            max_item_num = 10,                          # Max num of selected actions
            repeated_num = 10,                           # Number of repeated tests of a selected action space
            redundancy = 2,                             # Number of redundancy tests
            batch_size = 10,                            # Number of tasks in one batch, related to LLM API access limitation
            TPM_sleep = 5,                              # Sleep time when TPM is exceeded
        )
    )
    
    json_dump_path = './record/GGS_{}-bar-{}_raw-final.json'.format(args.data_type, args.model_name)
    write_to_json(total_task_info, json_dump_path)
