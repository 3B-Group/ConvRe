import os
import json
import time
import argparse

import tqdm

from datasets import load_dataset
from utils.llms_interface import LanguageModelInterface


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task", type=str, choices=['re2text', 'text2re'], required=True)
    parser.add_argument("--setting", type=str, required=True)

    # Default parameters
    parser.add_argument("--device", default="cuda", type=str, choices=['cpu', 'cuda', 'mps'])
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens", default=512, type=int)

    args = parser.parse_args()

    # load dataset from huggingface
    hf_dataset = load_dataset('3B-Group/ConvRe', f"en-{args.task}", token=True, split=args.setting)

    # load model
    llms = LanguageModelInterface(args)
    result_dic = {}

    pbar = tqdm.tqdm(range(len(hf_dataset)))
    for i in pbar:
        item = hf_dataset[i]
        while 1:
            flag = 0
            try:
                model_answer = llms.completion(item['query'])
                flag = 1
            except:
                print("API Error occurred, wait for 3 seconds")
                time.sleep(3)

            if flag == 1:
                break
        answer_dic = item.copy()
        answer_dic['prompt'] = model_answer.prompt_text
        answer_dic['prompt_info'] = model_answer.prompt_info
        answer_dic['prediction'] = model_answer.response_text
        answer_dic['logprobs'] = model_answer.logprobs

        result_dic[str(i)] = answer_dic

        # show current progress
        prediction = answer_dic['prediction'][:11].replace('\n', ' ')
        pbar.set_description(f"ground_truth: {answer_dic['answer']}, prediction: {prediction}")

    # Save result
    if not os.path.exists(f"Results-{args.task.upper()}"):
        os.mkdir(f"Results-{args.task.upper()}")
    with open(f"Results-{args.task.upper()}/{args.model_name.split('/')[-1]}_{args.setting}.json", 'w') as f:
        json_str = json.dumps(result_dic, indent=2)
        f.write(json_str)
