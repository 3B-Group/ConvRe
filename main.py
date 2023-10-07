import os
import json
import time
import argparse

import tqdm

from utils.data_processor import ConvReProcessor
from utils.llms_interface import LanguageModelInterface


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task", type=str, choices=['re2text', 'text2re'], required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument("--prompt", type=str, choices=['normal', 'hint', 'hint+cot'], required=True)
    parser.add_argument("--relation", type=str, choices=['normal', 'converse'], required=True)
    parser.add_argument("--n_shot", type=int, choices=[0, 1, 2, 3, 4, 5, 6], required=True)
    parser.add_argument("--example_type", type=str, choices=['hard', 'regular'], required=True)
    parser.add_argument("--text_type", type=str, choices=['regular', 'hard'], required=True)

    # Default arguments
    parser.add_argument("--device", default="cuda", type=str, choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--use_subset', action='store_true')
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens", default=512, type=int)

    args = parser.parse_args()

    # construct queries
    processor = ConvReProcessor(args)
    query_dic = processor.construct_prompt()

    # load model
    llms = LanguageModelInterface(args)
    result_dic = {}

    pbar = tqdm.tqdm(range(len(query_dic)))
    for i in pbar:
        item = query_dic[i]
        while True:
            flag = False
            try:
                model_answer = llms.completion(item['query'])
                flag = True
            except:
                print("API Error occurred, wait for 3 seconds. \
                      If you are using GPT or Cluade, this is normal. \
                      However, if you are running local models like llama2-chat, \
                      it's advisable to initiate debugging to identify the issue.")
                time.sleep(3)

            if flag == True:
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
    with open(f"Results-{args.task.upper()}/{args.model_name.split('/')[-1]}_{args.prompt}_{args.relation}_{args.n_shot}_{args.example_type}_{args.text_type}{'_subset' if args.use_subset else ''}.json", 'w') as f:
        json_str = json.dumps(result_dic, indent=2)
        f.write(json_str)
