import json
import argparse
from collections import defaultdict

from Modules.llms_evaluator import LLMsEvaluator





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--model_family", type=str, choices=['flan-t5', 'gpt-text', 'gpt-chat', 'claude', 'llama2'], required=True)
    parser.add_argument("--mode", default='paper', type=str, choices=['paper', 'leaderboard'])

    args = parser.parse_args()

    # read in prediction result
    with open(args.file_path, 'r') as f:
        prediction_dic = json.load(f)

    # prepare for evaluation
    evaluator = LLMsEvaluator(args)
    correct_num = 0
    correct_by_relation = defaultdict(int)
    answer_distribution = {'A': 0, 'B': 0}


    # calculate metrics
    for key in prediction_dic.keys():
        item = prediction_dic[key]
        ground_truth = item['answer']
        prediction = item['prediction']

        answer_distribution[ground_truth] += 1

        log_probs = None if item['logprobs'] == [] else item['logprobs']

        result = evaluator.evaluate(ground_truth=ground_truth, prediction=prediction, log_probs=log_probs)

        if result:
            correct_num += 1
            correct_by_relation[item['relation']] += 1

    print("Correct number of each relation")
    for key in correct_by_relation.keys():
        print(f"{key}: {correct_by_relation[key]}")

    print(f"Answer distribution: {answer_distribution}")
    print(f"Correct Number: {correct_num}, Total Number: {len(prediction_dic)}")
    print(f"Overall Accuracy: {round(correct_num / len(prediction_dic), 4)}")


