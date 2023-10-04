import os
import json


class ConvReProcessor:
    def __init__(self, args):
        self.args = args
        self.prompt = self.load_json_file(f"{self.args.task}_relations.json")
        self.examples = self.load_json_file(f"{self.args.task}_examples.json")
        self.datasets = self.load_json_file(f"triple_{'dataset' if self.args.use_subset == False else 'subset'}.json")

    def load_json_file(self, filename: str) -> dict:
        with open(os.path.join(self.args.data_dir, filename), 'r') as f:
            contents = json.load(f)
        return contents

    def construct_prompt(self) -> dict:
        prompt_dic = {}

        # Convert dataset dictionary into list
        triple_list = []
        for key in self.datasets.keys():
            triple_list += self.datasets[key]

        # Construct prompt
        for triple_index in range(len(triple_list)):
            current_dic = {}

            # read current instance
            current_triple = triple_list[triple_index]
            current_dic['answer'] = current_triple['answer']
            current_dic['head'] = current_triple['head']
            current_dic['relation'] = current_triple['relation']
            current_dic['tail'] = current_triple['tail']

            # read few shot examples
            few_shot_examples = ""
            if self.args.n_shot != 0:
                if 'cot' in self.args.prompt:
                    example_list = self.examples[f"{self.args.example_type}-cot"]
                else:
                    example_list = self.examples[self.args.example_type]

                for i in range(self.args.n_shot):
                    few_shot_examples += '\n' + example_list[i]

            few_shot_examples += "\n"

            # Read instruction
            relation_mapping = f"Instruction: {self.prompt[current_triple['relation']][self.args.relation]}"

            # Get question
            if self.args.task == 're2text':
                question = f"Question: (?, {current_triple['relation'].split(',')[-1].strip()}, {current_triple['tail']})"
            else:
                question = f"Question: {self.prompt[current_triple['relation']][self.args.text_type].replace('[N]', current_triple['tail'])}"

            # Get choice A and choice B
            if self.args.task == 're2text':
                wrong_choice_text_type = 'regular' if self.args.text_type == 'hard' else 'hard'
                wrong_choice_relation_type = 'converse' if self.args.relation == 'normal' else 'normal'
                correct_choice = self.prompt[current_triple['relation']][f"{self.args.relation}-{self.args.text_type}"].replace('[N]', current_triple['tail'])
                wrong_choice = self.prompt[current_triple['relation']][f"{wrong_choice_relation_type}-{wrong_choice_text_type}"].replace('[N]', current_triple['tail'])
            else:
                correct_choice = self.prompt[current_triple['relation']][f"{self.args.relation}-correct"].replace('[N]', current_triple['tail'])
                wrong_choice = self.prompt[current_triple['relation']][f"{self.args.relation}-wrong"].replace('[N]', current_triple['tail'])

            if current_triple['answer'] == 'A':
                choice_a = correct_choice
                choice_b = wrong_choice
            else:
                choice_a = wrong_choice
                choice_b = correct_choice

            # Get hint
            if 'hint' in self.args.prompt:
                hint = " Note that in this task, if the relation is defined in a converse manner, unlike the conventional definition, you should carefully choose the answer."
                hint_remind = "Look out for the ORDER of the entities in the instruction!"
            else:
                hint = ""
                hint_remind = ''

            cot = " Your answer should be in JSON format with the following keys: thought, answer." if 'cot' in self.args.prompt else ""

            # construct query
            if self.args.task == 're2text':
                query = f"Read the instruction and then answer the question using A or B.{hint}{cot}\n{few_shot_examples}{relation_mapping}\n{question}\nA: {choice_a}\nB: {choice_b}\nTo convert the question into a semantically equivalent natural language sentence, which choice is correct? {hint_remind}\nAnswer:"
            else:
                query = f"Read the instruction and then answer the question using A or B.{hint}{cot}\n{few_shot_examples}{relation_mapping}\n{question}\nA: {choice_a}\nB: {choice_b}\nTo convert the question into a semantically equivalent triple query, which choice is correct? {hint_remind}\nAnswer:"
            current_dic['query'] = query

            prompt_dic[triple_index] = current_dic
        return prompt_dic
