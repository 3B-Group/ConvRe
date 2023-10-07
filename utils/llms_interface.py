import os
import abc
from dataclasses import dataclass

import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig


@dataclass(frozen=True)
class LLMResponse:
    prompt_text: str
    response_text: str
    prompt_info: dict
    logprobs: list


class LanguageModels(abc.ABC):
    """ A pretrained Large language model"""

    @abc.abstractmethod
    def completion(self, prompt: str) -> LLMResponse:
        raise NotImplementedError("Override me!")


class GPTTextModel(LanguageModels):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_ORGANIZATION_ID = os.environ.get("OPENAI_ORGANIZATION_ID")

    def __init__(self, args) -> None:
        self.model_name = args.model_name
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens

        self.log_prob_num = 5

        openai.api_key = self.OPENAI_API_KEY
        openai.organization = self.OPENAI_ORGANIZATION_ID

    def completion(self, prompt: str) -> LLMResponse:
        response = openai.Completion.create(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            logprobs=self.log_prob_num
        )

        return self._raw_to_llm_response(response, prompt_text=prompt, max_tokens=self.max_tokens, temperature=self.temperature, log_prob_num=self.log_prob_num)

    @staticmethod
    def _raw_to_llm_response(model_response,
                             prompt_text: str,
                             max_tokens: int,
                             temperature: float,
                             log_prob_num: int) -> LLMResponse:
        answer = model_response['choices'][0].text
        logprobs = model_response['choices'][0]['logprobs']['top_logprobs']
        prompt_info = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'log_prob_num': log_prob_num
        }

        return LLMResponse(prompt_text=prompt_text, response_text=answer, prompt_info=prompt_info, logprobs=logprobs)


class GPTChatModel(LanguageModels):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_ORGANIZATION_ID = os.environ.get("OPENAI_ORGANIZATION_ID")

    def __init__(self, args):
        self.model_name = args.model_name
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens

        openai.api_key = self.OPENAI_API_KEY
        openai.organization = self.OPENAI_ORGANIZATION_ID

    def completion(self, prompt: str) -> LLMResponse:
        chat_message = [{"role": "system", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=chat_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return self._raw_to_llm_response(response, prompt_text=str(chat_message), max_tokens=self.max_tokens, temperature=self.temperature)

    @staticmethod
    def _raw_to_llm_response(model_response,
                             prompt_text: str,
                             max_tokens: int,
                             temperature: float) -> LLMResponse:
        answer = model_response['choices'][0]['message'].content
        prompt_info = {
            'max_tokens': max_tokens,
            'temperature': temperature
        }

        return LLMResponse(prompt_text=prompt_text, response_text=answer, prompt_info=prompt_info, logprobs=[])


class ClaudeModel(LanguageModels):
    CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")

    def __init__(self, args) -> None:
        self.model_name = args.model_name
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens

        self.anthropic = Anthropic(api_key=self.CLAUDE_API_KEY)

    def completion(self, prompt: str) -> LLMResponse:
        # the "Answer" is placed after AI_PROMPT to make the model's answer consistent with what we want
        response = self.anthropic.completions.create(
            model=self.model_name,
            max_tokens_to_sample=self.max_tokens,
            prompt=f"{HUMAN_PROMPT} {prompt[:-7]}{AI_PROMPT} Answer:",
            temperature=self.temperature,
        )

        return self._raw_to_llm_response(model_response=response, prompt_text=f"{HUMAN_PROMPT} {prompt[:-7]}{AI_PROMPT} Answer:", max_tokens=self.max_tokens, temperature=self.temperature)

    @staticmethod
    def _raw_to_llm_response(model_response,
                             prompt_text: str,
                             max_tokens: int,
                             temperature: float) -> LLMResponse:
        prompt_info = {
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        return LLMResponse(prompt_text=prompt_text, response_text=model_response.completion, prompt_info=prompt_info, logprobs=[])


class FlanT5Model(LanguageModels):

    def __init__(self, args) -> None:
        self.args = args
        self.max_tokens = args.max_tokens
        self.tokenizer = T5Tokenizer.from_pretrained(f"{args.model_name}")
        if self.args.device == 'cpu':
            self.model = T5ForConditionalGeneration.from_pretrained(f"{args.model_name}")
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(f"{args.model_name}", device_map="auto")

    def completion(self, prompt: str) -> LLMResponse:
        input_ids = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).input_ids.to(self.args.device)
        outputs = self.model.generate(input_ids, max_length=self.max_tokens)
        return self._raw_to_llm_response(model_response=self.tokenizer.decode(outputs[0]), prompt_text=prompt, max_tokens=self.max_tokens)

    @staticmethod
    def _raw_to_llm_response(model_response: str,
                             prompt_text: str,
                             max_tokens: int) -> LLMResponse:
        prompt_info = {
            'max_tokens': max_tokens
        }
        return LLMResponse(prompt_text=prompt_text, response_text=model_response, prompt_info=prompt_info, logprobs=[])


class Llama2Model(LanguageModels):
    sys_prompt = '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

'''

    def __init__(self, args) -> None:
        self.args = args
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.args.model_name}")
        if self.args.device == 'cpu':
            self.model = AutoModelForCausalLM.from_pretrained(f"{self.args.model_name}").eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(f"{self.args.model_name}", device_map="auto").eval()

    def completion(self, prompt: str) -> LLMResponse:
        prompt = self.sys_prompt + prompt[:-7] + " [/INST] Answer:"

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.args.device)
        if self.temperature == 0:
            outputs = self.model.generate(input_ids, max_length=self.max_tokens, do_sample=False)
        else:
            outputs = self.model.generate(input_ids, max_length=self.max_tokens, temperature=self.temperature)

        return self._raw_to_llm_response(model_response=self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].replace(prompt, ''), prompt_text=prompt, max_tokens=self.max_tokens, temperature=self.temperature)

    @staticmethod
    def _raw_to_llm_response(model_response: str,
                             prompt_text: str,
                             max_tokens: int,
                             temperature: float) -> LLMResponse:
        # implement later
        prompt_info = {
            'max_tokens': max_tokens,
            'temperature': temperature
        }

        return LLMResponse(prompt_text=prompt_text, response_text=model_response, prompt_info=prompt_info, logprobs=[])


class QWenModel(LanguageModels):
    def __init__(self, args) -> None:
        self.args = args
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True)
        if self.args.device == 'cpu':
            self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name, trust_remote_code=True).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name, device_map="auto", trust_remote_code=True).eval()

        self.model.generation_config = GenerationConfig.from_pretrained(self.args.model_name, trust_remote_code=True, )

    def completion(self, prompt: str) -> LLMResponse:
        if self.temperature == 0:
            response, history = self.model.chat(self.tokenizer, prompt, history=None, max_length=self.max_tokens, do_sample=False, max_new_tokens=None)
        else:
            response, history = self.model.chat(self.tokenizer, prompt, history=None, max_length=self.max_tokens, temperature=self.temperature, max_new_tokens=None)

        return self._raw_to_llm_response(model_response=response, prompt_text=prompt, max_tokens=self.max_tokens, temperature=self.temperature)

    @staticmethod
    def _raw_to_llm_response(model_response: str,
                             prompt_text: str,
                             max_tokens: int,
                             temperature: float) -> LLMResponse:
        prompt_info = {
            'max_tokens': max_tokens,
            "temperature": temperature
        }

        return LLMResponse(prompt_text=prompt_text, response_text=model_response, prompt_info=prompt_info, logprobs=[])


class InternlmModel(LanguageModels):
    def __init__(self, args) -> None:
        self.args = args
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, trust_remote_code=True)
        if self.args.device == 'cpu':
            self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name, trust_remote_code=True).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name, device_map="auto", trust_remote_code=True).eval()

    def completion(self, prompt: str) -> LLMResponse:
        if self.temperature == 0:
            reponse, history = self.model.chat(self.tokenizer, prompt, history=[], do_sample=False, max_length=self.max_tokens, max_new_tokens=None)
        else:
            reponse, history = self.model.chat(self.tokenizer, prompt, history=[], temperature=self.temperature, max_length=self.max_tokens, max_new_tokens=None)

        return self._raw_to_llm_response(model_response=reponse, prompt_text=prompt, max_tokens=self.max_tokens, temperature=self.temperature)

    @staticmethod
    def _raw_to_llm_response(model_response: str,
                             prompt_text: str,
                             max_tokens: int,
                             temperature: float) -> LLMResponse:
        prompt_info = {
            'max_tokens': max_tokens,
            "temperature": temperature
        }

        return LLMResponse(prompt_text=prompt_text, response_text=model_response, prompt_info=prompt_info, logprobs=[])


class LanguageModelInterface:
    model_family = {
        'gpt_text_models': ['text-ada-001', 'text-babbage-001', 'text-curie-001', 'text-davinci-002',
                            'text-davinci-003'],
        'gpt_chat_models': ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-0314', 'gpt-3.5-turbo-0301'],
        'claude_models': ['claude-1.3', 'claude-instant-1.1'],
        'flan_t5_models': ['flan-t5-small', 'flan-t5-base', 'flan-t5-large', 'flan-t5-xl', 'flan-t5-xxl'],
        'llama2_chat_models': ['llama-2-7b-chat-hf', 'llama-2-13b-chat-hf', 'llama-2-70b-chat-hf'],
        'qwen_models': ['qwen-7b-chat', 'qwen-14b-chat'],
        'internlm_models': ['internlm-chat-7b', 'internlm-chat-20b'],
    }

    model_mapping = {
        'gpt_text_models': GPTTextModel,
        'gpt_chat_models': GPTChatModel,
        'claude_models': ClaudeModel,
        'flan_t5_models': FlanT5Model,
        'llama2_chat_models': Llama2Model,
        'qwen_models': QWenModel,
        'internlm_models': InternlmModel,
    }

    def __init__(self, args) -> None:
        self.model_name = args.model_name
        name = self.model_name.split('/')[-1].lower()

        self.model_type = ''
        for key in self.model_family.keys():
            if name in self.model_family[key]:
                self.model_type = key
                break
        if self.model_type == '':
            raise ValueError(f"The interface for {self.model_name} is not implemented.")
        else:
            self.model = self.model_mapping[self.model_type](args)

    def completion(self, prompt: str) -> LLMResponse:
        return self.model.completion(prompt)
