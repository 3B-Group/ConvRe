import os
import abc


class BaseEvaluator(abc.ABC):
    """ A LLMs evaluator """

    @abc.abstractmethod
    def evaluate(self, ground_truth: str, prediction: str, log_probs=None) -> bool:
        raise NotImplementedError("Override me!")

    @staticmethod
    def evaluate_cot(ground_truth: str, prediction: str, mode: bool) -> bool:
        if mode == 'strict':
            prediction = eval(prediction.split('\n\n')[0])['answer']
            assert prediction in ['A', 'B']
        else:
            try:
                prediction = eval(prediction.split('\n\n')[0])['answer']
                assert prediction in ['A', 'B']
            except:
                return False

        result = True if prediction == ground_truth else False
        return result


class GPTTextEvaluator(BaseEvaluator):
    def __init__(self, mode) -> None:
        self.mode = mode

    def evaluate(self, ground_truth: str, prediction: str, log_probs=None) -> bool:
        condition1 = prediction in [' A', ' B']
        # sometimes the model may repeat the two choices, which should be considered wrong answer
        condition2 = prediction[:3] in [' A:', ' B:'] and not ('A:' in prediction and 'B:' in prediction)
        condition3 = prediction[:3] in [' A.', ' B.'] and not ('A.' in prediction and 'B.' in prediction)

        if self.mode == 'strict':
            if not (condition1 or condition2 or condition3):
                # choose the answer according to log probs
                probs = log_probs[0]
                assert ' A' in probs or ' B' in probs

                prob_a = probs[' A'] if ' A' in probs else -1000
                prob_b = probs[' B'] if ' B' in probs else -1000

                if prob_a == prob_b:
                    raise ValueError("Please manually decide the answer of the model")
                prediction = ' A' if prob_a > prob_b else ' B'
        else:
            if not (condition1 or condition2 or condition3):
                probs = log_probs[0]
                if not (' A' in probs or ' B' in probs):
                    return False
                
                prob_a = probs[' A'] if ' A' in probs else -1000
                prob_b = probs[' B'] if ' B' in probs else -1000

                if prob_a == prob_b:
                    return False
                prediction = ' A' if prob_a > prob_b else ' B'

        prediction = prediction[1]

        result = True if prediction == ground_truth else False
        return result


class GPTChatEvaluator(BaseEvaluator):
    def __init__(self, mode) -> None:
        self.mode = mode

    def evaluate(self, ground_truth: str, prediction: str, log_probs=None) -> bool:
        condition1 = prediction in ['A', 'B']
        # sometimes the model may repeat the two choices, which should be considered wrong answer
        condition2 = prediction[:2] in ['A:', 'B:'] and not ('A:' in prediction and 'B:' in prediction)
        condition3 = prediction[:2] in ['A.', 'B.'] and not ('A.' in prediction and 'B.' in prediction)

        if self.mode == 'strict':
            assert (condition1 or condition2 or condition3)
        else:
            if not (condition1 or condition2 or condition3):
                return False

        prediction = prediction[0]

        result = True if prediction == ground_truth else False
        return result


class ClaudeEvaluator(BaseEvaluator):
    def __init__(self, mode) -> None:
        self.mode = mode

    def evaluate(self, ground_truth: str, prediction: str, log_probs=None) -> bool:
        prediction = prediction.split('\n\n')[0]

        if self.mode == 'strict':
            assert prediction in [' A', ' B']
        else:
            if prediction not in [' A', ' B']:
                return False

        result = True if prediction[1] == ground_truth else False
        return result


class FlanT5Evaluator(BaseEvaluator):
    def __init__(self, mode) -> None:
        self.mode = mode

    def evaluate(self, ground_truth: str, prediction: str, log_probs=None) -> bool:
        if self.mode == 'strict':
            assert prediction in ['<pad> A</s>', '<pad> B</s>']
        else:
            if prediction not in ['<pad> A</s>', '<pad> B</s>']:
                return False

        result = True if prediction[6] == ground_truth else False
        return result


class Llama2Evaluator(BaseEvaluator):
    def __init__(self, mode) -> None:
        self.mode = mode

    def evaluate(self, ground_truth: str, prediction: str, log_probs=None) -> bool:
        # Llama2's answer usually start with choices and then explanation.
        prediction = prediction.split('\n')[0]
        if self.mode == 'strict':
            assert prediction[:2] in [' A', ' B']
        else:
            if prediction[:2] not in [' A', ' B']:
                return False
            
        prediction = prediction[1]
        result = True if prediction == ground_truth else False
        
        return result
    
class QwenEvaluator(BaseEvaluator):
    def __init__(self, mode) -> None:
        self.mode = mode

    def evaluate(self, ground_truth: str, prediction: str, log_probs=None) -> bool:
        # qwen model may answer 'the correct choice is' instead of just answering A or B.
        if 'the correct choice is a' in prediction.lower():
            prediction = 'A'
        elif 'the correct choice is b' in prediction.lower():
            prediction = 'B'
        if self.mode == 'strict':
            assert prediction[0] in ['A', 'B']
        else:
            if prediction[0] not in ['A', 'B']:
                return False
        
        prediction = prediction[0]
        result = True if prediction == ground_truth else False
        return result

class InternlmEvaluator(BaseEvaluator):
    def __init__(self, mode) -> None:
        self.mode = mode

    def evaluate(self, ground_truth: str, prediction: str, log_probs=None) -> bool:
        if self.mode == 'strict':
            assert prediction in ['A', 'B']
        else:
            if prediction not in ['A', 'B']:
                return False

        result = True if prediction == ground_truth else False
        
        return result


class LLMsEvaluator:
    model_mapping = {
        'flan-t5': FlanT5Evaluator,
        'claude': ClaudeEvaluator,
        'gpt-text': GPTTextEvaluator,
        'gpt-chat': GPTChatEvaluator,
        'llama2': Llama2Evaluator,
        'qwen': QwenEvaluator,
        'internlm': InternlmEvaluator,
    }

    def __init__(self, args):
        self.model_family = args.model_family
        # choose corresponding evaluators
        self.evaluator = self.model_mapping[self.model_family](args.mode)

    def evaluate(self, ground_truth: str, prediction: str, log_probs=None) -> bool:
        return self.evaluator.evaluate(ground_truth=ground_truth, prediction=prediction, log_probs=log_probs)