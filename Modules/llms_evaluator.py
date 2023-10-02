import os
import abc


class BaseEvaluator(abc.ABC):
    """ A LLMs evaluator """

    @abc.abstractmethod
    def evaluate(self, ground_truth: str, prediction: str, log_probs=None) -> bool:
        raise NotImplementedError("Override me!")

    @staticmethod
    def evaluate_cot(ground_truth: str, prediction: str, mode: bool) -> bool:
        if mode == 'paper':
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


class ClaudeEvaluator(BaseEvaluator):
    def __init__(self, mode):
        self.mode = mode

    def evaluate(self, ground_truth: str, prediction: str, log_probs=None) -> bool:
        prediction = prediction.split('\n\n')[0]

        if self.mode == 'paper':
            assert prediction in [' A', ' B']
        else:
            if prediction not in [' A', ' B']:
                return False

        result = True if prediction[1] == ground_truth else False
        return result


class FlanT5Evaluator(BaseEvaluator):
    def __init__(self, mode):
        self.mode = mode

    def evaluate(self, ground_truth: str, prediction: str, log_probs=None) -> bool:
        if self.mode == 'paper':
            assert prediction in ['<pad> A</s>', '<pad> B</s>']
        else:
            if prediction not in ['<pad> A</s>', '<pad> B</s>']:
                return False

        result = True if prediction[6] == ground_truth else False
        return result


class LLMsEvaluator:
    model_mapping = {
        'flan-t5': FlanT5Evaluator,
        'claude': ClaudeEvaluator,
        'gpt-text': 1
    }

    def __init__(self, args):
        self.model_family = args.model_family
        # choose corresponding evaluators
        self.evaluator = self.model_mapping[self.model_family](args.mode)

    def evaluate(self, ground_truth: str, prediction: str, log_probs=None) -> bool:
        return self.evaluator.evaluate(ground_truth=ground_truth, prediction=prediction, log_probs=log_probs)