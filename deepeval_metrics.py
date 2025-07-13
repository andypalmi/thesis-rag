from deepeval.metrics.dag import (
    DeepAcyclicGraph,
    TaskNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    VerdictNode,
)
from deepeval.metrics import (
    BaseMetric,
    AnswerRelevancyMetric,
    GEval,
    DAGMetric,
    FaithfulnessMetric,
    HallucinationMetric
)
from deepeval.test_case import LLMTestCaseParams
from typing import Optional, Union, Iterable, List, Dict
from deepeval.models.base_model import DeepEvalBaseLLM

def get_metrics(
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        thresholds: Union[float, Dict[str, float]] = 0.7,
        include_reason: bool = False
        ) -> List[BaseMetric]:
    """Returns a list of metrics for evaluating language model performance."""

    correctness_metric = GEval(
        name="Correctness",
        threshold=thresholds if isinstance(thresholds, float) else thresholds.get('Correctness', 0.9),
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        evaluation_steps=[
            "Check whether the actual output is either empty or explicitly states that the question cannot be answered with the given context. If the expected output also states that the question cannot be answered, assign the maximum score and skip further steps.",
            "If the expected output provides an answer, verify whether the facts in 'actual output' contradict any facts in 'expected output'.",
            "Heavily penalize omission of key details present in the expected output.",
            "Vague language or contradicting OPINIONS are acceptable and should not be penalized.",
            "If the expected output states that the question cannot be answered with the given context but the actual output attempts to provide an answer, apply a heavy penalty."
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT, 
            LLMTestCaseParams.ACTUAL_OUTPUT, 
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=model,
        _include_g_eval_suffix = False
    )
    specific_info_accuracy_metric = GEval(
        name="Specific Information Accuracy",
        threshold=thresholds if isinstance(thresholds, float) else thresholds.get('Specific Information Accuracy', 0.9),
        criteria=(
            "Evaluate whether the actual output appropriately responds to the input question given the context, "
            "without introducing specific information (e.g., names, places, numbers) that is not explicitly provided in the context. "
            "Use the expected output to determine whether the model should answer the question or state that it cannot answer."
        ),
        evaluation_steps=[
            "Carefully read the context and identify all specific information (such as names, places, numbers) explicitly mentioned.",
            "Review the expected output to understand whether the question is answerable given the context.",
            "Analyze the actual output to see if it includes specific information not present in the context.",
            "If the expected output indicates that the question cannot be answered:",
            "    - If the actual output correctly states that it cannot answer the question with the given context or provides an appropriate non-informative response, assign the highest possible score.",
            "    - If the actual output attempts to answer the question by introducing information not present in the context, assign the lowest possible score.",
            "If the expected output indicates that the question can be answered:",
            "    - If the actual output answers the question using only the information present in the context without adding any inferred or external information, assign a high score based on the answer's accuracy.",
            "    - If the actual output includes any specific information (names, places, numbers) that is not present in the context, assign a lower score accordingly.",
            "Provide a final score based on the above criteria, ensuring that the evaluation is consistent with the expected output."
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
        model=model,
        _include_g_eval_suffix = False
    )
    answer_relevancy = AnswerRelevancyMetric(
        threshold=thresholds if isinstance(thresholds, float) else thresholds.get('Answer Relevancy', 0.9),
        model=model,
        include_reason=include_reason
    )
    faithfulness_metric = FaithfulnessMetric(
        threshold=thresholds if isinstance(thresholds, float) else thresholds.get('Faithfulness', 0.9),
        model=model,
        include_reason=include_reason
    )
    hallucinationMetric = HallucinationMetric(
        threshold=thresholds if isinstance(thresholds, float) else thresholds.get('Hallucination', 0.8),
        model=model,
        include_reason=include_reason
    )
    """
        NOTE:
        DeepEval logs will show a different percentage of passed tests for Hallucination due to how this metric works.
        For Hallucination, lower scores -> better performances
        However we flip the score of this metric for easiness of comprehension of the results in the CSV and the 
        flipped threshold is not handled here.
        Just make sure to pass as threshold in the main the flipped threshold.
        e.g. If you want to only pass tests with a low Hallucination score (0.2 for example), then pass 1 - 0.2 = 0.8 in the main function.
    """
    metrics = [
        correctness_metric,
        specific_info_accuracy_metric,
        answer_relevancy,
        faithfulness_metric,
        hallucinationMetric
    ]
    return metrics