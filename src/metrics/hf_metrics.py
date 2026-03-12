import evaluate
import torch

# from qa_metrics import pedant


# class PEDANTS:
#     def __init__(self, name="PEDANTS/PANDA"):
#         self.name = name
#         self.metric = pedant.PEDANT()

#     def __call__(self, predictions, references, questions, **kwargs):
#         result = 0
#         for i in range(len(predictions)):
#             result += self.metric.get_score(
#                 reference=references[i], candidate=predictions[i], question=questions[i]
#             )
#         return {"score": result / len(predictions)}


class BLEU:
    def __init__(self, name="BLEU"):
        self.name = name
        self.metric = evaluate.load("bleu")

    def __call__(self, predictions, references, **kwargs):
        # hf wants list of lists of str
        references = [[ref] for ref in references]
        results = self.metric.compute(predictions=predictions, references=references)
        precisions = results.pop("precisions")
        for i in range(len(precisions)):
            results[f"precision{i + 1}"] = precisions[i]
        return results


class ROUGE:
    def __init__(self, name="ROUGE"):
        self.name = name
        self.metric = evaluate.load("rouge")

    def __call__(self, predictions, references, **kwargs):
        return self.metric.compute(predictions=predictions, references=references)


class METEOR:
    def __init__(self, name="METEOR"):
        self.name = name
        self.metric = evaluate.load("meteor")

    def __call__(self, predictions, references, **kwargs):
        return self.metric.compute(predictions=predictions, references=references)


# class BLEURT:
#     def __init__(self, name="BLEURT_20", checkpoint="BLEURT-20"):
#         self.name = name
#         self.metric = evaluate.load(
#             "bleurt", module_type="metric", config_name=checkpoint
#         )

#     def __call__(self, predictions, references, **kwargs):
#         scores = self.metric.compute(predictions=predictions, references=references)[
#             "scores"
#         ]
#         return {"score": torch.tensor(scores).mean().item()}


class BERTScore:
    def __init__(
        self,
        name="BERTScore_distil_base",
        lang="en",
        model_type="distilbert-base-uncased",
        batch_size=8,
        device=None,
    ):
        self.name = name
        self.metric = evaluate.load("bertscore")
        self.lang = lang
        self.model_type = model_type
        self.batch_size = batch_size
        self.device = device

    def __call__(self, predictions, references, **kwargs):
        results = self.metric.compute(
            predictions=predictions,
            references=references,
            lang=self.lang,
            model_type=self.model_type,
            batch_size=self.batch_size,
            device=self.device,
        )
        scores = {}
        for k in ["precision", "recall", "f1"]:
            scores[k] = torch.tensor(results[k]).mean().item()
        return scores
