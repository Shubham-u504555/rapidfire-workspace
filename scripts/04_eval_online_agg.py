from rapidfireai.experiment import Experiment
from rapidfireai.evals import OnlineAggregationEval

exp = Experiment(project="evals-rag-sft")
cfg = OnlineAggregationEval(
    predictions_glob="runs/**/predictions_*.jsonl",
    metrics=["accuracy", "exact_match", "bleu"],
    group_by=["task", "model", "config_tag"]
)
exp.run(cfg)

