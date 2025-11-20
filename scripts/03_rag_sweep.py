from rapidfireai.experiment import Experiment
from rapidfireai.rag import RAGConfig

exp = Experiment(project="rag-sweep")
sweep = [
    RAGConfig(
        corpus_dir="data/corpora",
        chunk_size=512, chunk_overlap=64,
        retriever="bm25", top_k=10,
        prompt_preset="baseline",
    ),
    RAGConfig(
        corpus_dir="data/corpora",
        chunk_size=1024, chunk_overlap=128,
        retriever="faiss", top_k=20,
        prompt_preset="cot",
    ),
]
exp.run(sweep)  # parallel multi-config

