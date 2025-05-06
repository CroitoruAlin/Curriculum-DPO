from typing import Iterable, List
from bert_score import BERTScorer
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

def load_bertscore():
    scorer = BERTScorer("microsoft/deberta-xlarge-mnli", use_fast_tokenizer=True)

    def compute_bertscore(
        candidates: Iterable[str], references: Iterable[str]
    ) -> np.ndarray:
        precision, recall, f1 = scorer.score(candidates, references)
        return precision.numpy(), recall.numpy(), f1.numpy()

    return compute_bertscore

def load_sentencetransformer():
    scorer = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def compute_bertscore(
        candidates: Iterable[str], references: Iterable[str]
    ) -> np.ndarray:
        # print(candidates)
        # print(references)
        embeddings_predictions = scorer.encode(candidates)
        embeddings_prompts = scorer.encode(references)

        cosine_similarities = []
        for embd_pred, embd_prompt in zip(embeddings_predictions, embeddings_prompts):
            cosine_similarities.append(
                1 - cosine(embd_pred, embd_prompt)
            )

        return np.array(cosine_similarities), np.array(cosine_similarities), np.array(cosine_similarities)

    return compute_bertscore
