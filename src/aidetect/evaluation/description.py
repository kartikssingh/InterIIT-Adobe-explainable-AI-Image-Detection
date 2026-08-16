"""Description-quality metrics.

ROUGE-L and BLEU are implemented in pure Python (no ``nltk``/``rouge-score``
dependency, no NLTK data download) so evaluation works offline. Reference-free
diagnostics — length distribution, lexical diversity, artifact grounding and
template repetition — are usually more informative here, since the competition
has no single gold description.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------- #
# Reference-based
# --------------------------------------------------------------------------- #


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence (O(len(a) * len(b)) time, O(len(b)) space)."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b):
            if token_a == token_b:
                current.append(previous[index] + 1)
            else:
                current.append(max(current[index], previous[index + 1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str, beta: float = 1.2) -> Dict[str, float]:
    """ROUGE-L precision / recall / F-measure."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = lcs_length(pred_tokens, ref_tokens)
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    if precision == 0.0 or recall == 0.0:
        return {"precision": precision, "recall": recall, "f1": 0.0}

    beta_sq = beta**2
    f1 = ((1 + beta_sq) * precision * recall) / (recall + beta_sq * precision)
    return {"precision": precision, "recall": recall, "f1": f1}


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    """Sentence-level BLEU with the standard brevity penalty."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        pred_ngrams = _ngrams(pred_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)
        total = sum(pred_ngrams.values())
        if total == 0:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, ref_ngrams[gram]) for gram, count in pred_ngrams.items())
        # Add-one smoothing keeps short sentences from collapsing to zero.
        precisions.append((overlap + 1.0) / (total + 1.0))

    log_mean = sum(math.log(p) for p in precisions if p > 0) / max(len(precisions), 1)
    brevity = min(1.0, math.exp(1.0 - len(ref_tokens) / max(len(pred_tokens), 1)))
    return brevity * math.exp(log_mean)


# --------------------------------------------------------------------------- #
# Reference-free
# --------------------------------------------------------------------------- #


def distinct_n(texts: Iterable[str], n: int = 2) -> float:
    """Fraction of unique n-grams across a corpus — low values mean templated output."""
    total = 0
    unique = set()
    for text in texts:
        tokens = tokenize(text)
        for index in range(len(tokens) - n + 1):
            gram = tuple(tokens[index : index + n])
            unique.add(gram)
            total += 1
    return len(unique) / total if total else 0.0


def stem(token: str) -> str:
    """Crude suffix stripper.

    Enough to make 'shadows'/'shadow', 'directions'/'direction' and
    'smoothness'/'smooth' match, without pulling in a stemming library.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 6 and token.endswith("ness"):
        return token[:-4]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


_GROUNDING_STOPWORDS = {
    "in", "of", "the", "or", "and", "a", "an", "within", "between", "with",
    "into", "from", "that", "this", "single", "same", "certain", "some",
}


def artifact_grounding(description: str, descriptors: Sequence[str]) -> float:
    """How much of the detected artifact vocabulary the description actually uses.

    Matching is stem-based, so a description saying "shadows" counts as covering
    the descriptor "Inconsistent shadow directions".
    """
    if not descriptors:
        return 0.0
    text_tokens = {stem(t) for t in tokenize(description)}
    covered = 0
    for descriptor in descriptors:
        keywords = {
            stem(t)
            for t in tokenize(descriptor)
            if t not in _GROUNDING_STOPWORDS and len(t) > 3
        }
        if keywords and len(keywords & text_tokens) / len(keywords) >= 0.34:
            covered += 1
    return covered / len(descriptors)


def length_stats(texts: Sequence[str]) -> Dict[str, float]:
    counts = [len(tokenize(text)) for text in texts if text]
    if not counts:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    ordered = sorted(counts)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "mean": sum(counts) / len(counts),
        "min": float(min(counts)),
        "max": float(max(counts)),
        "median": float(median),
    }


def evaluate_descriptions(
    predictions: Sequence[str],
    references: Sequence[str] = (),
    descriptor_lists: Sequence[Sequence[str]] = (),
    max_words: int = 50,
) -> Dict[str, object]:
    """Aggregate description quality across a run."""
    report: Dict[str, object] = {
        "count": len(predictions),
        "length": length_stats(predictions),
        "distinct_1": distinct_n(predictions, 1),
        "distinct_2": distinct_n(predictions, 2),
        "over_word_budget": sum(1 for p in predictions if len(tokenize(p)) > max_words),
        "empty": sum(1 for p in predictions if not p.strip()),
    }

    if descriptor_lists:
        scores = [
            artifact_grounding(pred, descriptors)
            for pred, descriptors in zip(predictions, descriptor_lists)
        ]
        report["artifact_grounding"] = sum(scores) / len(scores) if scores else 0.0

    if references:
        pairs = list(zip(predictions, references))
        rouge_scores = [rouge_l(pred, ref)["f1"] for pred, ref in pairs]
        bleu_scores = [bleu(pred, ref) for pred, ref in pairs]
        report["rouge_l_f1"] = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
        report["bleu"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    return report
