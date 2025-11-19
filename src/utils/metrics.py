import re
from collections import Counter

def normalize_text(text: str):
    """Lowercase + hapus tanda baca dan spasi ganda."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = " ".join(text.split())
    return text

def exact_match(pred: str, gold: str):
    return int(normalize_text(pred) == normalize_text(gold))

def f1_score(pred: str, gold: str):
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    common = pred_counter & gold_counter
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def rouge_l(pred: str, gold: str):
    """ROUGE-L sangat sederhana—LCS."""
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    # LCS Matrix
    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if pred_tokens[i] == gold_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    lcs = dp[m][n]
    return lcs / max(1, len(gold_tokens))
