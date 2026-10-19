"""difficulty.py — 文本难度：GPT-2 特征 + 难度模型推断
================================================================
特征与训练脚本保持一致：词级 surprisal、entropy、词长。
模型参数见 models/text_difficulty.pkl。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "text_difficulty.pkl"
LM_NAME = "uer/gpt2-chinese-cluecorpussmall"


@lru_cache(maxsize=1)
def load_lm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LM_NAME)
    model = AutoModelForCausalLM.from_pretrained(LM_NAME)
    model.eval()
    return tok, model


@lru_cache(maxsize=1)
def load_difficulty_model() -> dict:
    return pd.read_pickle(MODEL_PATH)


def char_level_stats(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (surprisal, entropy, char_to_token_ok)。按字符对齐。"""
    import torch

    tok, model = load_lm()
    # 不加 [CLS]/[SEP]，否则 token 与字符错位
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]
    with torch.no_grad():
        logits = model(**enc).logits[0]
    logp = torch.log_softmax(logits, dim=-1)
    probs = logp.exp()

    n = len(text)
    tok_texts = tok.convert_ids_to_tokens(ids.tolist())
    # 中文 GPT-2 通常 1 字 = 1 token；若不一致则按 token 长度展开
    surp_tok = np.zeros(len(ids))
    ent_tok = np.zeros(len(ids))
    for t in range(len(ids) - 1):
        surp_tok[t + 1] = float(-logp[t, ids[t + 1]])
        ent_tok[t + 1] = float(-(probs[t] * logp[t]).sum())

    expanded_s, expanded_e = [], []
    for tt, s, e in zip(tok_texts, surp_tok, ent_tok):
        k = max(1, len(tt.replace("Ġ", " ")))
        expanded_s.extend([s] * k)
        expanded_e.extend([e] * k)
    surp = np.array(expanded_s[:n] + [0.0] * max(0, n - len(expanded_s)))
    ent = np.array(expanded_e[:n] + [0.0] * max(0, n - len(expanded_e)))
    ok = np.ones(n, dtype=bool) if len(expanded_s) >= n else np.zeros(n, dtype=bool)
    return surp, ent, ok


def word_features(sentences: List[str], word_rows: List[dict]) -> pd.DataFrame:
    """为已分好的词计算 GPT-2 特征。

    word_rows: [{'sentence': str, 'word': str, 'char_start': int, 'length': int}]
    """
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    out = []
    for r in word_rows:
        s = r["sentence"]
        if s not in cache:
            cache[s] = char_level_stats(s)
        surp, ent, _ = cache[s]
        a, b = r["char_start"], r["char_start"] + r["length"]
        out.append({
            "t_surprisal": float(surp[a:b].sum()),
            "t_entropy": float(ent[a:b].mean()) if b > a else 0.0,
            "length": int(r["length"]),
        })
    return pd.DataFrame(out)


def predict_difficulty(feats: pd.DataFrame) -> np.ndarray:
    """返回 0-100 的难度分（越高越难）。"""
    m = load_difficulty_model()
    X = feats[m["features"]].values.astype(float)
    Xs = (X - np.array(m["scaler_mean"])) / np.array(m["scaler_scale"])
    yhat = Xs @ np.array(m["ridge_coef"]) + m["ridge_intercept"]
    lo, hi = m["score_lo"], m["score_hi"]
    score = (yhat - lo) / (hi - lo) * 100
    return np.clip(score, 0, 100)
