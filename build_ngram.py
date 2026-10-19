"""build_ngram.py — 用 GPT-2 蒸馏字级 n-gram，供静态网页近似计算 surprisal
================================================================
思路
    网页端无法运行 GPT-2。这里用 GPT-2 对项目内可用的中文语料逐字计算
    surprisal，再统计成「字符上下文 → 平均 surprisal」的查表模型：
        trigram(前2字+本字) → bigram(前1字+本字) → unigram(本字)
    前端按同样回退顺序查表，得到近似 surprisal。

验证
    按句子 80/20 划分，在留出句上比较 n-gram 近似与 GPT-2 真值的
    Pearson r 与 MAE。

输出
    docs/data/ngram.json
    docs/data/scoring.json   (Ridge 参数，供前端算难度分)
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import difficulty  # noqa: E402

BASE = HERE.parent
DOCS = HERE / "docs"
DATA = DOCS / "data"
DATA.mkdir(parents=True, exist_ok=True)

SEED = 42
MIN_TRIGRAM_COUNT = 2
MIN_BIGRAM_COUNT = 2


# ---------------------------------------------------------------------------
# 语料
# ---------------------------------------------------------------------------
def collect_corpus() -> list[str]:
    sents: list[str] = []

    # 1) 实验材料（300 句 × 4 条件）
    xlsx = BASE / "csy.xlsx"
    if xlsx.exists():
        for sheet in ("formal", "prac"):
            try:
                df = pd.read_excel(xlsx, sheet_name=sheet)
            except Exception:
                continue
            need = {"preceding", "lf", "hf", "invalid", "following"}
            if not need.issubset(df.columns):
                continue
            for _, r in df.iterrows():
                for col in ("lf", "hf", "invalid"):
                    if pd.isna(r.get(col)):
                        continue
                    s = f"{r['preceding']}{r[col]}{r['following']}"
                    if isinstance(s, str):
                        sents.append(s)

    # 2) 问卷文本
    for p in (BASE / "materials" / "questionaire1.txt",
              BASE / "materials" / "questionaire2.txt"):
        if p.exists():
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if len(line) >= 8:
                    sents.append(line)

    # 3) 项目内的说明性中文文本（只取中文段落）
    for p in (BASE / "extract.m", BASE / "makeExpSlide.m"):
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for m in re.findall(r"[\u4e00-\u9fff，。！？；：、]{10,}", txt):
                sents.append(m)

    # 去重、只保留含中文的句子
    out, seen = [], set()
    for s in sents:
        s = re.sub(r"\s+", "", s)
        if not re.search(r"[\u4e00-\u9fff]", s) or len(s) < 6:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def collect_vocab(sents: list[str]) -> list[str]:
    """从实验材料的词表提取词典，供前端最大匹配分词。"""
    words = set()
    try:
        em = pd.read_pickle(BASE / "python_analysis" / "full_v3" / "cache" / "em_word.pkl")
        words.update(em.wordtext.dropna().astype(str).tolist())
    except Exception:
        pass
    for s in sents:
        words.update(re.findall(r"[\u4e00-\u9fff]{2,4}", s))
    words = {w for w in words if 1 <= len(w) <= 6 and re.fullmatch(r"[\u4e00-\u9fff]+", w)}
    return sorted(words)


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def accumulate(sents: list[str]) -> tuple[dict, dict, dict, list]:
    uni = defaultdict(lambda: [0.0, 0])
    bi = defaultdict(lambda: [0.0, 0])
    tri = defaultdict(lambda: [0.0, 0])
    samples = []   # (sentence, surprisal array) 用于评估

    for i, s in enumerate(sents):
        surp, _ent, _ok = difficulty.char_level_stats(s)
        for j, ch in enumerate(s):
            if not re.match(r"[\u4e00-\u9fff]", ch):
                continue
            v = float(surp[j])
            uni[ch][0] += v; uni[ch][1] += 1
            if j >= 1:
                k = s[j - 1] + ch
                bi[k][0] += v; bi[k][1] += 1
            if j >= 2:
                k = s[j - 2] + s[j - 1] + ch
                tri[k][0] += v; tri[k][1] += 1
        samples.append((s, surp))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sents)} sentences ...", flush=True)
    return uni, bi, tri, samples


def lookup(s, j, uni, bi, tri, default=8.0) -> float:
    ch = s[j]
    if j >= 2:
        k = s[j - 2] + s[j - 1] + ch
        if k in tri:
            return tri[k][0] / tri[k][1]
    if j >= 1:
        k = s[j - 1] + ch
        if k in bi:
            return bi[k][0] / bi[k][1]
    if ch in uni:
        return uni[ch][0] / uni[ch][1]
    return default


def evaluate(samples, uni, bi, tri) -> dict:
    pred, true = [], []
    for s, surp in samples:
        for j, ch in enumerate(s):
            if not re.match(r"[\u4e00-\u9fff]", ch):
                continue
            pred.append(lookup(s, j, uni, bi, tri))
            true.append(float(surp[j]))
    pred, true = np.array(pred), np.array(true)
    return {
        "n_chars": int(len(pred)),
        "pearson_r": float(np.corrcoef(pred, true)[0, 1]),
        "mae": float(np.mean(np.abs(pred - true))),
        "true_mean": float(true.mean()),
        "pred_mean": float(pred.mean()),
    }


def main() -> None:
    print("collecting corpus ...")
    sents = collect_corpus()
    print(f"  {len(sents)} sentences, {sum(len(s) for s in sents)} chars")

    random.seed(SEED)
    idx = list(range(len(sents)))
    random.shuffle(idx)
    cut = int(len(idx) * 0.8)
    train = [sents[i] for i in idx[:cut]]
    test = [sents[i] for i in idx[cut:]]
    print(f"  train {len(train)} / test {len(test)}")

    print("distilling with GPT-2 ...")
    uni, bi, tri, train_samples = accumulate(train)
    _, _, _, test_samples = accumulate(test)

    metrics = evaluate(test_samples, uni, bi, tri)
    print("\n--- held-out approximation quality ---")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # 导出：只保留出现次数达阈值的上下文，控制体积
    uni_out = {k: round(v[0] / v[1], 3) for k, v in uni.items() if v[1] >= 1}
    bi_out = {k: round(v[0] / v[1], 3) for k, v in bi.items() if v[1] >= MIN_BIGRAM_COUNT}
    tri_out = {k: round(v[0] / v[1], 3) for k, v in tri.items() if v[1] >= MIN_TRIGRAM_COUNT}
    default = float(np.mean([v[0] / v[1] for v in uni.values()])) if uni else 8.0

    ngram = {
        "unigram": uni_out,
        "bigram": bi_out,
        "trigram": tri_out,
        "default_surprisal": round(default, 3),
        "metrics": metrics,
        "note": "将 trigram/bigram/unigram 回退，近似 GPT-2 字级 surprisal",
    }
    with open(DATA / "ngram.json", "w", encoding="utf-8") as f:
        json.dump(ngram, f, ensure_ascii=False, separators=(",", ":"))
    print(f"\nngram.json: {(DATA/'ngram.json').stat().st_size/1024/1024:.2f} MB "
          f"(uni {len(uni_out)}, bi {len(bi_out)}, tri {len(tri_out)})")

    # Ridge 参数与词表
    art = difficulty.load_difficulty_model()
    scoring = {
        "features": art["features"],
        "scaler_mean": [round(x, 6) for x in art["scaler_mean"]],
        "scaler_scale": [round(x, 6) for x in art["scaler_scale"]],
        "coef": [round(x, 6) for x in art["ridge_coef"]],
        "intercept": round(float(art["ridge_intercept"]), 6),
        "score_lo": round(float(art["score_lo"]), 6),
        "score_hi": round(float(art["score_hi"]), 6),
    }
    with open(DATA / "scoring.json", "w", encoding="utf-8") as f:
        json.dump(scoring, f, ensure_ascii=False, separators=(",", ":"))

    vocab = collect_vocab(sents)
    with open(DATA / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, separators=(",", ":"))
    print(f"scoring.json + vocab.json ({len(vocab)} words)")


if __name__ == "__main__":
    main()
