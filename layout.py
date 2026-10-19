"""layout.py — 文本布局与"注视 → 词"映射
================================================================
刺激布局假设（与本实验一致，可在界面上调整）:
    第 i 个字符占据像素区间 [LEFT + i*W, LEFT + (i+1)*W)
    默认 LEFT = 600, W = 50

流程
    句子 → jieba 分词 → 每个词的字符区间 → 像素区间
    注视点 (x, y) 落入某词区间即记为对该词的一次注视
    按标准定义计算 FFD / GD / TT / 跳读
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import jieba
import pandas as pd

from asc_io import Fixation

PUNCT = set("，。！？；：、,.!?;:“”‘’（）()《》<>—…· \t\n\r")


@dataclass
class Word:
    idx: int
    text: str
    char_start: int
    x_left: float
    x_right: float


def split_sentences(text: str) -> List[str]:
    """按换行与句末标点切句，保留句子内容。"""
    out, buf = [], []
    for ch in text:
        buf.append(ch)
        if ch in "\n。！？!?":
            s = "".join(buf).strip()
            if s:
                out.append(s)
            buf = []
    s = "".join(buf).strip()
    if s:
        out.append(s)
    return out or [text.strip()]


def build_words(sentence: str, left_margin: float = 600.0,
                char_width: float = 50.0) -> List[Word]:
    words: List[Word] = []
    pos = 0
    idx = 0
    for tok in jieba.cut(sentence):
        start = pos
        pos += len(tok)
        if not tok or all(c in PUNCT for c in tok):
            continue
        words.append(Word(idx=idx, text=tok, char_start=start,
                          x_left=left_margin + start * char_width,
                          x_right=left_margin + pos * char_width))
        idx += 1
    return words


def assign(fixations: Sequence[Fixation], words: Sequence[Word]) -> List[List[Fixation]]:
    assigned: List[List[Fixation]] = [[] for _ in words]
    for f in fixations:
        for w in words:
            if w.x_left <= f.x < w.x_right:
                assigned[w.idx].append(f)
                break
    return assigned


def word_metrics(words: Sequence[Word],
                 assigned: Sequence[Sequence[Fixation]]) -> pd.DataFrame:
    rows = []
    for w, fxs in zip(words, assigned):
        fxs = sorted(fxs, key=lambda f: f.sttime)
        if not fxs:
            rows.append(dict(idx=w.idx, word=w.text, char_start=w.char_start,
                             n_fix=0, ffd=0.0, gd=0.0, tt=0.0, skipping=1))
            continue
        # 首次通过：从第一次注视起，连续停留该词的注视段
        first = fxs[0]
        firstpass = [first]
        for f in fxs[1:]:
            if f.sttime - firstpass[-1].entime > 1:
                break
            firstpass.append(f)
        rows.append(dict(
            idx=w.idx, word=w.text, char_start=w.char_start, n_fix=len(fxs),
            ffd=float(first.duration),
            gd=float(sum(f.duration for f in firstpass)),
            tt=float(sum(f.duration for f in fxs)),
            skipping=0,
        ))
    return pd.DataFrame(rows)


def analyze_trial(fixations: Sequence[Fixation], sentence: str,
                  left_margin: float = 600.0,
                  char_width: float = 50.0) -> tuple[pd.DataFrame, List[Word]]:
    words = build_words(sentence, left_margin, char_width)
    assigned = assign(fixations, words)
    return word_metrics(words, assigned), words


def trial_summary(metrics: pd.DataFrame, total_time_ms: float) -> dict:
    n_words = len(metrics)
    n_read = int((metrics.skipping == 0).sum())
    read = metrics[metrics.skipping == 0]
    return {
        "n_words": n_words,
        "n_words_fixated": n_read,
        "skipping_rate": float(1 - n_read / n_words) if n_words else 0.0,
        "mean_gd_ms": float(read.gd.mean()) if len(read) else 0.0,
        "median_gd_ms": float(read.gd.median()) if len(read) else 0.0,
        "mean_ffd_ms": float(read.ffd.mean()) if len(read) else 0.0,
        "total_time_ms": float(total_time_ms),
        "reading_speed_wps": float(n_words / (total_time_ms / 1000)) if total_time_ms > 0 else 0.0,
    }
