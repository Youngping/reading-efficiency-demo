"""make_demo_data.py — 生成演示用的「任意文本 + 模拟 asc」
================================================================
用途
    静态演示版支持粘贴任意文本，但用户手头不一定有对应的眼动文件。
    这里生成一段非实验材料的文本，并按模型难度模拟一次阅读的眼动记录，
    便于完整走通"上传 asc → 词映射 → 指标 → 常模"的流程。

重要
    生成的 .asc 是**模拟数据**，仅用于演示界面与流程，不代表真实被试。
    文件名与文件头都标注 MOCK。

输出
    docs/data/sample_text.txt
    docs/data/sample_reading.asc
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import difficulty  # noqa: E402

BASE = HERE.parent
DATA = HERE / "docs" / "data"

SENTENCES = [
    "城市里的行道树不仅美化环境，还能降低夏季的地表温度。",
    "树冠遮挡阳光，叶片蒸腾水分，使周围空气变得湿润凉爽。",
]
LEFT_MARGIN = 600.0
CHAR_WIDTH = 50.0
Y = 720.0
SEED = 42


def load_vocab() -> set[str]:
    p = DATA / "vocab.json"
    return set(json.load(open(p, encoding="utf-8"))) if p.exists() else set()


def max_match(s: str, vocab: set[str], maxlen: int = 5) -> list[tuple[str, int]]:
    """与前端一致的前向最大匹配，返回 [(词, 字符起点)]。"""
    out, i = [], 0
    while i < len(s):
        hit = None
        for L in range(min(maxlen, len(s) - i), 0, -1):
            w = s[i:i + L]
            if L == 1 or w in vocab:
                hit = (w, i)
                i += L
                break
        if hit:
            if re.fullmatch(r"[\u4e00-\u9fff]+", hit[0]):
                out.append(hit)
    return out


def simulate_trial(text: str, words: list[tuple[str, int]], scores, rng) -> tuple[list[str], int]:
    """按难度模拟注视序列，返回 (asc 行, 结束时间)。"""
    lines, t = [], 0
    t += 300
    for (w, start), d in zip(words, scores):
        # 难度越高越不容易跳读
        skip_p = float(np.clip(0.62 - 0.0055 * d, 0.05, 0.65))
        if rng.random() < skip_p:
            continue
        x = LEFT_MARGIN + (start + len(w) / 2) * CHAR_WIDTH
        n_fix = 2 if (len(w) >= 3 and rng.random() < 0.28) else 1
        for _ in range(n_fix):
            dur = int(np.clip(rng.normal(150 + 2.4 * d, 42), 80, 520))
            lines.append(
                f"EFIX\tR\t{t}\t{t + dur}\t{dur}\t{x:.1f}\t{Y:.1f}\t{1000 + len(lines) % 500}")
            t += dur
            # 词内再注视之间加一个小间隔
            t += int(rng.integers(8, 30))
        t += int(rng.integers(25, 90))
    return lines, t


def main() -> None:
    rng = np.random.default_rng(SEED)
    vocab = load_vocab()

    out_lines = [
        "** CONVERTED FROM EDF **",
        "** MOCK DATA — 合成数据，仅用于演示，不代表真实被试 **",
        "**",
    ]
    t_clock = 1000
    for i, sent in enumerate(SENTENCES, 1):
        words = max_match(sent, vocab)
        rows = [{"sentence": sent, "word": w, "char_start": s, "length": len(w)}
                for w, s in words]
        feats = difficulty.word_features([sent], rows)
        scores = difficulty.predict_difficulty(feats)

        out_lines.append(f"MSG\t{t_clock} TRIALID DEMO{i}")
        out_lines.append(f"MSG\t{t_clock + 1} SENTENCEID DEMO{i}")
        out_lines.append(f"MSG\t{t_clock + 10} Sent MYKEYWORD 100")
        body, dur = simulate_trial(sent, words, scores, rng)
        base = t_clock + 20
        for line in body:
            parts = line.split("\t")
            parts[2] = str(int(parts[2]) + base)
            parts[3] = str(int(parts[3]) + base)
            out_lines.append("\t".join(parts))
        out_lines.append(f"MSG\t{base + dur + 100} Sent MYKEYWORD 200")
        t_clock = base + dur + 1500

        print(f"trial {i}: {len(words)} words, {len(body)} fixations, "
              f"difficulty {scores.mean():.1f}")

    (DATA / "sample_reading.asc").write_text("\n".join(out_lines) + "\n",
                                             encoding="utf-8")
    (DATA / "sample_text.txt").write_text("\n".join(SENTENCES), encoding="utf-8")
    print(f"saved: {DATA/'sample_reading.asc'} "
          f"({(DATA/'sample_reading.asc').stat().st_size/1024:.1f} KB)")
    print(f"saved: {DATA/'sample_text.txt'}")


if __name__ == "__main__":
    main()
