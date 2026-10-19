"""build_static.py — 生成 GitHub Pages 静态演示版所需的预计算数据
================================================================
输出（全部脱敏，不含任何被试级原始数据）:
    docs/data/materials.json   296 句 × 4 条件的词边界与难度分
    docs/data/norm.json        45 名被试的三项聚合常模（仅数值，无可识别信息）

说明
    - 词序列与字符边界直接取自眼动分析的 wordid/wordtext，保证与原始分析一致
    - 难度分用 models/text_difficulty.pkl（GPT-2 特征 + Ridge）逐词预计算
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import difficulty  # noqa: E402

BASE = HERE.parent
EM_PKL = BASE / "python_analysis" / "full_v3" / "cache" / "em_word.pkl"
NORM_PKL = BASE / "python_analysis" / "l3_modeling" / "l3_dataset.pkl"
DOCS = HERE / "docs"
DATA = DOCS / "data"
DATA.mkdir(parents=True, exist_ok=True)

# SENTENCEID 的 C1..C4 → (wfcond, preview)
COND_TO_WF = {1: (0, 1), 2: (1, 1), 3: (0, 0), 4: (1, 0)}


def build_materials() -> list[dict]:
    em = pd.read_pickle(EM_PKL)[["sentid", "wordid", "wfcond", "preview", "wordtext"]]
    em = em.drop_duplicates(["sentid", "wordid", "wfcond", "preview"])
    em = em.sort_values(["sentid", "wfcond", "preview", "wordid"])

    out: list[dict] = []
    for (sid, wf, pv), g in em.groupby(["sentid", "wfcond", "preview"], sort=True):
        words = g.wordtext.astype(str).tolist()
        text = "".join(words)
        char_start, rows = 0, []
        for w in words:
            rows.append({"w": w, "s": char_start, "l": len(w)})
            char_start += len(w)

        feats = pd.DataFrame([{"t_surprisal": 0.0, "t_entropy": 0.0, "length": r["l"]}
                              for r in rows])
        # 用真实 GPT-2 特征覆盖
        wrows = [{"sentence": text, "word": r["w"], "char_start": r["s"],
                  "length": r["l"]} for r in rows]
        feats = difficulty.word_features([text], wrows)
        scores = difficulty.predict_difficulty(feats)

        cond = next(c for c, (a, b) in COND_TO_WF.items() if (a, b) == (wf, pv))
        out.append({
            "key": f"E{int(sid)}C{cond}",
            "sid": int(sid),
            "cond": cond,
            "text": text,
            "words": [
                {"w": r["w"], "s": r["s"], "l": r["l"],
                 "d": round(float(scores[i]), 1),
                 "surp": round(float(feats.t_surprisal.iloc[i]), 3),
                 "ent": round(float(feats.t_entropy.iloc[i]), 3)}
                for i, r in enumerate(rows)
            ],
            "difficulty": round(float(scores.mean()), 1),
        })
        if len(out) % 100 == 0:
            print(f"  {len(out)} sentences ...")
    return out


def build_norm() -> dict:
    df = pd.read_pickle(NORM_PKL)
    return {
        "n": int(len(df)),
        "speed_wps": [round(float(x), 3) for x in sorted(df.speed_wps)],
        "mean_gd": [round(float(x), 2) for x in sorted(df.mean_gd)],
        "mean_skip": [round(float(x), 4) for x in sorted(df.mean_skip)],
        "credits": "45 subjects, sentence-level median metrics",
    }


def main() -> None:
    print("building materials ...")
    mats = build_materials()
    with open(DATA / "materials.json", "w", encoding="utf-8") as f:
        json.dump({"sentences": mats}, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  sentences: {len(mats)}")

    norm = build_norm()
    with open(DATA / "norm.json", "w", encoding="utf-8") as f:
        json.dump(norm, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  norm n = {norm['n']}")

    for p in (DATA / "materials.json", DATA / "norm.json"):
        print(f"  {p.name}: {p.stat().st_size/1024:.0f} KB")

    (DOCS / ".nojekyll").write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
