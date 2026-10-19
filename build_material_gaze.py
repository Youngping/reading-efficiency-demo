"""build_material_gaze.py — 生成实验材料的内置眼动数据
================================================================
从真实 asc 提取每个试次的注视序列，压缩为静态 JSON，
供网页在"实验材料"模式下自动加载，无需用户上传。

隐私：输出只保留被试编号（如 sub01），不写入姓名缩写等可识别信息。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import asc_io  # noqa: E402

OUT = HERE / "docs" / "data" / "material_gaze.json"


def find_asc() -> Path:
    cands = sorted((HERE / "test_data").glob("sub*.asc"))
    if not cands:
        raise SystemExit("未找到 test_data/sub*.asc")
    return cands[0]


def anonymize(stem: str) -> str:
    m = re.search(r"(sub\d+)", stem, re.I)
    return m.group(1).lower() if m else "anonymous"


def main() -> None:
    asc = find_asc()
    trials = asc_io.parse_asc(asc)
    data = {}
    for t in trials:
        if not t.tag:
            continue
        # [x, duration, start_time] 便于计算首次通过
        data[t.tag] = [[round(f.x, 1), int(f.duration), int(f.sttime)]
                       for f in sorted(t.fixations, key=lambda z: z.sttime)]

    payload = {
        "subject": anonymize(asc.stem),
        "n_trials": len(data),
        "note": "内置示例：一名被试的实验眼动记录（编号已脱敏），用于演示材料模式的自动叠加",
        "trials": data,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    print(f"subject: {payload['subject']}")
    print(f"trials: {len(data)}")
    print(f"fixations: {sum(len(v) for v in data.values())}")
    print(f"saved: {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    print("sample keys:", list(data)[:5])


if __name__ == "__main__":
    main()
