"""asc_io.py — EyeLink 数据读取
================================================================
职责
    1. 若输入是 .edf，调用本机 edf2asc 转换为 .asc
    2. 解析 .asc 中的注视（EFIX）、扫视（ESACC）与消息（MSG）事件
    3. 按试次标记切分；若无标记则整体作为单个试次

不依赖项目内其它模块，可独立运行。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_FLOAT = r"(-?\d+(?:\.\d+)?|\.\d+)"
_RE_EFIX = re.compile(rf"^EFIX\s+R\s+(\d+)\s+(\d+)\s+(\d+)\s+{_FLOAT}\s+{_FLOAT}\s+{_FLOAT}")
_RE_ESACC = re.compile(
    rf"^ESACC\s+R\s+(\d+)\s+(\d+)\s+(\d+)\s+{_FLOAT}\s+{_FLOAT}\s+{_FLOAT}\s+"
    rf"{_FLOAT}\s+{_FLOAT}\s+{_FLOAT}")
_RE_MSG = re.compile(r"^MSG\s+(\d+)\s+(.*)$")
_RE_TRIAL_START = re.compile(r"Sent\s+MYKEYWORD\s+100")
_RE_TRIAL_END = re.compile(r"Sent\s+MYKEYWORD\s+200")
_RE_SENTID = re.compile(r"SENTENCEID\s+(\S+)")

EDF2ASC_CANDIDATES = [
    r"C:\Program Files (x86)\SR Research\Eyelink\bin\edf2asc.exe",
    r"C:\Program Files\SR Research\Eyelink\bin\edf2asc.exe",
]


@dataclass
class Fixation:
    sttime: int
    entime: int
    duration: int
    x: float
    y: float


@dataclass
class Saccade:
    sttime: int
    entime: int
    duration: int
    sx: float
    sy: float
    ex: float
    ey: float
    amplitude: float


@dataclass
class Trial:
    index: int
    t_start: int = 0
    t_end: int = 0
    tag: str = ""
    fixations: List[Fixation] = field(default_factory=list)
    saccades: List[Saccade] = field(default_factory=list)


def find_edf2asc() -> Optional[str]:
    for p in EDF2ASC_CANDIDATES:
        if Path(p).exists():
            return p
    return shutil.which("edf2asc")


def convert_edf(edf_path: Path, out_dir: Optional[Path] = None) -> Path:
    """调用 edf2asc 将 .edf 转为 .asc，返回 asc 路径。"""
    exe = find_edf2asc()
    if exe is None:
        raise RuntimeError(
            "未找到 edf2asc。请安装 SR Research Eyelink 开发者工具，"
            "或直接上传已转换的 .asc 文件。")
    edf_path = Path(edf_path)
    out_dir = Path(out_dir) if out_dir else edf_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-y", str(edf_path), "-o", str(out_dir / (edf_path.stem + ".asc"))]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    asc = out_dir / (edf_path.stem + ".asc")
    if not asc.exists():
        raise RuntimeError(f"edf2asc 失败：{res.stderr or res.stdout}")
    return asc


def parse_asc(path: Path) -> List[Trial]:
    """解析 asc，返回试次列表。

    若文件含 'Sent MYKEYWORD 100/200' 标记则按标记切分；
    否则把全部注视视为一个试次（适用于单句阅读记录）。
    """
    path = Path(path)
    fxs: List[Fixation] = []
    scs: List[Saccade] = []
    marks: List[tuple[int, str, str]] = []   # (time, kind, tag)
    current_tag = ""

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = _RE_EFIX.match(line)
            if m:
                fxs.append(Fixation(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                    float(m.group(4)), float(m.group(5))))
                continue
            m = _RE_ESACC.match(line)
            if m:
                scs.append(Saccade(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                   float(m.group(4)), float(m.group(5)),
                                   float(m.group(6)), float(m.group(7)),
                                   float(m.group(8))))
                continue
            m = _RE_MSG.match(line)
            if m:
                t, msg = int(m.group(1)), m.group(2)
                sm = _RE_SENTID.search(msg)
                if sm:
                    current_tag = sm.group(1)
                    continue
                if _RE_TRIAL_START.search(msg):
                    marks.append((t, "start", current_tag))
                elif _RE_TRIAL_END.search(msg):
                    marks.append((t, "end", current_tag))

    trials: List[Trial] = []
    starts = [(t, tag) for t, k, tag in marks if k == "start"]
    ends = [t for t, k, _ in marks if k == "end"]
    if starts:
        for i, (t0, tag) in enumerate(starts):
            t1 = ends[i] if i < len(ends) else (starts[i + 1][0] if i + 1 < len(starts)
                                                else 10 ** 12)
            tr = Trial(index=i + 1, t_start=t0, t_end=t1, tag=tag)
            tr.fixations = [f for f in fxs if t0 <= f.sttime < t1]
            tr.saccades = [s for s in scs if t0 <= s.sttime < t1]
            if tr.fixations:
                trials.append(tr)
    else:
        tr = Trial(index=1, t_start=0, t_end=10 ** 12, tag="")
        tr.fixations = fxs
        tr.saccades = scs
        if fxs:
            trials.append(tr)

    return trials


def summarize(trials: List[Trial]) -> dict:
    n_fix = sum(len(t.fixations) for t in trials)
    dur = [f.duration for t in trials for f in t.fixations]
    return {
        "n_trials": len(trials),
        "n_fixations": n_fix,
        "median_fixation_ms": float(sorted(dur)[len(dur) // 2]) if dur else 0.0,
        "has_trial_markers": any(t.tag for t in trials),
    }
