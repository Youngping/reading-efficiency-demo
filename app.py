"""app.py — 中文阅读难度自动评估（Streamlit 演示程序）
================================================================
启动:
    cd reading_demo
    streamlit run app.py
"""
from __future__ import annotations

import html
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python_analysis" / "full_v3"))

import asc_io          # noqa: E402
import difficulty      # noqa: E402
import layout          # noqa: E402
import reader          # noqa: E402

st.set_page_config(page_title="中文阅读难度自动评估", page_icon="📖", layout="wide")

EM_PKL = HERE.parent / "python_analysis" / "full_v3" / "cache" / "em_word.pkl"
COND_MAP = {1: (0, 1), 2: (1, 1), 3: (0, 0), 4: (1, 0)}   # C1..C4 -> (wfcond, preview)
DEMO_TEXT = "他的行医技术在经过多年培养之后终于能够独当一面了"


# ---------------------------------------------------------------------------
# 材料索引：从眼动数据重建实验句子
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_materials() -> dict:
    if not EM_PKL.exists():
        return {}
    em = pd.read_pickle(EM_PKL)[["sentid", "wordid", "wfcond", "preview", "wordtext"]]
    em = em.drop_duplicates(["sentid", "wordid", "wfcond", "preview"])
    out = {}
    for (sid, wf, pv), g in em.groupby(["sentid", "wfcond", "preview"]):
        out[(int(sid), int(wf), int(pv))] = "".join(
            g.sort_values("wordid").wordtext.astype(str))
    return out


def sentence_from_tag(tag: str, mats: dict) -> str | None:
    m = re.match(r"^E(\d+)C(\d)$", str(tag).strip())
    if not m:
        return None
    sid, c = int(m.group(1)), int(m.group(2))
    if c not in COND_MAP:
        return None
    wf, pv = COND_MAP[c]
    return mats.get((sid, wf, pv))


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def heatmap_html(words, scores) -> str:
    parts = []
    for w, s in zip(words, scores):
        hue = 120 * (1 - float(s) / 100)
        parts.append(
            f'<span title="{html.escape(str(w))}：难度 {s:.0f}" '
            f'style="background:hsl({hue:.0f},72%,84%);padding:3px 5px;'
            f'margin:2px;border-radius:4px;display:inline-block">'
            f'{html.escape(str(w))}</span>')
    return ('<div style="line-height:2.4;font-size:17px;font-family:'
            "'Microsoft YaHei',sans-serif\">" + "".join(parts) + "</div>")


def trial_duration(fxs) -> float:
    if not fxs:
        return 0.0
    return float(max(f.entime for f in fxs) - min(f.sttime for f in fxs))


def save_upload(uploaded) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="reading_demo_"))
    path = tmpdir / uploaded.name
    path.write_bytes(uploaded.getbuffer())
    return path


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("呈现布局")
    left_margin = st.number_input("行首像素 x", min_value=0, max_value=4000,
                                  value=600, step=10)
    char_width = st.number_input("每字符宽度（像素）", min_value=8, max_value=200,
                                 value=50, step=1)
    st.caption("必须与实验呈现程序一致。默认值对应当前项目：字符 i 占据 "
               "[600 + i×50, 600 + (i+1)×50)。")

    st.divider()
    st.header("模型与常模")
    st.markdown(
        "- 文本难度：GPT-2 surprisal / entropy / 词长 → Ridge\n"
        "  （按句分组 5 折：R² 0.20，Spearman ρ 0.42）\n"
        "- 读者常模：45 名被试的句子级阅读指标\n"
        "- 读者层是**测量 + 常模定位**，不是预测"
    )

# ---------------------------------------------------------------------------
# 主体
# ---------------------------------------------------------------------------
st.title("中文阅读难度自动评估")
st.caption("上传一次阅读的眼动记录（.asc 或 .edf），得到逐词难度与读者效率评估。")

c1, c2 = st.columns([1.1, 1])
with c1:
    up = st.file_uploader("眼动数据（.asc 或 .edf）", type=["asc", "edf"],
                          help=".edf 需本机安装 SR Research 的 edf2asc。")
with c2:
    st.write("")
    st.write("")
    st.info("若文件含 SENTENCEID 标记，程序会自动从实验材料中还原该试次的句子文本；"
            "否则请手动在下方输入。")

# ---------- 解析文件 ----------
trials = []
parse_note = ""
if up is not None:
    try:
        raw = save_upload(up)
        if raw.suffix.lower() == ".edf":
            with st.spinner("调用 edf2asc 转换（大文件可能需要数十秒）…"):
                asc = asc_io.convert_edf(raw, raw.parent)
            parse_note = "已由 .edf 自动转换为 .asc。"
        else:
            asc = raw
        with st.spinner("解析注视事件…"):
            trials = asc_io.parse_asc(asc)
    except Exception as e:  # noqa: BLE001
        st.error(f"眼动文件解析失败：{e}")

mats = load_materials()

chosen_trial = None
if trials:
    labels = [f"试次 {i+1}   {t.tag or '（无标记）'}   {len(t.fixations)} 个注视"
              for i, t in enumerate(trials)]
    sel = st.selectbox("选择要分析的试次", options=list(range(len(trials))),
                       format_func=lambda i: labels[i])
    chosen_trial = trials[sel]
    st.caption(parse_note + f" 文件共解析到 {len(trials)} 个试次。")

auto_text = ""
if chosen_trial is not None:
    auto_text = sentence_from_tag(chosen_trial.tag, mats) or ""

text = st.text_area("本试次的句子文本", value=auto_text or DEMO_TEXT, height=90,
                    help="程序会自动尝试从材料库还原；也可手动修改。")

run = st.button("开始分析", type="primary", use_container_width=True)

# ---------- 分析 ----------
if run:
    if not text.strip():
        st.error("请先输入句子文本。")
        st.stop()

    sentence = layout.split_sentences(text)[0]

    # 词级映射
    metrics = None
    if chosen_trial is not None and chosen_trial.fixations:
        metrics, _ = layout.analyze_trial(chosen_trial.fixations, sentence,
                                          left_margin, char_width)
        metrics["sentence"] = sentence
        metrics["trial"] = 1

    if metrics is None:
        words = layout.build_words(sentence, left_margin, char_width)
        metrics = pd.DataFrame([{
            "idx": w.idx, "word": w.text, "char_start": w.char_start,
            "n_fix": 0, "ffd": 0.0, "gd": 0.0, "tt": 0.0, "skipping": 1,
            "sentence": sentence, "trial": 1} for w in words])

    # 文本难度
    with st.spinner("计算 GPT-2 文本特征并评分…"):
        rows = [{"sentence": sentence, "word": r.word,
                 "char_start": int(r.char_start), "length": len(str(r.word))}
                for _, r in metrics.iterrows()]
        feats = difficulty.word_features([sentence], rows)
        scores = difficulty.predict_difficulty(feats)
        diff_df = pd.DataFrame(rows)[["word"]].copy()
        diff_df["difficulty"] = scores
        diff_df["surprisal"] = feats.t_surprisal.values
        diff_df["entropy"] = feats.t_entropy.values

    st.divider()
    tab1, tab2 = st.tabs(["文本难度", "读者效率"])

    with tab1:
        overall = float(diff_df.difficulty.mean())
        hardest = diff_df.nlargest(3, "difficulty")
        a, b, c = st.columns(3)
        a.metric("整段难度分", f"{overall:.0f} / 100")
        b.metric("词数", f"{len(diff_df)}")
        c.metric("最难词", "、".join(hardest.word.tolist()))

        st.subheader("逐词难度")
        st.markdown(heatmap_html(diff_df.word, diff_df.difficulty),
                    unsafe_allow_html=True)
        st.caption("颜色由绿到红表示由易到难；悬停可看具体分值。")

        with st.expander("查看数值"):
            st.dataframe(diff_df.round(3), use_container_width=True,
                         hide_index=True)

    with tab2:
        if chosen_trial is None or not chosen_trial.fixations:
            st.warning("未提供有效眼动数据。读者效率层需要 .asc 或 .edf 文件。")
        else:
            total_time = trial_duration(chosen_trial.fixations)
            summary = layout.trial_summary(metrics, total_time)
            norm = reader.load_norm()
            ev = reader.evaluate(summary, norm)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("阅读速度", f"{ev['speed_wps']:.2f} 词/秒",
                      f"常模 {ev['percentile_speed']:.0f} 分位")
            k2.metric("平均凝视时间", f"{ev['mean_gd_ms']:.0f} ms",
                      f"常模 {ev['percentile_gd']:.0f} 分位")
            k3.metric("跳读率", f"{ev['skipping_rate']:.1%}",
                      f"常模 {ev['percentile_skip']:.0f} 分位")
            k4.metric("有效注视", f"{int(metrics.n_fix.sum())}")

            st.info(f"判读：**{ev['band']}**。{ev['note']}"
                    f"（常模 n = {ev['norm_n']}）")
            st.caption("注：单试次的速度估计受跳读与试次长度影响较大，"
                       "与常模比较时宜累积多个试次。")

            st.subheader("与常模对照")
            st.dataframe(reader.norm_table(norm), use_container_width=True,
                         hide_index=True)

            st.subheader("逐词眼动指标")
            show = metrics[["word", "n_fix", "ffd", "gd", "tt", "skipping"]].copy()
            show.columns = ["词", "注视次数", "FFD (ms)", "GD (ms)", "TT (ms)", "跳读"]
            st.dataframe(show.round(1), use_container_width=True, hide_index=True)

            with st.expander("试次级汇总"):
                st.json({k: round(v, 3) if isinstance(v, float) else v
                         for k, v in summary.items()})

    # 导出
    rows_html = "".join(
        f"<tr><td>{html.escape(r.word)}</td><td>{r.difficulty:.1f}</td>"
        f"<td>{r.surprisal:.2f}</td><td>{r.entropy:.2f}</td></tr>"
        for r in diff_df.itertuples())
    reader_html = ""
    if chosen_trial is not None and chosen_trial.fixations:
        reader_html = (
            f"<h2>读者效率</h2><p>阅读速度 {summary['reading_speed_wps']:.2f} 词/秒"
            f"（常模 {ev['percentile_speed']:.0f} 分位）；平均凝视时间 "
            f"{summary['mean_gd_ms']:.0f} ms（常模 {ev['percentile_gd']:.0f} 分位）；"
            f"跳读率 {summary['skipping_rate']:.1%}。判读：{ev['band']}。</p>")
    report = f"""<!doctype html><meta charset="utf-8">
<title>阅读难度评估报告</title>
<style>body{{font-family:'Microsoft YaHei',sans-serif;max-width:900px;
margin:40px auto;line-height:1.8}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:6px 10px;text-align:left}}</style>
<h1>阅读难度评估报告</h1>
<h2>文本</h2><p>{html.escape(sentence)}</p>
<h2>整段难度分</h2>
<p style="font-size:28px"><b>{float(diff_df.difficulty.mean()):.0f} / 100</b></p>
{reader_html}
<h2>逐词难度</h2><table><tr><th>词</th><th>难度分</th><th>surprisal</th>
<th>entropy</th></tr>{rows_html}</table>
<h2>说明</h2>
<p>难度分来自 GPT-2 特征与人群平均凝视时间训练的模型（按句分组 5 折 R²=0.20）。
读者效率为实测指标与 45 人常模的对照，不是预测。</p>"""
    st.download_button("下载 HTML 报告", report.encode("utf-8"),
                       file_name="reading_difficulty_report.html",
                       mime="text/html", use_container_width=True)
