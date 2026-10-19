/* 中文阅读效率自动评估 — 静态演示版
 * 作者：中山大学心理学系 刘彦平
 * 交互：选择文本 / 试次即自动更新，无独立分析按钮
 */
"use strict";

const STATE = {
  mode: "material",
  materials: [],
  byKey: new Map(),
  norm: null,
  ng: null,
  scoring: null,
  vocab: null,
  freeSentences: [],
  freeIndex: 0,
  trials: [],
  trialIndex: 0,
  fileName: "",
  materialGaze: new Map(),
  gazeSubject: "",
  sentence: null,
  reader: null,
};

const ENT_DEFAULT = 4.17;
const SVG_NS = "http://www.w3.org/2000/svg";
const GAZE_COLOR = "#dc2626";

const $ = (id) => document.getElementById(id);

/* ---------------- 数据加载（在线 fetch 或离线内联） ---------------- */
async function loadData(name) {
  if (window.__DATA__ && window.__DATA__[name]) return window.__DATA__[name];
  const r = await fetch(`data/${name}.json`);
  if (!r.ok) throw new Error(`${name}.json ${r.status}`);
  return r.json();
}

/* ---------------- 启动 ---------------- */
async function boot() {
  try {
    const [m, n, g] = await Promise.all([
      loadData("materials"),
      loadData("norm"),
      loadData("material_gaze").catch(() => null),
    ]);
    STATE.materials = m.sentences;
    STATE.norm = n;
    STATE.materials.forEach((s) => STATE.byKey.set(s.key, s));

    if (g && g.trials) {
      STATE.gazeSubject = g.subject || "";
      const valid = new Set(STATE.materials.map((s) => s.key));
      Object.entries(g.trials).forEach(([k, v]) => {
        if (valid.has(k)) STATE.materialGaze.set(k, v);
      });
      $("materialGazeHint").textContent =
        `内置被试 ${STATE.gazeSubject} 的实验眼动记录（${STATE.materialGaze.size} 句）；` +
        `句子前带 ● 的表示有记录，选中即自动叠加并计算阅读效率。`;
    } else {
      $("materialGazeHint").textContent = "未找到内置眼动数据，仅显示文本难度。";
    }

    fillSentenceSelect();
    // 默认选中第一个有内置眼动的句子
    const first = STATE.materials.find((s) => STATE.materialGaze.has(s.key));
    if (first) $("sentenceSelect").value = first.key;

    updateEyeAvailability();
    await refresh();
  } catch (e) {
    alert("数据加载失败：" + e.message +
      "\n请通过 HTTP 访问（GitHub Pages 或本地服务器），不要直接双击打开 HTML。");
  }
}

function fillSentenceSelect() {
  const sel = $("sentenceSelect");
  sel.innerHTML = "";
  for (const s of STATE.materials) {
    const opt = document.createElement("option");
    opt.value = s.key;
    const mark = STATE.materialGaze.has(s.key) ? "● " : "";
    opt.textContent = `${mark}${s.key} · ${s.text.slice(0, 16)}${s.text.length > 16 ? "…" : ""}`;
    sel.appendChild(opt);
  }
}

/* ---------------- 任意文本 ---------------- */
async function ensureFreeResources() {
  if (STATE.ng) return;
  $("textStatus").textContent = "加载近似模型…";
  const [ng, sc, vo] = await Promise.all([
    loadData("ngram"),
    loadData("scoring"),
    loadData("vocab"),
  ]);
  STATE.ng = ng;
  STATE.scoring = sc;
  STATE.vocab = new Set(vo);
  $("textStatus").textContent = "";
}

function isCJK(ch) { return /[\u4e00-\u9fff]/.test(ch); }

function charSurprisal(s, j) {
  const ng = STATE.ng;
  const ch = s[j];
  if (!isCJK(ch)) return 0;
  if (j >= 2) {
    const k = s.slice(j - 2, j + 1);
    if (ng.trigram[k] !== undefined) return ng.trigram[k];
  }
  if (j >= 1) {
    const k = s.slice(j - 1, j + 1);
    if (ng.bigram[k] !== undefined) return ng.bigram[k];
  }
  return ng.unigram[ch] !== undefined ? ng.unigram[ch] : ng.default_surprisal;
}

function ridgeScore(surp, ent, len) {
  const sc = STATE.scoring;
  const x = [surp, ent, len];
  let z = 0;
  for (let i = 0; i < 3; i++) z += ((x[i] - sc.scaler_mean[i]) / sc.scaler_scale[i]) * sc.coef[i];
  const yhat = z + sc.intercept;
  const v = ((yhat - sc.score_lo) / (sc.score_hi - sc.score_lo)) * 100;
  return Math.max(0, Math.min(100, v));
}

function maxMatch(s, vocab, maxlen = 5) {
  const out = [];
  let i = 0;
  while (i < s.length) {
    let hit = null;
    for (let L = Math.min(maxlen, s.length - i); L >= 1; L--) {
      const w = s.substr(i, L);
      if (L === 1 || vocab.has(w)) { hit = { w, s: i, l: L }; i += L; break; }
    }
    if (hit && /^[\u4e00-\u9fff]+$/.test(hit.w)) out.push(hit);
  }
  return out;
}

function scoreFreeSentence(text) {
  const words = maxMatch(text, STATE.vocab).map((w) => {
    let surp = 0;
    for (let k = 0; k < w.l; k++) surp += charSurprisal(text, w.s + k);
    const d = Math.round(ridgeScore(surp, ENT_DEFAULT, w.l) * 10) / 10;
    return { ...w, surp, ent: ENT_DEFAULT, d };
  });
  return { key: "FREE", text, words };
}

function splitSentences(text) {
  const t = text.trim();
  if (!t) return [];
  const lines = t.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  if (lines.length > 1) return lines;
  const out = [];
  let buf = "";
  for (const ch of t) {
    buf += ch;
    if ("。！？!?".includes(ch)) { out.push(buf.trim()); buf = ""; }
  }
  if (buf.trim()) out.push(buf.trim());
  return out;
}

function fillFreeSentenceSelect() {
  const sel = $("freeSentenceSelect");
  sel.innerHTML = "";
  STATE.freeSentences.forEach((s, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `第 ${i + 1} 句 · ${s.slice(0, 20)}${s.length > 20 ? "…" : ""}`;
    sel.appendChild(opt);
  });
  $("freeSentenceRow").hidden = STATE.freeSentences.length === 0;
}

/* ---------------- 眼动叠加 ---------------- */
function fixationsForKey(key) {
  const t = STATE.trials.find((x) => x.tag === key);
  return t ? t.fixations : null;
}

function materialFixations(key) {
  const arr = STATE.materialGaze.get(key);
  if (!arr) return null;
  return arr.map(([x, dur, st]) => ({ x, dur, st, en: st + dur }));
}

function currentFixations() {
  if (STATE.mode === "material") {
    return materialFixations($("sentenceSelect").value);
  }
  if (!STATE.trials.length) return null;
  const t = STATE.trials[STATE.trialIndex];
  return t ? t.fixations : null;
}

function charPositions(container) {
  const wrapRect = $("gazeWrap").getBoundingClientRect();
  const map = [];
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const text = node.textContent;
    for (let i = 0; i < text.length; i++) {
      const r = document.createRange();
      r.setStart(node, i);
      r.setEnd(node, i + 1);
      const rect = r.getBoundingClientRect();
      map.push({ x: rect.left - wrapRect.left, w: rect.width });
    }
  }
  return map;
}

function drawGaze(fixations) {
  const svg = $("gazeSvg");
  svg.innerHTML = "";
  if (!fixations || !fixations.length) {
    svg.setAttribute("height", "0");
    $("gazeWrap").style.paddingTop = "10px";
    return 0;
  }

  const line = $("sentenceLine");
  const map = charPositions(line);
  if (!map.length) return 0;

  const left = +$("leftMargin").value;
  const cw = +$("charWidth").value;
  const height = 62;
  let drawn = 0;
  svg.setAttribute("height", String(height));
  $("gazeWrap").style.paddingTop = `${height + 6}px`;

  for (const f of fixations) {
    const ci = (f.x - left) / cw;
    if (ci < -0.5 || ci > map.length) continue;
    const i0 = Math.max(0, Math.min(map.length - 1, Math.floor(ci)));
    const frac = ci - i0;
    const px = map[i0].x + frac * map[i0].w;
    if (px < -20 || px > line.clientWidth + 20) continue;

    const r = 4 + 13 * Math.sqrt(Math.max(0, Math.min(1, (f.dur - 60) / 440)));
    const cy = 40;

    const stem = document.createElementNS(SVG_NS, "line");
    stem.setAttribute("x1", px); stem.setAttribute("y1", cy + r);
    stem.setAttribute("x2", px); stem.setAttribute("y2", height);
    stem.setAttribute("stroke", GAZE_COLOR);
    stem.setAttribute("stroke-width", "1");
    stem.setAttribute("opacity", "0.28");
    svg.appendChild(stem);

    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", px);
    circle.setAttribute("cy", cy);
    circle.setAttribute("r", r.toFixed(1));
    circle.setAttribute("fill", "none");
    circle.setAttribute("stroke", GAZE_COLOR);
    circle.setAttribute("stroke-width", "2.1");
    circle.setAttribute("opacity", "0.95");

    const title = document.createElementNS(SVG_NS, "title");
    title.textContent = `注视 ${Math.round(f.dur)} ms`;
    circle.appendChild(title);
    svg.appendChild(circle);
    drawn++;
  }
  return drawn;
}

/* ---------------- 渲染 ---------------- */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function card(k, v, s = "") {
  return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`;
}

function renderDifficulty(sentence, fixations) {
  const rows = sentence.words;
  if (!rows.length) { $("difficultyBlock").hidden = true; return; }

  const overall = rows.reduce((a, r) => a + r.d, 0) / rows.length;
  const hardest = rows.slice().sort((a, b) => b.d - a.d).slice(0, 3).map((r) => r.w).join("、");

  $("diffCards").innerHTML =
    card("整段难度分", `${overall.toFixed(0)} / 100`) +
    card("词数", rows.length) +
    card("最难词", hardest);

  const text = sentence.text;
  let pos = 0;
  let html = "";
  for (const r of rows) {
    if (r.s > pos) html += `<span class="punct">${escapeHtml(text.slice(pos, r.s))}</span>`;
    const dd = Math.max(0, Math.min(100, r.d));
    const hue = 120 * (1 - dd / 100);
    html += `<span class="w" title="${escapeHtml(r.w)}：难度 ${dd.toFixed(1)}" ` +
            `style="background:hsl(${hue.toFixed(0)},58%,87%)">${escapeHtml(r.w)}</span>`;
    pos = r.s + r.l;
  }
  if (pos < text.length) html += `<span class="punct">${escapeHtml(text.slice(pos))}</span>`;
  $("sentenceLine").innerHTML = html;

  $("difficultyBlock").hidden = false;

  const paint = () => {
    const drawn = drawGaze(fixations);
    const legend = $("gazeLegend");
    if (fixations && fixations.length) {
      let t = `词背景色：绿 → 红表示由易到难。空心圆：一次注视，圆越大注视越久（共 ${drawn} 次`;
      if (drawn < fixations.length) {
        t += `，另有 ${fixations.length - drawn} 次落在句外未显示`;
      }
      t += "）。";
      legend.textContent = t;
    } else if (STATE.mode === "material" && !STATE.materialGaze.has(sentence.key)) {
      legend.textContent = "该句没有内置眼动记录，仅显示文本难度。句子前带 ● 的条目可直接查看注视点。";
    } else {
      legend.textContent = "词背景色：绿 → 红表示由易到难。加载 .asc 后，此处会叠加注视圆点。";
    }
  };
  // 立即绘制一次，并在下一帧重绘以适配最终布局
  paint();
  requestAnimationFrame(paint);
}

function renderReader(rows, summary) {
  const norm = STATE.norm;
  const pSpeed = percentile(summary.speed, norm.speed_wps);
  const pGd = percentile(summary.meanGd, norm.mean_gd);
  const pSkip = percentile(summary.skipping, norm.mean_skip);

  $("readerCards").innerHTML =
    card("阅读速度", `${summary.speed.toFixed(2)} 词/秒`, `常模 ${pSpeed.toFixed(0)} 分位`) +
    card("平均凝视时间", `${summary.meanGd.toFixed(0)} ms`, `常模 ${pGd.toFixed(0)} 分位`) +
    card("跳读率", `${(summary.skipping * 100).toFixed(1)}%`, `常模 ${pSkip.toFixed(0)} 分位`) +
    card("有效注视", summary.nFixated);

  let band, note;
  if (pSpeed >= 70) { band = "偏快"; note = `该读者的阅读速度高于约 ${pSpeed.toFixed(0)}% 的常模被试。`; }
  else if (pSpeed >= 30) { band = "中等"; note = `该读者的阅读速度处于常模中间区间（${pSpeed.toFixed(0)} 分位）。`; }
  else { band = "偏慢"; note = `该读者的阅读速度低于约 ${(100 - pSpeed).toFixed(0)}% 的常模被试。`; }
  $("verdict").innerHTML = `判读：<b>${band}</b>。${note}（常模 n = ${norm.n}）`;
  $("readerHint").textContent =
    "注：单试次的速度估计受跳读与试次长度影响较大，与常模比较时宜累积多个试次。";

  let html = "<thead><tr><th>词</th><th>注视次数</th><th>FFD (ms)</th><th>GD (ms)</th><th>TT (ms)</th><th>跳读</th><th>难度</th></tr></thead><tbody>";
  for (const r of rows) {
    html += `<tr><td>${r.w}</td><td>${r.n}</td><td>${r.skip ? "—" : r.ffd.toFixed(0)}</td>` +
            `<td>${r.skip ? "—" : r.gd.toFixed(0)}</td><td>${r.skip ? "—" : r.tt.toFixed(0)}</td>` +
            `<td>${r.skip ? "是" : "否"}</td><td>${Number(r.d).toFixed(1)}</td></tr>`;
  }
  $("wordTable").innerHTML = html + "</tbody>";

  STATE.reader = { rows, summary, band, note, pSpeed };
  $("readerSection").hidden = false;
}

/* ---------------- 指标 ---------------- */
function analyzeTrial(fixations, words, left, cw) {
  const assigned = words.map(() => []);
  for (const f of fixations) {
    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      const x0 = left + w.s * cw;
      const x1 = left + (w.s + w.l) * cw;
      if (f.x >= x0 && f.x < x1) { assigned[i].push(f); break; }
    }
  }
  return words.map((w, i) => {
    const fxs = assigned[i].slice().sort((a, b) => a.st - b.st);
    if (!fxs.length) return { w: w.w, d: w.d, n: 0, ffd: 0, gd: 0, tt: 0, skip: 1 };
    const firstpass = [fxs[0]];
    for (let k = 1; k < fxs.length; k++) {
      if (fxs[k].st - firstpass[firstpass.length - 1].en > 1) break;
      firstpass.push(fxs[k]);
    }
    return {
      w: w.w, d: w.d, n: fxs.length,
      ffd: fxs[0].dur,
      gd: firstpass.reduce((a, f) => a + f.dur, 0),
      tt: fxs.reduce((a, f) => a + f.dur, 0),
      skip: 0,
    };
  });
}

function summarize(rows, fixations) {
  const nWords = rows.length;
  const read = rows.filter((r) => r.skip === 0);
  const st = Math.min(...fixations.map((f) => f.st));
  const en = Math.max(...fixations.map((f) => f.en));
  const total = en - st;
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
  return {
    nWords,
    nFixated: read.length,
    skipping: nWords ? 1 - read.length / nWords : 0,
    meanGd: mean(read.map((r) => r.gd)),
    totalMs: total,
    speed: total > 0 ? nWords / (total / 1000) : 0,
  };
}

function percentile(v, arr) {
  if (!arr || !arr.length) return NaN;
  return (100 * arr.filter((x) => x < v).length) / arr.length;
}

/* ---------------- ASC 解析 ---------------- */
const RE_EFIX = /^EFIX\s+R\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+(?:\.\d+)?|\.\d+)\s+(-?\d+(?:\.\d+)?|\.\d+)/;
const RE_MSG = /^MSG\s+(\d+)\s+(.*)$/;
const RE_SENTID = /SENTENCEID\s+(\S+)/;
const RE_START = /Sent\s+MYKEYWORD\s+100/;
const RE_END = /Sent\s+MYKEYWORD\s+200/;

function parseAsc(text) {
  const fixations = [];
  const marks = [];
  let tag = "";
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    let m = RE_EFIX.exec(line);
    if (m) {
      fixations.push({ st: +m[1], en: +m[2], dur: +m[3], x: parseFloat(m[4]), y: parseFloat(m[5]) });
      continue;
    }
    m = RE_MSG.exec(line);
    if (m) {
      const t = +m[1], msg = m[2];
      const sm = RE_SENTID.exec(msg);
      if (sm) { tag = sm[1]; continue; }
      if (RE_START.test(msg)) marks.push({ t, kind: "start", tag });
      else if (RE_END.test(msg)) marks.push({ t, kind: "end", tag });
    }
  }
  const starts = marks.filter((m) => m.kind === "start");
  const ends = marks.filter((m) => m.kind === "end").map((m) => m.t);
  const trials = [];
  if (starts.length) {
    for (let i = 0; i < starts.length; i++) {
      const t0 = starts[i].t;
      const t1 = ends[i] !== undefined ? ends[i] : (starts[i + 1] ? starts[i + 1].t : Infinity);
      const fx = fixations.filter((f) => f.st >= t0 && f.st < t1);
      if (fx.length) trials.push({ index: i + 1, tag: starts[i].tag, fixations: fx });
    }
  } else if (fixations.length) {
    trials.push({ index: 1, tag: "", fixations });
  }
  return trials;
}

/* ---------------- 状态与刷新 ---------------- */
function updateEyeAvailability() {
  const isMat = STATE.mode === "material";
  // 材料模式使用内置眼动，整个上传模块隐藏
  $("eyePanel").hidden = isMat;
  if (isMat) return;

  const hasText = STATE.freeSentences.length > 0;
  $("fileInput").disabled = !hasText;
  $("eyeTag").textContent = hasText ? "可选" : "需先选择文本";
  $("eyeHint").textContent = hasText
    ? "加载 .asc 后，选择句子或试次即自动显示注视点与阅读效率。"
    : "请先上传 .txt 文本文件，再加载眼动数据。";
  $("fileStatus").textContent = STATE.trials.length
    ? `${STATE.fileName} · ${STATE.trials.length} 个试次`
    : (hasText ? "未加载 · 仅显示文本难度" : "未加载");
  $("trialRow").hidden = !STATE.trials.length;
}

function syncTrialSelect(key) {
  const i = STATE.trials.findIndex((t) => t.tag === key);
  if (i >= 0) {
    STATE.trialIndex = i;
    $("trialSelect").value = String(i);
  }
}

async function currentSentence() {
  if (STATE.mode === "material") {
    return STATE.byKey.get($("sentenceSelect").value) || null;
  }
  if (!STATE.freeSentences.length) return null;
  await ensureFreeResources();
  const i = Math.min(STATE.freeIndex, STATE.freeSentences.length - 1);
  return scoreFreeSentence(STATE.freeSentences[i]);
}

async function refresh() {
  $("readerSection").hidden = true;
  STATE.reader = null;

  const sentence = await currentSentence();
  if (!sentence || !sentence.words.length) {
    $("difficultyBlock").hidden = true;
    return;
  }
  STATE.sentence = sentence;
  const fx = currentFixations();
  renderDifficulty(sentence, fx);

  if (fx && fx.length) {
    const left = +$("leftMargin").value;
    const cw = +$("charWidth").value;
    const rows = analyzeTrial(fx, sentence.words, left, cw);
    const summary = summarize(rows, fx);
    renderReader(rows, summary);
  }
}

/* ---------------- 事件 ---------------- */
function bind() {
  $("modeSelect").addEventListener("change", async (e) => {
    STATE.mode = e.target.value;
    $("materialBlock").hidden = STATE.mode !== "material";
    $("freeBlock").hidden = STATE.mode !== "free";
    updateEyeAvailability();
    await refresh();
  });

  $("sentenceSelect").addEventListener("change", refresh);

  $("freeSentenceSelect").addEventListener("change", async (e) => {
    STATE.freeIndex = +e.target.value;
    // 句子与试次按顺序对应：换句子即换到同序号的试次
    if (STATE.trials.length) {
      STATE.trialIndex = Math.min(STATE.freeIndex, STATE.trials.length - 1);
      $("trialSelect").value = String(STATE.trialIndex);
    }
    await refresh();
  });

  $("textInput").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".txt")) {
      alert("仅支持 .txt 文本文件");
      e.target.value = "";
      return;
    }
    STATE.freeSentences = splitSentences(await f.text());
    STATE.freeIndex = 0;
    $("textStatus").textContent = f.name;
    fillFreeSentenceSelect();
    updateEyeAvailability();
    await refresh();
  });

  $("fileInput").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".asc")) {
      alert("仅支持 .asc 眼动文件");
      e.target.value = "";
      return;
    }
    STATE.fileName = file.name;
    STATE.trials = parseAsc(await file.text());
    STATE.trialIndex = 0;

    const sel = $("trialSelect");
    sel.innerHTML = "";
    STATE.trials.forEach((t, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = `试次 ${t.index}${t.tag ? " " + t.tag : ""} · ${t.fixations.length} 个注视`;
      sel.appendChild(opt);
    });
    sel.value = "0";

    if (STATE.mode === "material") {
      const mi = STATE.trials.findIndex((t) => t.tag && STATE.byKey.has(t.tag));
      if (mi >= 0) {
        STATE.trialIndex = mi;
        sel.value = String(mi);
        $("sentenceSelect").value = STATE.trials[mi].tag;
      }
    } else if (STATE.freeSentences.length) {
      STATE.freeIndex = 0;
      $("freeSentenceSelect").value = "0";
    }

    updateEyeAvailability();
    await refresh();
  });

  $("trialSelect").addEventListener("change", async (e) => {
    STATE.trialIndex = +e.target.value;
    if (STATE.mode === "material") {
      const t = STATE.trials[STATE.trialIndex];
      if (t && t.tag && STATE.byKey.has(t.tag)) {
        $("sentenceSelect").value = t.tag;
      }
    } else if (STATE.freeSentences.length) {
      STATE.freeIndex = Math.min(STATE.trialIndex, STATE.freeSentences.length - 1);
      $("freeSentenceSelect").value = String(STATE.freeIndex);
    }
    await refresh();
  });

  $("leftMargin").addEventListener("change", refresh);
  $("charWidth").addEventListener("change", refresh);

  $("exportBtn").addEventListener("click", () => {
    const r = STATE.reader;
    const s = STATE.sentence;
    if (!r || !s) return;
    const overall = s.words.reduce((a, x) => a + x.d, 0) / s.words.length;
    const rowsHtml = r.rows.map((x) =>
      `<tr><td>${x.w}</td><td>${Number(x.d).toFixed(1)}</td><td>${x.skip ? "—" : x.gd.toFixed(0)}</td></tr>`).join("");
    const html = `<!doctype html><meta charset="utf-8"><title>阅读效率评估报告</title>
<style>body{font-family:"Microsoft YaHei",sans-serif;max-width:900px;margin:40px auto;line-height:1.8}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:6px 10px;text-align:left}</style>
<h1>阅读效率评估报告</h1>
<p style="color:#666">中山大学心理学系 · 刘彦平 开发</p>
<h2>文本</h2><p>${s.text}</p>
<h2>整段难度分</h2><p style="font-size:28px"><b>${overall.toFixed(0)} / 100</b></p>
<h2>阅读效率</h2><p>阅读速度 ${r.summary.speed.toFixed(2)} 词/秒（常模 ${r.pSpeed.toFixed(0)} 分位）；
平均凝视时间 ${r.summary.meanGd.toFixed(0)} ms；跳读率 ${(r.summary.skipping * 100).toFixed(1)}%。
判读：${r.band}。</p>
<h2>逐词</h2><table><tr><th>词</th><th>难度分</th><th>GD (ms)</th></tr>${rowsHtml}</table>
<h2>说明</h2><p>难度分：实验材料来自 GPT-2 特征 + Ridge 预计算；任意文本为 n-gram 蒸馏近似。
阅读效率为实测指标与 45 人常模的对照，不是预测。</p>`;
    const blob = new Blob([html], { type: "text/html;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "reading_efficiency_report.html";
    a.click();
    URL.revokeObjectURL(a.href);
  });

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(refresh, 150);
  });
}

boot();
bind();
