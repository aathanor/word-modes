// Charts for the site, drawn with Observable Plot from data/viz.json.
// Any element with data-chart="<name>" is filled; data-src gives the path to viz.json.
(function () {
  const MODE = {"O": "#1f4e79", "A-gen": "#d9822b", "A-idiom": "#8a8a8a", "A-def": "#111111", "X": "#d5d4cf"};
  const MODEL = {
    bert: {name: "BERT", color: "#1baf7a", symbol: "circle"},
    decoder: {name: "Decoder", color: "#6250d6", symbol: "square"},
    mpnet: {name: "mpnet", color: "#eda100", symbol: "diamond"},
    decoder_target: {name: "Decoder, at the word", color: "#898781", symbol: "triangle"}
  };
  const INK = "#52514e", GRID = "#e1e0d9", MUTED = "#898781";
  const HALF_A = ["cat", "hammer", "apple", "river", "chair", "horse", "bottle", "bridge", "knife", "tree"];
  const HALF_B = ["dog", "fork", "tomato", "mountain", "door", "cow", "lamp", "road", "spoon", "stone"];
  const WORDS = HALF_A.concat(HALF_B);
  const f2 = x => (x == null ? "n/a" : x.toFixed(2));
  const f3 = x => (x == null ? "n/a" : x.toFixed(3));

  function width(el) { return Math.max(300, Math.min(el.clientWidth || 720, 860)); }
  function base(el, opts) {
    const o = Object.assign({width: width(el), style: {fontSize: "12px", fontFamily: '"Inter Variable", Inter, system-ui, sans-serif', color: INK, background: "transparent"},
                             marginLeft: 70, marginBottom: 50, marginRight: 24}, opts);
    if (o.x) o.x = Object.assign({labelAnchor: "center", labelOffset: 40, labelArrow: "none"}, o.x);
    return o;
  }
  function legend(models, extra) {
    const items = models.filter(m => MODEL[m]).map(m => {
      const s = MODEL[m];
      const shape = {circle: '<circle cx="6" cy="6" r="4.5"/>', square: '<rect x="1.5" y="1.5" width="9" height="9"/>',
                     diamond: '<path d="M6 0.5 L11.5 6 L6 11.5 L0.5 6Z"/>', triangle: '<path d="M6 1 L11.5 11 L0.5 11Z"/>'}[s.symbol];
      return `<span class="lg"><svg width="12" height="12" fill="${s.color}">${shape}</svg>${s.name}</span>`;
    });
    return `<div class="viz-legend">${items.join("")}${extra || ""}</div>`;
  }
  function modeLegend(labels) {
    return `<div class="viz-legend">${labels.map(l =>
      `<span class="lg"><span class="sw" style="background:${MODE[l]}"></span>${l}</span>`).join("")}</div>`;
  }
  const mark = m => (MODEL[m] || {color: MUTED, symbol: "circle"});
  const modelsIn = (rows, key = "model") => Object.keys(MODEL).filter(m => rows.some(r => r[key] === m));
  function put(el, ...parts) { el.replaceChildren(...parts.map(p => typeof p === "string" ? html(p) : p)); }
  function html(s) { const d = document.createElement("div"); d.innerHTML = s; return d; }

  // ------------------------------------------------------------ home: the picture
  const charts = {};
  charts.hero = (el, v) => {
    const h = v.hero;
    const labels = ["O", "A-gen", "A-idiom"].filter(l => h.points.some(p => p.l === l));
    const ex = h.examples.map((e, i) => ({...e, i: i + 1}));
    const chart = Plot.plot(base(el, {
      height: 70 * labels.length + 50, marginLeft: 64,
      x: {label: `probe score, ${MODEL[h.model].name}: ← object-mode · abstract-mode →`, grid: true},
      fy: {domain: labels, label: null}, y: {axis: null},
      marks: [
        Plot.rectY(h.points.filter(p => labels.includes(p.l)),
          Plot.binX({y: "count"}, {x: "x", fy: "l", fill: p => MODE[p.l], thresholds: 50, inset: 1})),
        Plot.ruleX([0], {stroke: MUTED}),
        Plot.ruleX(ex, {x: "x", stroke: "#000", strokeWidth: 1.5, fy: "l"}),
        Plot.text(ex, {x: "x", fy: "l", text: d => String(d.i), frameAnchor: "top", dy: -2,
                       fontWeight: 700, fontSize: 13, fill: "#000", stroke: "white", strokeWidth: 3})
      ]
    }));
    const cards = ex.map(e => `<li><span class="ex-n">${e.i}</span><span class="ex-s">“${e.s}”</span>
      <span class="ex-m"><span class="sw" style="background:${MODE[e.l]}"></span>${e.l} · ${e.w} · score ${f2(e.x)}</span></li>`).join("");
    put(el, modeLegend(labels), chart,
        `<ol class="examples">${cards}</ol>`,
        `<p class="viz-note">${h.points.length.toLocaleString()} sentences of the ten words the probe never saw. Numbers mark the three sentences quoted below the chart.</p>`);
  };

  // ------------------------------------------------------------ stages
  charts.counts = (el, v) => {
    const rows = [];
    (v.counts || []).forEach(r => { rows.push({w: r.word, g: "Wikipedia", n: r.wikipedia}); rows.push({w: r.word, g: "fiction", n: r.fiction}); });
    const col = {"Wikipedia": "#52514e", "fiction": "#b0aea6"};
    const chart = Plot.plot(base(el, {
      height: 20 * WORDS.length + 40, marginLeft: 70,
      x: {label: "sentences", grid: true, domain: [0, 300]}, y: {domain: WORDS, label: null},
      marks: [Plot.barX(rows, {x: "n", y: "w", fill: d => col[d.g], stroke: "white", strokeWidth: 2,
                               tip: true, title: d => `${d.w}, ${d.g}: ${d.n}`}),
              Plot.ruleX([150, 300], {stroke: GRID})]
    }));
    put(el, `<div class="viz-legend"><span class="lg"><span class="sw" style="background:#52514e"></span>Wikipedia</span><span class="lg"><span class="sw" style="background:#b0aea6"></span>fiction</span></div>`,
        chart, `<p class="viz-note">Target: 150 per word per genre; tomato has only 50 fiction sentences. Half A is the first ten words (cat to tree), Half B the last ten (dog to stone).</p>`);
  };

  charts.labels = (el, v) => {
    const order = ["O", "A-gen", "A-idiom", "A-def", "X"];
    const rows = v.labels.map(r => ({g: r.genre === "wikipedia" ? "Wikipedia" : "fiction", l: r.label, n: r.n}));
    const chart = Plot.plot(base(el, {
      height: 120, marginLeft: 80,
      x: {label: "sentences", grid: true}, y: {domain: ["Wikipedia", "fiction"], label: null},
      color: {domain: order, range: order.map(l => MODE[l])},
      marks: [Plot.barX(rows, Plot.stackX({x: "n", y: "g", fill: "l", order, stroke: "white", strokeWidth: 2,
                               tip: true, title: d => `${d.g}: ${d.n} ${d.l}`}))]
    }));
    put(el, modeLegend(order), chart,
        `<p class="viz-note">Wikipedia has far more general statements than fiction: among O and A-gen sentences, 39% are A-gen in Wikipedia and 12% in fiction. A genre detector could pass for a mode detector; E3 tests that. X (excluded) is mostly nouns used as modifiers or materials, then other senses and parts of speech.</p>`);
  };

  charts.kappa = (el, v) => {
    const rows = v.kappa.map(r => ({...r, pass: r.kappa >= v.kappa_min}));
    const chart = Plot.plot(base(el, {
      height: 60 + 34 * rows.length, marginLeft: 100,
      x: {domain: [0, 1], label: "Cohen's κ against the 300-sentence sample", grid: true},
      y: {domain: rows.map(r => r.round), label: null},
      marks: [
        Plot.rectX([{x1: v.kappa_min, x2: 1}], {x1: "x1", x2: "x2", fill: "#0ca30c", fillOpacity: 0.07}),
        Plot.ruleX([v.kappa_min], {stroke: "#0ca30c", strokeWidth: 1.5}),
        Plot.text([v.kappa_min], {x: d => d, frameAnchor: "top", text: () => `gate ${v.kappa_min}`, dx: 4, textAnchor: "start", fill: "#006300"}),
        Plot.dot(rows, {x: "kappa", y: "round", r: 7, fill: d => d.pass ? "#0ca30c" : "#d03b3b", stroke: "white", strokeWidth: 2,
                        tip: true, title: d => `${d.round}: ${d.labeler}, ${d.criterion}, κ = ${f3(d.kappa)}`}),
        Plot.text(rows, {x: "kappa", y: "round", text: d => f2(d.kappa), dy: -14, fill: INK, fontWeight: 600})
      ]
    }));
    const list = rows.map(r => `<li><span class="kp ${r.pass ? "kp-ok" : "kp-no"}">${r.pass ? "✓ passed" : "✕ failed"}</span>
      <strong>${r.round}</strong> · ${r.labeler} · ${r.criterion} · κ = ${f2(r.kappa)}</li>`).join("");
    put(el, chart, `<ul class="kappa-list">${list}</ul>`);
  };

  // ------------------------------------------------------------ rung charts
  charts.e1 = (el, v) => {
    const rows = v.e1.filter(r => r.word !== "(pooled)");
    const pooled = v.e1.filter(r => r.word === "(pooled)");
    const ms = modelsIn(rows);
    const chart = Plot.plot(base(el, {
      height: 20 * WORDS.length + 70,
      x: {domain: [0.5, 1], label: "AUC of the best of the first 10 unlabeled directions", grid: true},
      y: {domain: WORDS.concat(["all words"]), label: null},
      marks: [
        Plot.rectX([{a: 0.8, b: 1}], {x1: "a", x2: "b", fill: "#0ca30c", fillOpacity: 0.06}),
        Plot.ruleX([0.8], {stroke: "#0ca30c"}),
        Plot.dot(rows, {x: "best_auc", y: "word", fill: d => mark(d.model).color, symbol: d => mark(d.model).symbol, r: 4.5,
                        stroke: "white", strokeWidth: 1, tip: true,
                        title: d => `${d.word}, ${MODEL[d.model].name}: best direction PC${d.best_pc}, AUC ${f2(d.best_auc)}`}),
        Plot.dot(pooled, {x: "best_auc", y: () => "all words", fill: d => mark(d.model).color, symbol: d => mark(d.model).symbol, r: 6,
                          stroke: "white", strokeWidth: 1, tip: true, title: d => `all words pooled, ${MODEL[d.model].name}: AUC ${f2(d.best_auc)}`})
      ]
    }));
    put(el, legend(ms), chart, `<p class="viz-note">Each dot: the best of a word's first ten unlabeled directions, scored by how well it separates object-mode from general statements. In the green zone the mode is a main direction of that word's variation. Pooled over all words (bottom row), the best direction is clearly above chance and under the bar: the mark is there, and thin.</p>`);
  };

  charts.e2 = (el, v) => {
    const rows = v.e2w;
    const ms = modelsIn(rows);
    const nulls = WORDS.map(w => {
      const r = rows.filter(d => d.word === w && d.null95 != null);
      return {word: w, lo: 0.5, hi: Math.max(...r.map(d => d.null95))};
    });
    const chart = Plot.plot(base(el, {
      height: 20 * WORDS.length + 50,
      x: {domain: [0.4, 1], label: "AUC on a word the probe never saw (leave-one-word-out)", grid: true},
      y: {domain: WORDS, label: null},
      marks: [
        Plot.barX(nulls, {x1: "lo", x2: "hi", y: "word", fill: "#e1e0d9", insetTop: 5, insetBottom: 5}),
        Plot.ruleX([0.5], {stroke: MUTED}),
        Plot.ruleX([0.7], {stroke: "#0ca30c"}),
        Plot.dot(rows, {x: "lowo_auc", y: "word", fill: d => mark(d.model).color, symbol: d => mark(d.model).symbol, r: 4.5,
                        stroke: "white", strokeWidth: 1, tip: true,
                        title: d => `${d.word}, ${MODEL[d.model].name}: AUC ${f3(d.lowo_auc)} (null 95th ${f2(d.null95)}; ${d.n_O} O, ${d.n_Agen} A-gen)`})
      ]
    }));
    put(el, legend(ms, `<span class="lg"><span class="sw" style="background:#e1e0d9"></span>chance range (permutation null, 95th percentile)</span>`), chart,
        `<p class="viz-note">Each dot is one word held out entirely from training. Green line: the 0.7 bar for “supported”.</p>`);
  };

  function dumbbell(el, rows, from, to, labelFrom, labelTo, xlab, dom) {
    const chart = Plot.plot(base(el, {
      height: 44 * rows.length + 50, marginLeft: 150,
      x: {domain: dom, label: xlab, grid: true}, y: {domain: rows.map(r => r.y), label: null},
      marks: [
        Plot.link(rows, {x1: from, x2: to, y1: "y", y2: "y", stroke: d => mark(d.model).color, strokeWidth: 2}),
        Plot.dot(rows, {x: from, y: "y", r: 5, fill: "white", stroke: d => mark(d.model).color, strokeWidth: 2,
                        tip: true, title: d => `${d.y}: ${labelFrom} ${f3(d[from])}`}),
        Plot.dot(rows, {x: to, y: "y", r: 5.5, fill: d => mark(d.model).color, symbol: d => mark(d.model).symbol,
                        stroke: "white", strokeWidth: 1, tip: true, title: d => `${d.y}: ${labelTo} ${f3(d[to])}`})
      ]
    }));
    return chart;
  }

  charts.e3 = (el, v) => {
    const rows = [];
    v.e3.forEach(r => {
      rows.push({model: r.model, y: `${MODEL[r.model].name}: Wiki → fiction`, within: r["within-fiction auc"], cross: r["wiki->fiction auc"]});
      rows.push({model: r.model, y: `${MODEL[r.model].name}: fiction → Wiki`, within: r["within-wiki auc"], cross: r["fiction->wiki auc"]});
    });
    put(el, `<div class="viz-legend"><span class="lg"><svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="white" stroke="#52514e" stroke-width="2"/></svg>trained and tested within the target genre</span><span class="lg"><svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="#52514e"/></svg>trained on the other genre</span></div>`,
        dumbbell(el, rows, "within", "cross", "within genre", "across genres", "AUC", [0.75, 1]),
        `<p class="viz-note">The short gap between the hollow and the filled dot is what crossing genres costs. A genre detector would fall to chance here.</p>`);
  };

  charts.e4 = (el, v) => {
    const rows = v.e4.map(r => ({model: r.model, y: MODEL[r.model].name, gram: r["grammar LOWO auc"], emb: r["embedding LOWO auc"],
                                  frac: r["pairs fraction positive"], ties: r["pairs ties"]}));
    const pairs = Plot.plot(base(el, {
      height: 44 * rows.length + 50, marginLeft: 150,
      x: {domain: [0, 1], label: "share of the 200 minimal pairs placed the right way round", grid: true, tickFormat: "%"},
      y: {domain: rows.map(r => r.y), label: null},
      marks: [
        Plot.barX(rows, {x: 1, y: "y", fill: "#efeeea", insetTop: 12, insetBottom: 12}),
        Plot.barX(rows, {x: "frac", y: "y", fill: d => mark(d.model).color, insetTop: 12, insetBottom: 12,
                         tip: true, title: d => d.ties ? `${d.y}: all ${d.ties} pairs are ties — both sentences are identical up to the target word, so this readout cannot tell them apart` : `${d.y}: ${(d.frac * 100).toFixed(1)}%`}),
        Plot.ruleX([0.5], {stroke: MUTED}),
        Plot.text(rows, {x: d => Math.max(d.frac, 0), y: "y", text: d => d.ties ? "untestable (identical prefixes)" : `${Math.round(d.frac * 100)}%`,
                         dx: 6, textAnchor: "start", fill: INK})
      ]
    }));
    put(el, `<div class="viz-legend"><span class="lg"><svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="white" stroke="#52514e" stroke-width="2"/></svg>grammar-only classifier</span><span class="lg"><svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="#52514e"/></svg>embedding probe</span></div>`,
        dumbbell(el, rows, "gram", "emb", "grammar only", "embedding", "AUC on held-out words", [0.7, 1]),
        `<h4 class="viz-h">Minimal pairs: same words up to the predicate</h4>`, pairs);
  };

  charts.e5 = (el, v) => {
    const rows = v.e5.map(r => ({model: r.model, y: MODEL[r.model].name, before: r["auc before"], after: r["auc after (features residualized)"]}));
    put(el, `<div class="viz-legend"><span class="lg"><svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="white" stroke="#52514e" stroke-width="2"/></svg>before</span><span class="lg"><svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="#52514e"/></svg>after removing concreteness</span></div>`,
        dumbbell(el, rows, "before", "after", "before", "after", "AUC", [0.85, 1]),
        `<p class="viz-note">The dots almost coincide: concreteness of the surrounding words carries almost none of the mark.</p>`);
  };

  charts.e6 = (el, v) => {
    const order = ["O", "A-idiom", "A-gen", "A-def"];
    const rows = v.dist.filter(d => order.includes(d.label));
    const ms = modelsIn(rows);
    const chart = Plot.plot(base(el, {
      height: 70 * ms.length + 50, marginLeft: 70,
      x: {label: "probe score (trained on O vs A-gen only) ← object · abstract →", grid: true},
      fy: {domain: ms, label: null, tickFormat: m => MODEL[m].name}, y: {domain: order, axis: null},
      marks: [
        Plot.ruleY(rows, {x1: "q10", x2: "q90", y: "label", fy: "model", stroke: d => MODE[d.label], strokeOpacity: 0.5}),
        Plot.barX(rows, {x1: "q25", x2: "q75", y: "label", fy: "model", fill: d => MODE[d.label], insetTop: 2, insetBottom: 2,
                         tip: true, title: d => `${MODEL[d.model].name}, ${d.label} (n=${d.n}): median ${f2(d.q50)}, middle half ${f2(d.q25)} to ${f2(d.q75)}`}),
        Plot.tickX(rows, {x: "q50", y: "label", fy: "model", stroke: "white", strokeWidth: 2})
      ]
    }));
    put(el, modeLegend(order), chart,
        `<p class="viz-note">Bars: the middle half of each group's scores; whiskers: 10th to 90th percentile; white tick: median. Only ${(rows.find(d => d.label === "A-def") || {n: 0}).n} definitions occurred, too few to decide.</p>`);
  };

  charts.e7 = (el, v) => {
    const rows = [];
    v.e7.forEach(r => {
      const pair = r.pair.split(" vs ").map(m => MODEL[m] ? MODEL[m].name : m).join(" · ");
      rows.push({pair, k: "all sentences", rho: r.spearman});
      rows.push({pair, k: "within O", rho: r["spearman within O"]});
      rows.push({pair, k: "within A-gen", rho: r["spearman within A-gen"]});
    });
    const col = {"all sentences": "#111111", "within O": MODE["O"], "within A-gen": MODE["A-gen"]};
    const sym = {"all sentences": "circle", "within O": "square", "within A-gen": "diamond"};
    const chart = Plot.plot(base(el, {
      height: 46 * v.e7.length + 50, marginLeft: 150,
      x: {domain: [0, 1], label: "Spearman ρ between the two models' scores (Half-B sentences)", grid: true},
      y: {domain: [...new Set(rows.map(r => r.pair))], label: null},
      marks: [
        Plot.ruleX([0.5], {stroke: "#0ca30c"}),
        Plot.dot(rows, {x: "rho", y: "pair", fill: d => col[d.k], symbol: d => sym[d.k], r: 5.5, stroke: "white", strokeWidth: 1,
                        tip: true, title: d => `${d.pair}, ${d.k}: ρ = ${f2(d.rho)}`})
      ]
    }));
    put(el, `<div class="viz-legend">${Object.keys(col).map(k => `<span class="lg"><span class="sw" style="background:${col[k]}"></span>${k}</span>`).join("")}</div>`,
        chart, `<p class="viz-note">Green line: the 0.5 bar for “supported”. Agreement inside each class shows the models rank sentences alike beyond the O/A-gen split itself.</p>`);
  };

  async function init() {
    const els = document.querySelectorAll("[data-chart]");
    if (!els.length) return;
    const cache = {};
    for (const el of els) {
      const src = el.dataset.src || "data/viz.json";
      try {
        cache[src] = cache[src] || await (await fetch(src)).json();
        const fn = charts[el.dataset.chart];
        if (fn) fn(el, cache[src]);
      } catch (e) {
        el.innerHTML = `<p class="viz-note">Chart unavailable (${e.message}).</p>`;
      }
    }
  }
  document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", init) : init();
})();
