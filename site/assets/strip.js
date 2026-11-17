// Interactive strips for rungs 2 and 6: Half-B sentences on the probe axis,
// scored by a probe trained on Half A. Hover a point to read the sentence.
(function () {
  const NAMES = {bert: "BERT", decoder: "Decoder", decoder_target: "Decoder, at the word", mpnet: "mpnet"};
  const COLOR = {"O": "#1f4e79", "A-gen": "#d9822b", "A-def": "#000000", "A-idiom": "#7f7f7f"};
  const LABELS = ["O", "A-gen", "A-def", "A-idiom"];
  const HALF_B = ["dog", "fork", "tomato", "mountain", "door", "cow", "lamp", "road", "spoon", "stone"];

  function hash(s) { let h = 0; for (const c of s) h = (h * 31 + c.charCodeAt(0)) | 0; return h; }
  function jitter(s) { return ((Math.abs(hash(s)) % 1000) / 1000 - 0.5) * 0.5; }

  function draw(el, data, model) {
    const view = el.dataset.view;
    const key = model + "_swap";
    const rows = data.rows.filter(r => r.h === "B" && LABELS.includes(r.l) && r[key] !== null);
    const cats = view === "words" ? HALF_B : LABELS;
    const pts = rows.map(r => ({
      x: r[key], y: cats.indexOf(view === "words" ? r.w : r.l) + jitter(r.sid),
      l: r.l, w: r.w, s: r.s, gl: r.gl || "—"
    }));
    const width = Math.max(320, Math.min(el.clientWidth || 700, 900));
    const chart = Plot.plot({
      width, height: 26 * cats.length + 60, marginLeft: 70,
      x: {label: "probe score (higher = further toward abstract-mode)", grid: true},
      y: {domain: [cats.length - 0.5, -0.5], ticks: cats.map((_, i) => i),
          tickFormat: i => cats[i], label: null},
      marks: [
        Plot.ruleX([0], {stroke: "#bbb", strokeDasharray: "2,3"}),
        Plot.dot(pts.filter(p => p.l === "O" || p.l === "A-gen"),
                 {x: "x", y: "y", r: 2.4, fill: p => COLOR[p.l], fillOpacity: 0.65}),
        Plot.dot(pts.filter(p => p.l === "A-def"),
                 {x: "x", y: "y", r: 3.4, stroke: "#000", fill: "none", strokeWidth: 1}),
        Plot.dot(pts.filter(p => p.l === "A-idiom"),
                 {x: "x", y: "y", r: 3.6, symbol: "triangle", fill: COLOR["A-idiom"]}),
        Plot.tip(pts, Plot.pointer({x: "x", y: "y",
          title: p => `${p.s}\n\nword: ${p.w} · label: ${p.l} · second labeler: ${p.gl}\nscore: ${p.x.toFixed(2)}`,
          maxRadius: 12}))
      ]
    });
    el.replaceChildren(chart);
  }

  async function init() {
    const els = document.querySelectorAll(".strip");
    if (!els.length) return;
    const data = await (await fetch("../data/sentences.json")).json();
    const models = data.models.filter(m => data.rows.some(r => r[m + "_swap"] !== null));
    document.querySelectorAll(".strip-model").forEach(sel => {
      sel.innerHTML = models.map(m => `<option value="${m}">${NAMES[m] || m}</option>`).join("");
      sel.addEventListener("change", () => els.forEach(el => draw(el, data, sel.value)));
    });
    els.forEach(el => draw(el, data, models[0]));
  }
  document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", init) : init();
})();
