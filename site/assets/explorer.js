// Sentence explorer: filter, sort and read every sentence with its labels and scores.
(function () {
  const NAMES = {bert: "BERT", decoder: "Decoder", decoder_final: "Decoder, final token", mpnet: "mpnet"};
  const LABELS = ["O", "A-gen", "A-def", "A-idiom", "X"];
  const PAGE = 50;
  const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const opt = (v, t) => `<option value="${esc(v)}">${esc(t)}</option>`;

  async function init() {
    const root = document.getElementById("explorer");
    const data = await (await fetch("data/sentences.json")).json();
    const words = [...new Set(data.rows.map(r => r.w))];
    const models = data.models.filter(m => data.rows.some(r => r[m + "_swap"] !== null));
    root.innerHTML = `
      <div class="filters">
        <label>word<select id="f-w">${opt("", "all")}${words.map(w => opt(w, w)).join("")}</select></label>
        <label>genre<select id="f-g">${opt("", "both")}${opt("wikipedia", "Wikipedia")}${opt("fiction", "fiction")}</select></label>
        <label>label<select id="f-l">${opt("", "all")}${LABELS.map(l => opt(l, l)).join("")}</select></label>
        <label>second labeler<select id="f-gl">${opt("", "any")}${opt("*", "in sample")}${opt("disagree", "disagrees")}${LABELS.map(l => opt(l, l)).join("")}</select></label>
        <label>score from<select id="f-m">${models.map(m => opt(m, NAMES[m] || m)).join("")}</select></label>
        <label>sort<select id="f-s">${opt("", "as harvested")}${opt("asc", "score, low to high")}${opt("desc", "score, high to low")}</select></label>
        <label>contains<input id="f-t" type="search" size="14"></label>
      </div>
      <div class="pager" id="pager"></div>
      <table class="table sent-table"><thead><tr><th>word</th><th>sentence</th><th>label</th><th>second</th><th>score</th></tr></thead>
      <tbody id="rows"></tbody></table>`;
    let page = 0;
    const $ = id => document.getElementById(id);

    function filtered() {
      const w = $("f-w").value, g = $("f-g").value, l = $("f-l").value, gl = $("f-gl").value;
      const t = $("f-t").value.toLowerCase(), m = $("f-m").value + "_swap", s = $("f-s").value;
      let rows = data.rows.filter(r =>
        (!w || r.w === w) && (!g || r.g === g) && (!l || r.l === l) &&
        (!gl || (gl === "*" ? r.gl : gl === "disagree" ? (r.gl && r.gl !== r.l) : r.gl === gl)) &&
        (!t || r.s.toLowerCase().includes(t)));
      if (s) rows = rows.filter(r => r[m] !== null).sort((a, b) => s === "asc" ? a[m] - b[m] : b[m] - a[m]);
      return {rows, m};
    }

    function render() {
      const {rows, m} = filtered();
      const pages = Math.max(1, Math.ceil(rows.length / PAGE));
      page = Math.min(page, pages - 1);
      $("pager").innerHTML = `${rows.length} sentences · page ${page + 1} of ${pages}
        <button id="prev" ${page === 0 ? "disabled" : ""}>previous</button>
        <button id="next" ${page >= pages - 1 ? "disabled" : ""}>next</button>`;
      $("prev").onclick = () => { page--; render(); };
      $("next").onclick = () => { page++; render(); };
      $("rows").innerHTML = rows.slice(page * PAGE, (page + 1) * PAGE).map(r => {
        const src = r.url ? `<a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.src)}</a>` : esc(r.src);
        const sc = r[m] === null ? "" : r[m].toFixed(2);
        return `<tr><td>${esc(r.w)}<br><small>${esc(r.g)}</small></td>
          <td class="s">${esc(r.s)}<br><small>${src}</small></td>
          <td class="lab lab-${esc(r.l)}">${esc(r.l)}</td>
          <td class="lab lab-${esc(r.gl)}">${esc(r.gl || "")}</td><td>${sc}</td></tr>`;
      }).join("");
    }
    root.querySelectorAll("select, input").forEach(el => el.addEventListener("input", () => { page = 0; render(); }));
    render();
  }
  document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", init) : init();
})();
