"use strict";
const el = id => document.getElementById(id);
let snapshot;
const number = (x, digits = 1) => x == null ? "—" : Number(x).toLocaleString(undefined, {maximumFractionDigits: digits});
function chart(parent, history) {
  const points = history.filter(x => x.miss != null);
  if (!points.length) return;
  const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 300 90"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Recent training deadline miss fraction");
  const start = points[0].decisions, span = Math.max(1, points[points.length - 1].decisions - start);
  const line = document.createElementNS(ns, "polyline");
  line.setAttribute("points", points.map(p => `${5 + 290 * (p.decisions - start) / span},${5 + 70 * (1 - Math.max(0, Math.min(1, p.miss)))}`).join(" "));
  line.setAttribute("fill", "none"); line.setAttribute("stroke", "#67e7c1"); line.setAttribute("stroke-width", "2"); svg.append(line);
  const text = document.createElementNS(ns, "text");text.setAttribute("x", "5");text.setAttribute("y", "88");text.setAttribute("fill", "#a8b8ca");text.setAttribute("font-size", "10");text.textContent = "Recent training miss rate · lower is better";svg.append(text);parent.append(svg);
}
function renderRuns() {
  const root = el("joint-runs");root.replaceChildren();
  if (!snapshot.runs.length) { const p = document.createElement("p");p.className = "joint-empty";p.textContent = "No joint runs yet. Start joint-train from the terminal.";root.append(p); }
  for (const run of snapshot.runs) {
    const card = document.createElement("article");card.className = "panel joint-card";
    const h = document.createElement("h2");h.textContent = run.name;card.append(h);
    const p = run.progress, target = run.config.training?.decisions || 1, progress = document.createElement("progress");progress.max = target;progress.value = p.decisions || 0;card.append(progress);
    for (const [left, right] of [[run.phase, `${number(p.decisions, 0)} / ${number(target, 0)} decisions`], ["Recent deadline misses", p.recent_miss_rate == null ? "Waiting for completed sessions" : `${number(p.recent_miss_rate * 100)}%`], ["Throughput", `${number(p.decisions_per_second, 0)} decisions/s`]]) {
      const row = document.createElement("div");row.className = "joint-meta";const a = document.createElement("span"), b = document.createElement("span");a.textContent = left;b.textContent = right;row.append(a, b);card.append(row);
    }
    chart(card, run.history);
    const details = document.createElement("details"), summary = document.createElement("summary"), diag = document.createElement("p");summary.textContent = "Joint policy diagnostics";diag.className = "hint";diag.textContent = `Pre-clip gradient norms: sender ${number(p.sender?.gradient_norm, 3)}, receiver ${number(p.receiver?.gradient_norm, 3)}. Samples in last rollout: ${number(p.sender?.samples, 0)} / ${number(p.receiver?.samples, 0)}.`;details.append(summary, diag);card.append(details);root.append(card);
  }
}
function renderResults() {
  const suite = el("joint-suite").value, rows = snapshot.tables[suite] || [], root = el("joint-table");root.replaceChildren();el("joint-comparison").textContent = "";
  el("joint-study").textContent = `Study: ${snapshot.study.phase || "not started"}${snapshot.study.run ? ` · ${snapshot.study.run}` : ""}${snapshot.study.case_total ? ` · ${snapshot.study.case_completed}/${snapshot.study.case_total} cases` : ""}. Reference: ${snapshot.baseline || "selected on validation"}.`;
  const selected = snapshot.selection.checkpoint, initial = selected?.endsWith("/initial.pt");
  el("joint-selection").textContent = selected ? `Selected checkpoint: ${selected}.${initial ? " Untrained baseline-following policy: training did not improve validation ranking." : ""}` : "Checkpoint selection pending.";
  if (!rows.length) { const p = document.createElement("p");p.className = "joint-empty";p.textContent = "Results will appear here after training and validation finish.";root.append(p);return; }
  const table = document.createElement("table"), header = table.createTHead().insertRow();
  for (const text of ["Policy", "Late interactive", "Image MiB/s", "Map MiB/s", "Confirmed / admitted", "Wire / delivered byte", "Mean session p95"]) {const cell = document.createElement("th");cell.textContent = text;header.append(cell);}
  const body = table.createTBody();
  for (const row of rows) {const tr = body.insertRow();for (const value of [row.name === "joint" ? (initial ? "selected initial policy" : "selected joint policy") : row.name, `${number(row.interactive_miss_rate * 100)}%`, number(row.image_goodput / 1048576, 3), number(row.map_goodput / 1048576, 3), `${row.completed} / ${row.admitted}`, number(row.wire_per_delivered_byte, 2), row.mean_session_p95 == null ? "—" : `${number(row.mean_session_p95 * 1000)} ms`]) tr.insertCell().textContent = value;}
  root.append(table);el("joint-download").hidden = false;
  const cmp = snapshot.comparisons[suite];el("joint-comparison").textContent = cmp ? `Predeclared win condition: ${cmp.win ? "met" : "not met"}. Deadline misses count desired interactive opportunities, including backpressure. Bulk delivery and total-wire guards also apply.${initial ? " The selected policy is untrained; this is not evidence of a learning gain." : ""}` : "Evaluation in progress.";
}
async function refresh() {
  try {const response = await fetch("/api/joint/status");if (!response.ok) throw new Error(`Status ${response.status}`);snapshot = await response.json();renderRuns();renderResults();el("joint-status").textContent = snapshot.runs.some(r => r.phase === "training") ? "Training" : "Local results";el("joint-error").hidden = true;}
  catch (error) {el("joint-error").hidden = false;el("joint-error").textContent = error.message;}
  setTimeout(refresh, 5000);
}
el("joint-suite").onchange = () => snapshot && renderResults();refresh();
