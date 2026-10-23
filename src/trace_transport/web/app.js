"use strict";
const $ = id => document.getElementById(id);
const colors = ["#67e7c1", "#80b8ff", "#f1bd74", "#c19bff", "#ed95b5", "#9bdc89", "#78c9d5", "#dec883"];
let catalog, job, timer, polling = false;
const fmt = (value, digits = 1) => value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
const percent = (a, b) => b ? 100 * a / b : 0;
function message(text) { $("error").hidden = !text; $("error").textContent = text || ""; }
async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function preset() {
  const scenario = catalog.scenarios.find(item => item.name === $("scenario").value);
  const n = scenario.network;
  $("scenario-description").textContent = scenario.description;
  for (const [id, value] of Object.entries({flows: scenario.flows, payload: scenario.message_bytes / 1024,
    bandwidth: n.bandwidth_bps / 1e6, rtt: n.base_rtt * 1000, jitter: n.jitter * 1000,
    loss: n.loss * 100, "ack-loss": n.ack_loss * 100, corruption: n.corruption * 100,
    "ack-corruption": n.ack_corruption * 100, queue: n.queue_bytes / 1024, deadline: n.deadline})) $(id).value = value;
}
function plan() {
  const number = id => Number($(id).value);
  const data = {scenario: $("scenario").value, trials: number("trials"), flows: number("flows"),
    seed: number("seed"), message_bytes: Math.round(number("payload") * 1024),
    controllers: ["fixed", "aimd", "learned"].filter(id => $(id).checked),
    checkpoint: $("checkpoint").value || null, reward: $("reward").value,
    network: {bandwidth_bps: number("bandwidth") * 1e6, base_rtt: number("rtt") / 1000,
      jitter: number("jitter") / 1000, loss: number("loss") / 100, ack_loss: number("ack-loss") / 100,
      corruption: number("corruption") / 100, ack_corruption: number("ack-corruption") / 100,
      queue_bytes: Math.round(number("queue") * 1024), deadline: number("deadline")}};
  if ($("use-text").checked) data.text = $("text").value;
  return data;
}
function cell(row, text, className) { const td = row.insertCell(); td.textContent = text; if (className) td.className = className; return td; }
function renderComparison(rows) {
  const body = $("comparison");
  if (!rows.length) return;
  body.replaceChildren();
  for (const item of rows) {
    const row = body.insertRow();
    cell(row, item.controller === "learned" ? "Learned policy" : item.controller === "fixed" ? "Fixed rate" : "AIMD", "policy-name");
    cell(row, `${item.completed_transfers}/${item.transfers} · ${fmt(item.completion_percent, 0)}%`);
    cell(row, `${fmt(item.exact_receiver_percent, 0)}%`);
    cell(row, `${fmt(item.drop_percent)}%`);
    cell(row, `${fmt(item.retransmission_percent)}%`);
    cell(row, item.completion_p50_seconds == null ? "—" : `${fmt(item.completion_p50_seconds, 3)}s`);
    cell(row, fmt(item.mean_discounted_rl_return, 3), "score");
    row.title = "Click to inspect the latest completed trial for this controller";
    row.tabIndex = 0;
    const inspect = () => { const trace = [...job.traces].reverse().find(t => t.controller === item.controller); if (trace) { $("trace").value = String(trace.index); inspectTrace(); } };
    row.onclick = inspect;
    row.onkeydown = event => { if (event.key === "Enter") inspect(); };
  }
}
function renderFlows(current) {
  if (!current) return;
  $("senders-title").textContent = `${current.flows.length} SENDERS`;
  const rate = current.flows.reduce((sum, flow) => sum + flow.rate_bytes_per_second, 0);
  $("senders-info").textContent = `${fmt(rate / 1024)} KiB/s combined`;
  const drops = Object.values(current.network).reduce((sum, link) => sum + link.queue_drops + link.random_drops + link.outage_drops, 0);
  $("drops-info").textContent = `${drops} dropped datagrams this trial`;
  $("received-info").textContent = `${current.flows.filter(flow => flow.receiver_complete).length}/${current.flows.length} messages reconstructed`;
  $("current-run").textContent = `${current.controller} · seed ${current.seed} · t=${fmt(current.time, 2)}s`;
  const container = $("flow-bars"); container.replaceChildren();
  current.flows.forEach((flow, index) => {
    const fraction = flow.payload_bytes ? percent(flow.acknowledged_bytes, flow.payload_bytes) : flow.complete ? 100 : 0;
    const element = document.createElement("div"); element.className = "flow";
    const label = document.createElement("div"); label.className = "flow-label";
    const title = document.createElement("span"); title.textContent = `SENDER ${index + 1}`;
    const detail = document.createElement("span"); detail.textContent = `${fmt(fraction, 0)}% ACKed · ${fmt(flow.rtt_seconds * 1000, 0)} ms RTT`;
    label.append(title, detail);
    const track = document.createElement("div"); track.className = "flow-track";
    const fill = document.createElement("div"); fill.style.width = `${fraction}%`; fill.style.background = colors[index % colors.length]; track.append(fill);
    element.append(label, track); container.append(element);
  });
}
function chart(id, points, value, fixedMaximum) {
  const svg = $(id), NS = "http://www.w3.org/2000/svg";
  svg.replaceChildren();
  const draw = (type, attributes, text) => { const element = document.createElementNS(NS, type); for (const [key, item] of Object.entries(attributes)) element.setAttribute(key, item); if (text != null) element.textContent = text; svg.append(element); return element; };
  const width = 480, height = 180, left = 38, top = 14, right = 12, bottom = 30;
  const xmax = points.length ? Math.max(.1, points[points.length - 1].time) : 1;
  const values = points.flatMap(point => point.flows.map(value));
  const ymax = fixedMaximum || Math.max(1, ...values);
  for (let i = 0; i <= 4; i++) {
    const y = top + (height - top - bottom) * i / 4;
    draw("line", {x1: left, x2: width - right, y1: y, y2: y, stroke: "#283743", "stroke-dasharray": "3 4"});
    draw("text", {x: left - 7, y: y + 3, "text-anchor": "end"}, fmt(ymax * (1 - i / 4), 0));
  }
  for (let i = 0; i <= 4; i++) draw("text", {x: left + (width - left - right) * i / 4, y: height - 8, "text-anchor": "middle"}, fmt(xmax * i / 4));
  if (!points.length) { draw("text", {x: width / 2, y: height / 2, "text-anchor": "middle"}, "A real trial trace will appear here"); return; }
  for (let flow = 0; flow < points[0].flows.length; flow++) {
    const d = points.map((point, index) => `${index ? "L" : "M"}${left + point.time / xmax * (width - left - right)},${top + (1 - value(point.flows[flow]) / ymax) * (height - top - bottom)}`).join(" ");
    draw("path", {d, fill: "none", stroke: colors[flow % colors.length], "stroke-width": "2", "stroke-linejoin": "round"});
    const last = points[points.length - 1];
    draw("circle", {cx: left + last.time / xmax * (width - left - right), cy: top + (1 - value(last.flows[flow]) / ymax) * (height - top - bottom), r: 2.5, fill: colors[flow % colors.length]});
  }
}
function renderCharts(points) {
  chart("delivery-chart", points, flow => flow.payload_bytes ? percent(flow.acknowledged_bytes, flow.payload_bytes) : flow.complete ? 100 : 0, 100);
  chart("rate-chart", points, flow => flow.rate_bytes_per_second / 1024);
}
function render(state) {
  clearTimeout(timer);
  job = state;
  const running = ["queued", "running", "stopping"].includes(state.status);
  $("run").disabled = running; $("run").textContent = running ? "Comparison running…" : "▶ Run comparison";
  $("stop").hidden = !running; $("stop").disabled = state.status === "stopping";
  $("status").textContent = state.status[0].toUpperCase() + state.status.slice(1);
  $("status").className = `status ${state.status}`;
  $("batch-title").textContent = `${state.finished_runs} / ${state.total_runs} controller trials scored`;
  $("batch-detail").textContent = `${state.plan.scenario} · ${state.plan.flows} concurrent senders · ${fmt(state.plan.message_bytes / 1024)} KiB/flow · seeds ${state.plan.seed}–${state.plan.seed + state.plan.trials - 1}`;
  $("batch-progress").style.width = `${percent(state.finished_runs, state.total_runs)}%`;
  $("elapsed").textContent = `${fmt(state.elapsed_wall_seconds)}s wall time`;
  const count = field => state.summary.reduce((sum, row) => sum + row[field], 0);
  const tested = count("transfers"), complete = count("completed_transfers");
  $("completion").textContent = tested ? `${fmt(percent(complete, tested), 0)}%` : "—";
  $("completion-detail").textContent = `${complete} / ${tested} verified transfers scored`;
  $("crc").textContent = tested ? String(count("corruption_rejections")) : "—";
  const weightedRetry = tested ? percent(count("retransmitted_payload_bytes"), count("payload_sent_bytes")) : null;
  $("retry").textContent = weightedRetry == null ? "—" : `${fmt(weightedRetry)}%`;
  $("wrong").textContent = tested ? String(count("wrong_payloads")) : "—";
  renderComparison(state.summary); renderFlows(state.current);
  const traceSelect = $("trace"), previous = traceSelect.value;
  traceSelect.replaceChildren(new Option("Follow current trial", "live"));
  for (const trace of state.traces) traceSelect.add(new Option(`${trace.controller} · seed ${trace.seed}`, String(trace.index)));
  traceSelect.value = [...traceSelect.options].some(option => option.value === previous) ? previous : "live";
  if (traceSelect.value === "live") {
    renderCharts(state.points);
    $("trace-caption").textContent = state.current ? `${state.current.controller} · seed ${state.current.seed}. Each line is a sender; x-axis is simulated seconds.` : "Each line is a sender. The x-axis is simulated seconds.";
  }
  for (const kind of ["json", "csv"]) { const link = $(`download-${kind}`); link.href = `/api/jobs/${state.id}/report.${kind}`; link.classList.remove("disabled"); }
  $("scoring-note").textContent = state.plan.reward === "completion" ? "Scoring: transport-v2 · γ 0.999." : "Scoring: transport-v1 · γ 0.99.";
  message(state.error || (state.status === "cancelled" ? "Batch stopped. Completed trials are retained; the interrupted trial was not scored." : ""));
  if (running) timer = setTimeout(poll, 250);
}
async function poll() {
  if (polling || !job) return;
  polling = true;
  try { render(await api(`/api/jobs/${job.id}`)); } catch (error) { message(error.message); $("run").disabled = false; }
  finally { polling = false; }
}
async function inspectTrace() {
  if (!job) return;
  if ($("trace").value === "live") { renderCharts(job.points); $("trace-caption").textContent = "Following the current trial. Each line is a sender; x-axis is simulated seconds."; return; }
  try { const trace = await api(`/api/jobs/${job.id}/trace/${$("trace").value}`); renderCharts(trace.points); $("trace-caption").textContent = `${trace.controller} · seed ${trace.seed}. Each line is a sender; x-axis is simulated seconds.`; }
  catch (error) { message(error.message); }
}
$("setup").onsubmit = async event => {
  event.preventDefault(); message(""); clearTimeout(timer);
  try {
    const state = await api("/api/jobs", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(plan())});
    $("comparison").replaceChildren(); $("trace").value = "live";
    localStorage.setItem("trace-last-job", state.id); render(state);
  } catch (error) { message(error.message); }
};
$("stop").onclick = async () => { try { render(await api(`/api/jobs/${job.id}/cancel`, {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"})); } catch (error) { message(error.message); } };
$("scenario").onchange = preset; $("trace").onchange = inspectTrace;
(async function init() {
  renderCharts([]);
  try {
    catalog = await api("/api/catalog");
    for (const scenario of catalog.scenarios) $("scenario").add(new Option(scenario.name[0].toUpperCase() + scenario.name.slice(1), scenario.name));
    $("scenario").value = "lossy"; preset();
    for (const model of catalog.models) $("checkpoint").add(new Option(model.label, model.id));
    const best = catalog.models.find(model => model.id === "artifacts/robust-v2-best.pt") || catalog.models[0];
    if (best) $("checkpoint").value = best.id;
    $("learned").checked = Boolean(best && catalog.learning_available);
    $("learned").disabled = !best || !catalog.learning_available;
    if (!catalog.learning_available) $("learned-hint").textContent = "Enable with: uv sync --locked --extra learn";
    else if (!best) $("learned-hint").textContent = "Train or add a checkpoint to enable";
    const previous = catalog.active_job || localStorage.getItem("trace-last-job");
    if (previous) { try { render(await api(`/api/jobs/${previous}`)); } catch (_) { localStorage.removeItem("trace-last-job"); } }
  } catch (error) { message(`Cannot connect to the test bench: ${error.message}`); }
})();
