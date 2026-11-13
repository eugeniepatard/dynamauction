/* DynamAuction website: interactive results chart (main scenario, mean campaign length 50).
 * Data: Table "main results" of the paper (grouped by family; a second chart shows MPC for each horizon H) (mean episode reward ± 95% CI over 1 000 test campaigns, and paired
 * difference with Shading(α = 0.4, M = 20)). Clicking a bar opens a window explaining the policy. */
(function () {
  "use strict";

  const FAMILY = {
    mpc:      { name: "MPC (ours)",       color: "var(--accent)" },
    shading:  { name: "Tuned shading",    color: "var(--bid)" },
    rl:       { name: "Deep RL",          color: "var(--lime)" },
    myopic:   { name: "Myopic",           color: "var(--myopic)" },
    constant: { name: "Constant",         color: "var(--line-strong)" },
  };

  const ABOUT = {
    mpc: {
      how: "Model Predictive Control with myopic valuations. At every auction, MPC tries the multipliers α ∈ {0.1, …, 1} of a myopic value V̂₀ predicted by a small neural network. For each α it simulates M possible futures over a window of H time units (bidding α·V̂₀ now, then α_cont·V̂₀), keeps the α with the best simulated reward, bids α·V̂₀, and plans again at the next auction.",
      why: "The simulated futures carry the user's fatigue, so MPC sees that winning now makes the next displays worth less, and can bid below the myopic value when it pays.",
    },
    shading: {
      how: "Bids a fixed fraction α of the value of the current display. The value is estimated by simulation: M pairs of futures, one where the auction is won and one where it is lost, both continued with a zero bid and sharing their random numbers; their mean difference is the value of the display.",
      why: "α = 0.4 is the best of a sweep over {0.1, …, 1}. A well-chosen constant fraction already accounts for much of the cost of fatigue.",
    },
    greedy: {
      how: "Shading with α = 1: bids the full simulated value of the current display.",
      why: "It ignores how a win today lowers the value of tomorrow's displays, so it wins too often and tires the user.",
    },
    fipla: {
      how: "Bids the oracle myopic value V₀: the expected reward of the current display alone, given the current fatigue. This is optimal for a single isolated second-price auction.",
      why: "Like GreedyMyopic, it is blind to the effect of a win on the future displays.",
    },
    constant: {
      how: "Bids the same amount at every auction, whatever the user, the display or the fatigue.",
      why: "b = 0.11 is the best of a two-stage sweep (step 0.1, then 0.01).",
    },
    ppo: {
      how: "Proximal Policy Optimization (Stable-Baselines3): an on-policy actor-critic that learns a bidding policy from interaction with the environment, with clipped policy updates. It never sees the oracle value V₀.",
      why: "Its results vary with the training seed (3.52, 3.53 and 1.84 for its three trainings, same hyperparameters, selected by a grid search on validation campaigns). The bar is their mean.",
    },
    sac: {
      how: "Soft Actor-Critic (Stable-Baselines3): an off-policy actor-critic that maximizes reward plus an entropy bonus, learning from a replay buffer. It never sees V₀.",
      why: "It reaches its final level within about 10⁶ training steps, with a policy that wins many cheap auctions and keeps the users fatigued. The bar is the mean of three trainings.",
    },
    td3: {
      how: "Twin Delayed DDPG (Stable-Baselines3): an off-policy deterministic actor-critic with two critics to curb overestimation, delayed policy updates and target smoothing. It never sees V₀.",
      why: "Unstable here: one of its three trainings converges to a policy that never bids and does not recover. The bar is the mean of the three trainings.",
    },
  };

  // main chart: three blocks of policies (values: main table of the paper, eta = 50)
  const GROUPS = [
    { name: "Planning · MPC (ours)", sub: "H = 35, M = 60", fam: "mpc", rows: [
      { label: "MPC (H = 35, M = 60, α_cont = 0.4)", lines: ["MPC", "α_cont = 0.4"], fam: "mpc", about: "mpc", r: 4.328, ci: 0.275, d: 0.65, dci: 0.13 },
      { label: "MPC (H = 35, M = 60, α_cont = 1)", lines: ["MPC", "α_cont = 1"], fam: "mpc", about: "mpc", r: 4.206, ci: 0.269, d: 0.53, dci: 0.13 },
    ] },
    { name: "Myopic & heuristics", fam: "myopic", rows: [
      { label: "Shading (α = 0.4, M = 20)", lines: ["Shading", "α = 0.4"], fam: "shading", about: "shading", r: 3.680, ci: 0.234, d: null },
      { label: "GreedyMyopic", lines: ["Greedy", "Myopic"], fam: "myopic", about: "greedy", r: 2.826, ci: 0.167, d: -0.85, dci: 0.16 },
      { label: "FiPla", lines: ["FiPla"], fam: "myopic", about: "fipla", r: 2.821, ci: 0.160, d: -0.86, dci: 0.16 },
      { label: "Constant (b = 0.11)", lines: ["Constant", "b = 0.11"], fam: "constant", about: "constant", r: 1.026, ci: 0.099, d: -2.65, dci: 0.19 },
    ] },
    { name: "Deep RL", fam: "rl", rows: [
      { label: "PPO", lines: ["PPO"], fam: "rl", about: "ppo", r: 2.965, ci: 0.182, d: -0.71, dci: 0.12 },
      { label: "TD3", lines: ["TD3"], fam: "rl", about: "td3", r: 2.006, ci: 0.121, d: -1.67, dci: 0.15 },
      { label: "SAC", lines: ["SAC"], fam: "rl", about: "sac", r: 0.707, ci: 0.119, d: -2.97, dci: 0.24 },
    ] },
  ];
  // second chart: MPC with M = 60 and alpha_cont = 1, for each planning horizon H
  const HORIZON = [
    { label: "MPC (H = 10, M = 60, α_cont = 1)", lines: ["H = 10"], fam: "mpc", about: "mpc", r: 3.938, ci: 0.238, d: 0.26, dci: 0.11 },
    { label: "MPC (H = 25, M = 60, α_cont = 1)", lines: ["H = 25"], fam: "mpc", about: "mpc", r: 4.112, ci: 0.257, d: 0.43, dci: 0.11 },
    { label: "MPC (H = 35, M = 60, α_cont = 1)", lines: ["H = 35"], fam: "mpc", about: "mpc", r: 4.206, ci: 0.269, d: 0.53, dci: 0.13 },
  ];
  const SHADING = 3.680;

  const fmt = (v) => v.toFixed(3);
  const sgn = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2);
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

  // one vertical bar with its CI, value and label lines; returns svg
  function barSvg(d, i, x, w, y, base) {
    const f = FAMILY[d.fam], cx = x + w / 2;
    const top = y(d.r + d.ci);
    let g = `<g class="bar" data-i="${i}" tabindex="0" role="button" aria-label="${esc(d.label)}: ${fmt(d.r)}. Open the explanation">` +
      `<rect class="hit" x="${x - 6}" y="${y(4.75)}" width="${w + 12}" height="${base - y(4.75) + 52}" rx="6"/>` +
      `<rect class="fill" x="${x}" y="${y(d.r)}" width="${w}" height="${base - y(d.r)}" rx="4" style="fill:${f.color}"/>` +
      `<line class="ci" x1="${cx}" x2="${cx}" y1="${y(d.r - d.ci)}" y2="${top}"/>` +
      `<line class="ci" x1="${cx - 6}" x2="${cx + 6}" y1="${top}" y2="${top}"/>` +
      `<line class="ci" x1="${cx - 6}" x2="${cx + 6}" y1="${y(d.r - d.ci)}" y2="${y(d.r - d.ci)}"/>` +
      `<text class="val" x="${cx}" y="${top - 7}" text-anchor="middle">${(d.r + 1e-9).toFixed(2)}</text>`;
    d.lines.forEach((t, j) => {
      g += `<text class="${j ? "lab2" : "lab"}" x="${cx}" y="${base + 18 + j * 14}" text-anchor="middle">${esc(t)}</text>`;
    });
    return g + `</g>`;
  }

  function wire(host, list) {
    const tip = host.parentElement.querySelector(".chart-tip");
    host.querySelectorAll(".bar").forEach((g) => {
      const d = list[+g.dataset.i];
      g.addEventListener("mousemove", (e) => {
        const box = host.getBoundingClientRect();
        tip.innerHTML = `<b>${esc(d.label)}</b><br>reward ${fmt(d.r)} ± ${d.ci.toFixed(3)}` +
          (d.d === null ? "<br>reference" : `<br>vs Shading ${sgn(d.d)} ± ${d.dci.toFixed(2)}`) + `<br><span>click for details</span>`;
        tip.hidden = false;
        tip.style.left = Math.max(0, Math.min(e.clientX - box.left + 14, box.width - 230)) + "px";
        tip.style.top = (e.clientY - box.top + 14) + "px";
      });
      g.addEventListener("mouseleave", () => { tip.hidden = true; });
      g.addEventListener("click", () => open(d, tip));
      g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(d, tip); } });
    });
  }

  function render() {
    const host = document.getElementById("chart");
    if (!host) return;
    const W = 860, H = 386, left = 40, right = 8, top = 70, base = 318, gap = 22, pad = 16, bw = 52, inner = 20;
    const max = 4.75, y = (v) => base - (v / max) * (base - top);
    const flat = [];
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="group" aria-label="Mean episode reward per policy, grouped by family">`;
    let grid = "", blocks = "", bars = "";
    for (let v = 0; v <= 4; v++) {
      grid += `<line class="grid" x1="${left}" x2="${W - right}" y1="${y(v)}" y2="${y(v)}"/>` +
             `<text class="tick" x="${left - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
    }
    let x = left + 10;
    GROUPS.forEach((G) => {
      const gi = G.fam === "mpc" ? 44 : inner;   // MPC labels are longer
      const used = G.rows.length * bw + (G.rows.length - 1) * gi;
      const gw = Math.max(used + 2 * pad, 220), p0 = (gw - used) / 2;
      blocks += `<rect class="grp grp-${G.fam}" x="${x}" y="14" width="${gw}" height="${H - 18}" rx="12"/>` +
             `<text class="grp-name grp-name-${G.fam}" x="${x + gw / 2}" y="36" text-anchor="middle">${esc(G.name)}</text>` +
             (G.sub ? `<text class="grp-sub" x="${x + gw / 2}" y="52" text-anchor="middle">${esc(G.sub)}</text>` : "");
      G.rows.forEach((d, j) => {
        bars += barSvg(d, flat.length, x + p0 + j * (bw + gi), bw, y, base);
        flat.push(d);
      });
      x += gw + gap;
    });
    svg += blocks + grid + bars;   // grid lines drawn over the blocks, under the bars
    svg += `<line class="axis0" x1="${left}" x2="${W - right}" y1="${base}" y2="${base}"/>`;
    svg += `</svg>`;
    host.innerHTML = svg;
    wire(host, flat);
  }

  function renderHorizon() {
    const host = document.getElementById("chart-h");
    if (!host) return;
    const W = 460, H = 340, left = 40, top = 70, base = 290, bw = 76, step = 120, x0 = 78;
    const max = 4.75, y = (v) => base - (v / max) * (base - top);
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="group" aria-label="Mean episode reward of MPC for planning horizons H = 10, 25 and 35">`;
    for (let v = 0; v <= 4; v++) {
      svg += `<line class="grid" x1="${left}" x2="${W - 10}" y1="${y(v)}" y2="${y(v)}"/>` +
             `<text class="tick" x="${left - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
    }
    // reference: tuned shading, which does not plan
    svg += `<line class="ref" x1="${left}" x2="${W - 10}" y1="${y(SHADING)}" y2="${y(SHADING)}"/>` +
           `<text class="ref-lab" x="${W - 6}" y="${y(SHADING) - 6}" text-anchor="end">Shading</text>`;
    HORIZON.forEach((d, i) => { svg += barSvg(d, i, x0 + i * step, bw, y, base); });
    // rising arrow above the bars
    const p = HORIZON.map((d, i) => [x0 + i * step + bw / 2, y(d.r + d.ci) - 26]);
    const a = [p[0][0] - 30, p[0][1] + 6], b = [p[2][0] + 30, p[2][1] - 6];
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]) * 180 / Math.PI;
    svg += `<defs><marker id="h-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">` +
           `<path d="M0 0 L10 5 L0 10 z" class="h-head"/></marker></defs>` +
           `<path class="h-arrow" d="M${a[0]} ${a[1]} L${b[0]} ${b[1]}" marker-end="url(#h-arrow)"/>` +
           `<text class="h-text" transform="translate(${(a[0] + b[0]) / 2} ${(a[1] + b[1]) / 2 - 10}) rotate(${ang.toFixed(1)})" text-anchor="middle">performance increases with planning</text>`;
    svg += `<line class="axis0" x1="${left}" x2="${W - 10}" y1="${base}" y2="${base}"/>`;
    svg += `<text class="tick" x="${x0 + step + bw / 2}" y="${H - 6}" text-anchor="middle">planning horizon H (time units)</text>`;
    svg += `</svg>`;
    host.innerHTML = svg;
    wire(host, HORIZON);
  }

  function open(d, tip) {
    const dlg = document.getElementById("algo-dialog");
    const a = ABOUT[d.about], f = FAMILY[d.fam];
    dlg.querySelector(".dlg-family").textContent = f.name;
    dlg.querySelector(".dlg-family").style.setProperty("--fam", f.color);
    dlg.querySelector(".dlg-title").textContent = d.label;
    dlg.querySelector(".dlg-how").textContent = a.how;
    dlg.querySelector(".dlg-why").textContent = a.why;
    dlg.querySelector(".dlg-reward").textContent = fmt(d.r) + " ± " + d.ci.toFixed(3);
    dlg.querySelector(".dlg-delta").textContent = d.d === null ? "reference" : sgn(d.d) + " ± " + d.dci.toFixed(2);
    if (tip) tip.hidden = true;
    if (typeof dlg.showModal === "function") dlg.showModal(); else dlg.setAttribute("open", "");
  }

  function initDialog() {
    const dlg = document.getElementById("algo-dialog");
    if (!dlg) return;
    dlg.querySelector(".dlg-close").addEventListener("click", () => dlg.close());
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });   // click on the backdrop
  }

  document.addEventListener("DOMContentLoaded", () => { render(); renderHorizon(); initDialog(); });
})();
