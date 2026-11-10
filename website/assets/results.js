/* DynamAuction website: interactive results chart (main scenario, mean campaign length 50).
 * Data: Table "main results" of the paper (mean episode reward ± 95% CI over 1 000 test campaigns, and paired
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

  const DATA = [
    { label: "MPC (H = 35, M = 60, α_cont = 0.4)", fam: "mpc", about: "mpc", r: 4.328, ci: 0.275, d: 0.65, dci: 0.13 },
    { label: "MPC (H = 25, M = 60, α_cont = 0.4)", fam: "mpc", about: "mpc", r: 4.285, ci: 0.268, d: 0.61, dci: 0.12 },
    { label: "MPC (H = 25, M = 80)", fam: "mpc", about: "mpc", r: 4.282, ci: 0.268, d: 0.60, dci: 0.13 },
    { label: "MPC (H = 35, M = 60)", fam: "mpc", about: "mpc", r: 4.206, ci: 0.269, d: 0.53, dci: 0.13 },
    { label: "MPC (H = 25, M = 60)", fam: "mpc", about: "mpc", r: 4.112, ci: 0.257, d: 0.43, dci: 0.11 },
    { label: "MPC (H = 10, M = 60)", fam: "mpc", about: "mpc", r: 3.938, ci: 0.238, d: 0.26, dci: 0.11 },
    { label: "Shading (α = 0.4, M = 20)", fam: "shading", about: "shading", r: 3.680, ci: 0.234, d: null },
    { label: "PPO", fam: "rl", about: "ppo", r: 2.965, ci: 0.182, d: -0.71, dci: 0.12 },
    { label: "GreedyMyopic", fam: "myopic", about: "greedy", r: 2.826, ci: 0.167, d: -0.85, dci: 0.16 },
    { label: "FiPla", fam: "myopic", about: "fipla", r: 2.821, ci: 0.160, d: -0.86, dci: 0.16 },
    { label: "TD3", fam: "rl", about: "td3", r: 2.006, ci: 0.121, d: -1.67, dci: 0.15 },
    { label: "Constant (b = 0.11)", fam: "constant", about: "constant", r: 1.026, ci: 0.099, d: -2.65, dci: 0.19 },
    { label: "SAC", fam: "rl", about: "sac", r: 0.707, ci: 0.119, d: -2.97, dci: 0.24 },
  ];

  const fmt = (v) => v.toFixed(3);
  const sgn = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2);
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

  function render() {
    const host = document.getElementById("chart");
    if (!host) return;
    const rowH = 34, top = 28, labelW = 250, W = 760, barW = W - labelW - 70;
    const H = top + DATA.length * rowH + 8;
    const max = 4.75;
    const x = (v) => labelW + (v / max) * barW;
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="group" aria-label="Mean episode reward per policy, main scenario">`;
    for (let v = 0; v <= 4; v += 1) {
      svg += `<line class="grid" x1="${x(v)}" x2="${x(v)}" y1="${top - 8}" y2="${H - 4}"/>` +
             `<text class="tick" x="${x(v)}" y="${top - 14}" text-anchor="middle">${v}</text>`;
    }
    DATA.forEach((d, i) => {
      const y = top + i * rowH;
      const f = FAMILY[d.fam];
      svg += `<g class="bar" data-i="${i}" tabindex="0" role="button" aria-label="${esc(d.label)}: ${fmt(d.r)}. Open the explanation">` +
        `<rect class="hit" x="0" y="${y}" width="${W}" height="${rowH}"/>` +
        `<text class="lab" x="${labelW - 12}" y="${y + rowH / 2 + 4}" text-anchor="end">${esc(d.label)}</text>` +
        `<rect class="fill" x="${x(0)}" y="${y + 7}" width="${x(d.r) - x(0)}" height="${rowH - 14}" rx="4" style="fill:${f.color}"/>` +
        `<line class="ci" x1="${x(d.r - d.ci)}" x2="${x(d.r + d.ci)}" y1="${y + rowH / 2}" y2="${y + rowH / 2}"/>` +
        `<line class="ci" x1="${x(d.r - d.ci)}" x2="${x(d.r - d.ci)}" y1="${y + rowH / 2 - 5}" y2="${y + rowH / 2 + 5}"/>` +
        `<line class="ci" x1="${x(d.r + d.ci)}" x2="${x(d.r + d.ci)}" y1="${y + rowH / 2 - 5}" y2="${y + rowH / 2 + 5}"/>` +
        `<text class="val" x="${x(d.r + d.ci) + 8}" y="${y + rowH / 2 + 4}">${(d.r + 1e-9).toFixed(2)}</text>` +
        `</g>`;
    });
    svg += `</svg>`;
    host.innerHTML = svg;

    const tip = document.getElementById("chart-tip");
    host.querySelectorAll(".bar").forEach((g) => {
      const d = DATA[+g.dataset.i];
      g.addEventListener("mousemove", (e) => {
        const box = host.getBoundingClientRect();
        tip.innerHTML = `<b>${esc(d.label)}</b><br>reward ${fmt(d.r)} ± ${d.ci.toFixed(3)}` +
          (d.d === null ? "<br>reference" : `<br>vs Shading ${sgn(d.d)} ± ${d.dci.toFixed(2)}`) + `<br><span>click for details</span>`;
        tip.hidden = false;
        tip.style.left = Math.min(e.clientX - box.left + 14, box.width - 230) + "px";
        tip.style.top = (e.clientY - box.top + 14) + "px";
      });
      g.addEventListener("mouseleave", () => { tip.hidden = true; });
      g.addEventListener("click", () => open(d));
      g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(d); } });
    });
  }

  function open(d) {
    const dlg = document.getElementById("algo-dialog");
    const a = ABOUT[d.about], f = FAMILY[d.fam];
    dlg.querySelector(".dlg-family").textContent = f.name;
    dlg.querySelector(".dlg-family").style.setProperty("--fam", f.color);
    dlg.querySelector(".dlg-title").textContent = d.label;
    dlg.querySelector(".dlg-how").textContent = a.how;
    dlg.querySelector(".dlg-why").textContent = a.why;
    dlg.querySelector(".dlg-reward").textContent = fmt(d.r) + " ± " + d.ci.toFixed(3);
    dlg.querySelector(".dlg-delta").textContent = d.d === null ? "reference" : sgn(d.d) + " ± " + d.dci.toFixed(2);
    document.getElementById("chart-tip").hidden = true;
    if (typeof dlg.showModal === "function") dlg.showModal(); else dlg.setAttribute("open", "");
  }

  function initDialog() {
    const dlg = document.getElementById("algo-dialog");
    if (!dlg) return;
    dlg.querySelector(".dlg-close").addEventListener("click", () => dlg.close());
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });   // click on the backdrop
  }

  document.addEventListener("DOMContentLoaded", () => { render(); initDialog(); });
})();
