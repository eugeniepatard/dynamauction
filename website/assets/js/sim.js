/* DynamAuction website: the animated pipeline of one environment step, and a small in-browser campaign demo.
 * The demo re-implements a few primitives of the library in JavaScript (same formulas, simplified), for illustration:
 *   value of a display  p = clip(m · (1ᵀu)(1ᵀx), 0, 1), m given by the carry-over context (survival kernel),
 *   oracle value        V0 = p · (1 − exp(−δ (T − t)))  (delayed conversions; = p if the reward is immediate),
 *   conversion          Bernoulli(p), after a delay Exp(δ); credited only if it happens before the last auction.
 */
(function () {
  "use strict";

  /* ------------------------------------------------------------------ pipeline ------------------------------------ */
  function initPipeline() {
    const root = document.getElementById("pipeline");
    if (!root) return;
    const steps = Array.from(root.querySelectorAll(".pl-step"));
    const detail = root.querySelector(".pl-detail");
    const toggle = root.querySelector(".pl-toggle");
    const order = [0, 1, 2, 3, 4, 5];
    const loop = [1, 2, 3, 4, 5];
    let pos = 0, playing = !window.matchMedia("(prefers-reduced-motion: reduce)").matches, timer = null;

    function show(i) {
      steps.forEach((s, j) => {
        s.classList.toggle("is-active", j === i);
        s.classList.toggle("is-done", j < i);
        s.setAttribute("aria-pressed", j === i ? "true" : "false");
      });
      root.style.setProperty("--pl-progress", String(i / (steps.length - 1)));
      root.classList.toggle("is-looping", i === 5);
      const s = steps[i];
      detail.querySelector(".pl-detail-title").textContent = s.dataset.title;
      detail.querySelector(".pl-detail-body").textContent = s.dataset.body;
    }
    function next() {
      const cur = pos < order.length ? order[pos] : loop[(pos - order.length) % loop.length];
      show(cur);
      pos += 1;
    }
    function setPlaying(p) {
      playing = p;
      toggle.textContent = p ? "Pause" : "Play";
      toggle.setAttribute("aria-pressed", p ? "true" : "false");
      clearInterval(timer);
      if (p) timer = setInterval(next, 2800);
    }
    steps.forEach((s, i) => s.addEventListener("click", () => { setPlaying(false); pos = i + 1; show(i); }));
    toggle.addEventListener("click", () => setPlaying(!playing));
    next();
    setPlaying(playing);
  }

  /* ------------------------------------------------------------------ demo: model --------------------------------- */
  const DELTA = 1.0;           // conversion delay rate δ (main preset)
  const MEAN_T = 10;           // short demo campaigns (the benchmark uses a mean of 50)
  const MIN_GAP = 0.3;         // demo only: minimum time between two auctions

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const normal = (r) => { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  const sum = (v) => v.reduce((a, b) => a + b, 0);
  const clip01 = (v) => Math.min(1, Math.max(0, v));
  const f2 = (v) => (Math.abs(v) < 0.005 ? "0.00" : v.toFixed(2));

  const RATES = {
    poisson: { f: () => 1, max: 1 },
    sparse: { f: () => 0.5, max: 0.5 },
    burst: { f: (t) => (t < 3 ? 2 : 0.5), max: 2 },
    decay: { f: (t) => 1.2 * Math.pow(0.5, Math.floor(t / 4)), max: 1.2 },
  };

  function drawCompetitor(kind, r) {
    if (kind === "uniform") return r();
    if (kind === "exponential") return -0.3 * Math.log(1 - r());
    return Math.exp(-1 + normal(r));     // LogNormal(-1, 1)
  }

  function buildCampaign(cfg) {
    const r = mulberry32(cfg.seed);
    const T = cfg.horizon === "fixed" ? MEAN_T : Math.max(2.5, -MEAN_T * Math.log(1 - r()));
    let u;
    do { u = [normal(r), normal(r), normal(r)]; } while (sum(u) < 0.8);   // a user receptive to the ad, so that something happens
    const rate = RATES[cfg.arrivals];
    const times = [];
    for (let t = 0; ;) {
      t += -Math.log(1 - r()) / rate.max;
      if (t > T) break;
      // demo only: at least MIN_GAP between two auctions, so that they can be told apart on screen
      if (r() < rate.f(t) / rate.max && (!times.length || t - times[times.length - 1] >= MIN_GAP)) times.push(t);
    }
    if (!times.length) times.push(0);
    const auctions = times.map((t, i) => ({
      i, t, x: [normal(r), normal(r), normal(r)], B: drawCompetitor(cfg.competition, r),
      uc: r(), delay: -Math.log(1 - r()) / DELTA,
    }));
    return { T, u, auctions };
  }

  function makeContext(type) { return { type, wins: [], stored: 1, n: 0, tLast: null }; }
  function contextValue(c, t) {
    switch (c.type) {
      case "fatigue": return c.wins.length ? c.stored * (1 - 0.9 * Math.exp(-(t - c.tLast))) : c.stored;
      case "recovering": return c.wins.reduce((s, ti) => s * (1 - 0.9 * Math.exp(-0.3 * (t - ti))), 1);
      case "recency": return c.tLast === null ? t + 50 : t - c.tLast;
      default: return c.n;    // saturation
    }
  }
  function contextWin(c, t) {
    if (c.type === "fatigue") c.stored = contextValue(c, t);
    c.wins.push(t); c.n += 1; c.tLast = t;
  }
  function convProb(c, val, rel) {
    return c.type === "saturation" ? clip01(rel * Math.exp(-val / 3)) : clip01(val * rel);
  }
  function contextLabel(c, val, mode) {
    if (c.type === "fatigue") {
      if (mode === "full") {
        const w = c.wins.map((x) => x.toFixed(1));
        return "S = {" + (w.length > 4 ? w.slice(-4).join(", ") + ", …" : w.join(", ")) + "}";
      }
      return "φ(S, t) = " + f2(val);
    }
    if (c.type === "recovering") return "S = " + f2(val);
    if (c.type === "recency") return c.tLast === null ? "Δt = no win yet" : "Δt = " + val.toFixed(1);
    return "n = " + val + " wins";
  }

  /* ------------------------------------------------------------------ demo: animation ----------------------------- */
  const W = 960, H = 400, LINE_Y = 300, STOP_X = 380, MAN_X = 324, PX = 110, RIVAL_X = 434, GAVEL_X = 610;
  const DUR = { observe: 0.8, bid: 0.85, rival: 1.15, absorb: 0.5, gavel: 0.85, result: 0.7, click: 1.4, leave: 0.6 };

  function initDemo() {
    const root = document.getElementById("demo");
    if (!root) return;
    const canvas = root.querySelector("canvas");
    const g = canvas.getContext("2d");
    const $ = (id) => document.getElementById(id);
    const els = {
      auction: $("d-auction"), competition: $("d-competition"), arrivals: $("d-arrivals"), horizon: $("d-horizon"),
      context: $("d-context"), obs: $("d-obs"), conv: $("d-conv"), policy: $("d-policy"), alpha: $("d-alpha"),
      cst: $("d-const"), speed: $("d-speed"), play: $("d-play"), reset: $("d-reset"), fresh: $("d-new"),
      alphaOut: $("d-alpha-out"), cstOut: $("d-const-out"), alphaRow: $("d-alpha-row"), cstRow: $("d-const-row"),
      seedOut: $("d-seed"), step: $("d-step"),
    };
    let seed = 97, S = null, running = false, last = 0, colors = {};

    function readColors() {
      const cs = getComputedStyle(document.documentElement);
      const v = (n) => cs.getPropertyValue(n).trim();
      colors = { ink: v("--ink"), ink2: v("--ink-2"), muted: v("--muted"), line: v("--line"), accent: v("--accent"),
        accentInk: v("--accent-ink"), bid: v("--bid"), bidInk: v("--bid-ink"), lime: v("--lime"), limeInk: v("--lime-ink"), surface: v("--surface"), bg: v("--bg"), rival: v("--rival"), soft: v("--accent-soft"), bidSoft: v("--bid-soft") };
    }
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); draw(); });

    function cfg() {
      return { seed, auction: els.auction.value, competition: els.competition.value, arrivals: els.arrivals.value,
        horizon: els.horizon.value, context: els.context.value, obs: els.obs.value, conv: els.conv.value,
        policy: els.policy.value, alpha: +els.alpha.value, cst: +els.cst.value };
    }
    function syncControls() {
      els.alphaRow.hidden = els.policy.value !== "shading";
      els.cstRow.hidden = els.policy.value !== "constant";
      els.alphaOut.textContent = (+els.alpha.value).toFixed(1);
      els.cstOut.textContent = (+els.cst.value).toFixed(2);
      els.obs.disabled = els.context.value !== "fatigue";
      if (els.obs.disabled) els.obs.value = "summary";
      els.seedOut.textContent = "campaign #" + seed;
    }

    function reset() {
      const c = cfg();
      const camp = buildCampaign(c);
      S = { c, ...camp, ctx: makeContext(c.context), t: 0, k: 0, phase: "travel", pt: 0, cur: null,
        paid: 0, convN: 0, lost: 0, sinceReward: 0, sinceConv: [], pending: [], popups: [], popup: null,
        lastT: camp.auctions[camp.auctions.length - 1].t, done: [] };
      setRunning(false);
      draw();
    }
    function setRunning(r) {
      running = r;
      els.play.textContent = r ? "Pause" : (S && S.waiting ? "Continue" : S && S.phase === "end" ? "Replay" : "Start");
      if (r) { last = performance.now(); requestAnimationFrame(frame); }
    }

    function startAuction(a) {
      const st = S, c = st.c;
      const val = contextValue(st.ctx, a.t);
      const rel = sum(st.u) * sum(a.x);
      const p = convProb(st.ctx, val, rel);
      const v0 = c.conv === "immediate" ? p : p * (1 - Math.exp(-DELTA * (st.T - a.t)));
      const bid = c.policy === "fipla" ? v0 : c.policy === "shading" ? c.alpha * v0 : c.cst;
      const win = bid > a.B;
      const pay = win ? (c.auction === "second" ? a.B : bid) : 0;
      st.cur = { a, val, p, v0, bid, win, pay,
        obs: [
          ["time t", a.t.toFixed(2) + "   ·   step " + (a.i + 1)],
          ["context", contextLabel(st.ctx, val, c.obs)],
          ["user u", "(" + st.u.map((z) => z.toFixed(1)).join(", ") + ")"],
          ["display x", "(" + a.x.map((z) => z.toFixed(1)).join(", ") + ")"],
          ["exogenous y", "0"],
          ["reward since last", (st.sinceReward >= 0 ? "+" : "") + f2(st.sinceReward)],
          ["conv. since last", st.sinceConv.length ? st.sinceConv.map((z) => "t=" + z.toFixed(1)).join(", ") : "none"],
        ] };
      st.sinceReward = 0; st.sinceConv = [];
      go("observe");
    }

    function resolveAuction() {
      const st = S, cur = st.cur, a = cur.a;
      st.paid += cur.pay; st.sinceReward -= cur.pay;
      if (cur.win) {
        contextWin(st.ctx, a.t);
        if (a.uc < cur.p) {
          if (st.c.conv === "immediate") {
            st.convN += 1; st.sinceReward += 1;
            st.popup = { t0: 0, credited: true, delayed: false };
            cur.click = true;
          } else if (a.t + a.delay <= st.T) {
            st.pending.push({ t: a.t + a.delay, won: a.t });
          }
        }
      }
      st.done.push({ t: a.t, win: cur.win });
    }

    function go(phase) { S.phase = phase; S.pt = 0; S.rate = null; }

    function update(dt) {
      const st = S;
      st.pt += dt;
      if (st.popup) { st.popup.t0 += dt; if (st.popup.t0 > DUR.click) st.popup = null; }
      if (!st.popup && st.popups.length) st.popup = { t0: 0, ...st.popups.shift() };
      switch (st.phase) {
        case "travel": {
          if (st.popup && st.popup.credited) break;                // the bidder stops to celebrate a conversion
          const N = st.auctions.length;
          const target = st.k < N ? st.auctions[st.k].t : st.T;
          if (st.rate == null) {                                   // one constant speed per stretch, at least 0.7 s long
            const gap = target - st.t;
            st.rate = Math.min(Math.max(0.8, gap / 1.8), Math.max(gap, 1e-6) / 0.7);
          }
          st.t = Math.min(target, st.t + st.rate * dt);
          st.pending = st.pending.filter((cv) => {
            if (cv.t > st.t) return true;
            const credited = cv.t <= st.lastT;
            if (credited) { st.convN += 1; st.sinceReward += 1; st.sinceConv.push(cv.t); } else st.lost += 1;
            st.popups.push({ credited, delayed: true, won: cv.won });
            return false;
          });
          if (st.t >= target) {
            if (st.k < N) startAuction(st.auctions[st.k]);
            else { go("end"); setRunning(false); }
          }
          break;
        }
        case "observe":
          if (st.pt >= DUR.observe) {
            if (els.step.checked) { if (!st.waiting) { st.waiting = true; setRunning(false); } }
            else go("bid");
          }
          break;
        case "bid": if (st.pt >= DUR.bid) go("rival"); break;
        case "rival": if (st.pt >= DUR.rival) go("absorb"); break;
        case "absorb": if (st.pt >= DUR.absorb) go("gavel"); break;
        case "gavel": if (st.pt >= DUR.gavel) { resolveAuction(); go("result"); } break;
        case "result": if (st.pt >= DUR.result) go(st.cur.click ? "click" : "leave"); break;
        case "click": if (st.pt >= DUR.click) go("leave"); break;
        case "leave": if (st.pt >= DUR.leave) { st.k += 1; st.cur = null; go("travel"); } break;
      }
    }

    function frame(now) {
      if (!running) return;
      const dt = Math.min(0.05, (now - last) / 1000) * (+els.speed.value);
      last = now;
      update(dt);
      draw();
      if (running) requestAnimationFrame(frame);
    }

    /* ---------------- drawing ---------------- */
    const ease = (x) => (x < 0 ? 0 : x > 1 ? 1 : x * x * (3 - 2 * x));
    const xOf = (t) => STOP_X + (t - S.t) * PX;   // the future comes from the right

    function text(s, x, y, size, color, opts = {}) {
      g.font = (opts.weight || 500) + " " + size + "px " + (opts.mono ? "'IBM Plex Mono', monospace" : "'IBM Plex Sans', sans-serif");
      g.fillStyle = color; g.textAlign = opts.align || "left"; g.textBaseline = opts.base || "alphabetic";
      g.globalAlpha = opts.alpha == null ? 1 : opts.alpha;
      g.fillText(s, x, y);
      g.globalAlpha = 1;
    }
    function rrect(x, y, w, h, r) {
      g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
      g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
    }

    function stickman(x, footY, color, o = {}) {
      const walk = o.walk || 0, jump = o.jump || 0;
      const y = footY - jump;
      g.strokeStyle = color; g.fillStyle = color; g.lineWidth = 3.5; g.lineCap = "round"; g.lineJoin = "round";
      g.beginPath(); g.arc(x, y - 60, 9, 0, Math.PI * 2); g.fill();
      g.beginPath(); g.moveTo(x, y - 50); g.lineTo(x, y - 24); g.stroke();
      const sw = walk ? Math.sin(walk) * 10 : 7;                       // legs: swing when walking, apart when standing
      g.beginPath(); g.moveTo(x - sw, y); g.lineTo(x, y - 24); g.lineTo(x + sw, y); g.stroke();
      const dir = o.dir || -1;                                          // -1: faces left, +1: faces right
      if (o.cheer) {                                                    // both arms up: a conversion!
        g.beginPath(); g.moveTo(x - 13, y - 74); g.lineTo(x, y - 44); g.lineTo(x + 13, y - 74); g.stroke();
        return;
      }
      const a1 = o.reach ? 17 * dir : (walk ? Math.sin(walk + Math.PI) * 8 : 8);
      const a2 = walk ? -Math.sin(walk + Math.PI) * 8 : -8;
      g.beginPath(); g.moveTo(x + a1, y - 30 - (o.reach ? 10 : 0)); g.lineTo(x, y - 44); g.lineTo(x + a2, y - 30); g.stroke();
    }

    function moneyBag(x, y, label, fill, ink, scale = 1, textColor = "#3d2600") {
      g.save(); g.translate(x, y); g.scale(scale, scale);
      g.fillStyle = fill;
      g.beginPath(); g.ellipse(0, 3, 15, 13, 0, 0, Math.PI * 2); g.fill();
      g.beginPath(); g.moveTo(-6, -9); g.lineTo(6, -9); g.lineTo(9, -16); g.lineTo(-9, -16); g.closePath(); g.fill();
      g.strokeStyle = ink; g.lineWidth = 1.5; g.beginPath(); g.moveTo(-7, -9); g.lineTo(7, -9); g.stroke();
      g.restore();
      if (scale > 0.4) text(label, x, y + 7 * scale, 10 * scale, textColor, { mono: true, align: "center", weight: 600 });
    }

    function gavel(x, y, angle) {
      g.save(); g.translate(x + 30, y - 6); g.rotate(angle);
      g.strokeStyle = colors.ink2; g.lineWidth = 4; g.lineCap = "round";
      g.beginPath(); g.moveTo(0, 0); g.lineTo(-34, 0); g.stroke();
      g.fillStyle = colors.ink; rrect(-46, -11, 18, 22, 3); g.fill();
      g.restore();
      g.fillStyle = colors.ink2; rrect(x - 26, y + 14, 34, 8, 2); g.fill();     // sound block
    }

    function screenPopup(p) {
      const k = Math.min(1, p.t0 / 0.25) * Math.min(1, (DUR.click - p.t0) / 0.25);
      g.globalAlpha = Math.max(0, k);
      const x = 700, y = 20;
      g.fillStyle = colors.surface; g.strokeStyle = colors.line; g.lineWidth = 1.5;
      rrect(x - 8, y - 6, 252, 160, 10); g.fill(); g.stroke();
      g.strokeStyle = colors.ink2; g.lineWidth = 2; rrect(x + 6, y + 8, 110, 72, 5); g.stroke();
      g.beginPath(); g.moveTo(x + 61, y + 80); g.lineTo(x + 61, y + 92); g.moveTo(x + 44, y + 93); g.lineTo(x + 78, y + 93); g.stroke();
      g.fillStyle = colors.line; for (let i = 0; i < 3; i++) g.fillRect(x + 14, y + 16 + i * 9, 50, 4);
      g.fillStyle = p.credited ? colors.lime : colors.line; rrect(x + 68, y + 16, 40, 30, 3); g.fill();
      text("AD", x + 88, y + 35, 10, "#213d00", { mono: true, align: "center", weight: 600 });
      const cx = x + 92, cy = y + 40;                                       // cursor
      g.fillStyle = colors.ink; g.strokeStyle = colors.surface; g.lineWidth = 1;
      g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx, cy + 16); g.lineTo(cx + 4, cy + 12); g.lineTo(cx + 8, cy + 19);
      g.lineTo(cx + 10, cy + 18); g.lineTo(cx + 7, cy + 11); g.lineTo(cx + 12, cy + 11); g.closePath(); g.fill(); g.stroke();
      g.save(); g.translate(x + 180, y + 40); g.rotate(-0.12);
      text("CLICK!", 0, 0, 28, p.credited ? colors.limeInk : colors.muted, { weight: 700, align: "center" }); g.restore();
      g.strokeStyle = p.credited ? colors.limeInk : colors.muted; g.lineWidth = 2;
      for (let i = 0; i < 5; i++) { const a = -0.9 + i * 0.45; g.beginPath(); g.moveTo(cx + 4 + Math.cos(a) * 8, cy - 2 + Math.sin(a) * 8); g.lineTo(cx + 4 + Math.cos(a) * 15, cy - 2 + Math.sin(a) * 15); g.stroke(); }
      text("Someone clicked on the ad!", x + 6, y + 116, 13, colors.ink, { weight: 600 });
      const sub = p.credited ? (p.delayed ? "+1 conversion (won at t=" + p.won.toFixed(1) + ")" : "+1 conversion, immediate")
                             : "after the last auction: not credited";
      text(sub, x + 6, y + 136, 11, colors.muted, { mono: true });
      g.globalAlpha = 1;
    }

    function obsCard(cur) {
      const x = 224, y = 16, w = 294, h = 198;
      g.fillStyle = colors.surface; g.strokeStyle = colors.line; g.lineWidth = 1.5;
      rrect(x, y, w, h, 10); g.fill(); g.stroke();
      g.strokeStyle = colors.line; g.beginPath(); g.moveTo(x + 150, y + h); g.lineTo(STOP_X, LINE_Y - 24); g.stroke();
      text("WHAT THE BIDDER OBSERVES", x + 14, y + 22, 10, colors.muted, { mono: true, weight: 600 });
      text("V₀ = " + f2(cur.v0) + " · hidden", x + w - 12, y + 22, 10, colors.muted, { mono: true, align: "right", alpha: 0.55 });
      cur.obs.forEach(([k, v], i) => {
        text(k, x + 14, y + 48 + i * 21, 11, colors.muted, { mono: true });
        text(v, x + 146, y + 48 + i * 21, 11, k === "context" ? colors.accent : colors.ink, { mono: true, weight: k === "context" ? 600 : 500 });
      });
    }

    function hud() {
      const st = S;
      g.fillStyle = colors.bg; rrect(12, 12, 200, 92, 10); g.fill();
      text("PAYMENTS", 24, 34, 10, colors.muted, { mono: true, weight: 600 });
      text((st.paid > 0.005 ? "−" : "") + f2(st.paid), 200, 34, 13, colors.ink, { mono: true, align: "right", weight: 600 });
      text("CONVERSIONS", 24, 58, 10, colors.muted, { mono: true, weight: 600 });
      text((st.convN ? "+" : "") + st.convN.toFixed(2), 200, 58, 13, colors.limeInk, { mono: true, align: "right", weight: 600 });
      g.strokeStyle = colors.line; g.beginPath(); g.moveTo(24, 70); g.lineTo(200, 70); g.stroke();
      text("NET REWARD", 24, 92, 10, colors.muted, { mono: true, weight: 600 });
      const net = st.convN - st.paid;
      text((net >= 0 ? "+" : "") + f2(net), 200, 92, 14, colors.accent, { mono: true, align: "right", weight: 700 });
    }

    function draw() {
      if (!S) return;
      const dpr = window.devicePixelRatio || 1;
      const cw = canvas.clientWidth || W;
      if (canvas.width !== Math.round(cw * dpr)) { canvas.width = Math.round(cw * dpr); canvas.height = Math.round(cw * dpr * H / W); }
      const sc = canvas.width / W;
      g.setTransform(sc, 0, 0, sc, 0, 0);
      g.clearRect(0, 0, W, H);
      g.fillStyle = colors.surface; g.fillRect(0, 0, W, H);
      const st = S, ph = st.phase, p01 = (d) => ease(st.pt / d);

      // time line, ticks every time unit, scrolling to the right
      g.strokeStyle = colors.ink2; g.lineWidth = 2; g.beginPath(); g.moveTo(0, LINE_Y); g.lineTo(W, LINE_Y); g.stroke();
      for (let k = Math.floor(st.t - 4); k <= Math.ceil(st.t + 6); k++) {
        const x = xOf(k); if (x < -10 || x > W + 10) continue;
        g.strokeStyle = colors.line; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x, LINE_Y + 4); g.lineTo(x, LINE_Y + 10); g.stroke();
        if (k >= 0) text("t=" + k, x, LINE_Y + 26, 10, colors.muted, { mono: true, align: "center", alpha: 0.7 });
      }

      // end of the campaign: checkered flag
      const fx = xOf(st.T);
      if (fx > -40 && fx < W + 40) {
        g.strokeStyle = colors.ink; g.lineWidth = 2.5; g.beginPath(); g.moveTo(fx, LINE_Y); g.lineTo(fx, LINE_Y - 70); g.stroke();
        for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) {
          g.fillStyle = (r + c) % 2 ? colors.surface : colors.ink; g.fillRect(fx + c * 8, LINE_Y - 70 + r * 8, 8, 8);
        }
        g.strokeStyle = colors.ink; g.lineWidth = 1; g.strokeRect(fx, LINE_Y - 70, 32, 24);
        text("T", fx - 6, LINE_Y - 56, 11, colors.muted, { mono: true, align: "right" });
      }

      // auctions on the line (labels skipped when two auctions are too close)
      let lastLabel = Infinity;
      st.auctions.forEach((a, j) => {
        const isCur = st.cur && st.cur.a === a;
        let x = xOf(a.t); if (x < -20 || x > W + 20) return;
        const past = st.done.find((d) => d.t === a.t);
        if (past && !isCur) {
          g.fillStyle = past.win ? colors.bid : colors.surface; g.strokeStyle = past.win ? colors.bid : colors.muted; g.lineWidth = 2;
          g.globalAlpha = 0.65; g.beginPath(); g.arc(x, LINE_Y, 7, 0, Math.PI * 2); g.fill(); g.stroke(); g.globalAlpha = 1;
          if (Math.abs(lastLabel - x) > 30) { text(past.win ? "won" : "lost", x, LINE_Y - 14, 9, colors.muted, { mono: true, align: "center", alpha: 0.8 }); lastLabel = x; }
        } else {
          const r = isCur && (ph === "absorb" || ph === "gavel") ? 10 + 2 * Math.sin(st.pt * 18) : isCur ? 10 : 8;
          g.fillStyle = colors.bid; g.beginPath(); g.arc(x, LINE_Y, r, 0, Math.PI * 2); g.fill();
          g.strokeStyle = colors.surface; g.lineWidth = 2; g.stroke();
          if (!isCur && Math.abs(lastLabel - x) > 34) { text("t=" + a.t.toFixed(1), x, LINE_Y - 16, 9, colors.muted, { mono: true, align: "center", alpha: 0.75 }); lastLabel = x; }
        }
      });

      // the bidder (black): walks while the line moves, jumps over the auction he leaves behind
      let jump = 0;
      if (ph === "travel" || ph === "leave") st.done.forEach((d) => { const dx = xOf(d.t) - MAN_X; if (Math.abs(dx) < 26) jump = Math.max(jump, 22 * Math.cos((dx / 26) * Math.PI / 2)); });
      const cheer = !!(st.popup && st.popup.credited);
      if (cheer) jump = Math.abs(Math.sin(performance.now() / 140)) * 22;    // happy jumps
      const walking = ph === "travel" && !cheer;
      stickman(MAN_X, LINE_Y, colors.ink, { walk: walking ? performance.now() / 110 : 0, jump, reach: ph === "bid" && !cheer, dir: 1, cheer });
      if (cheer) {
        g.strokeStyle = colors.limeInk; g.lineWidth = 2.5; g.lineCap = "round";
        const cy = LINE_Y - 66 - jump;
        for (let i = 0; i < 6; i++) { const an = i * Math.PI / 3 + performance.now() / 600; g.beginPath(); g.moveTo(MAN_X + Math.cos(an) * 26, cy + Math.sin(an) * 26); g.lineTo(MAN_X + Math.cos(an) * 34, cy + Math.sin(an) * 34); g.stroke(); }
      }

      const cur = st.cur;
      if (cur) {
        // the competitor (purple) climbs up from below the line, then leaves
        let ry = null;
        if (ph === "rival") ry = LINE_Y + 90 * (1 - p01(0.6));
        else if (["absorb", "gavel", "result", "click"].includes(ph)) ry = LINE_Y;
        else if (ph === "leave") ry = LINE_Y + 110 * p01(DUR.leave);
        if (ry !== null) { g.save(); g.beginPath(); g.rect(0, 0, W, LINE_Y + 1 + (ry - LINE_Y)); g.restore(); stickman(RIVAL_X, ry, colors.rival, { reach: ph === "rival", dir: -1 }); }

        // money bags: the bidder's (amber) and the competitor's (purple), absorbed by the auction
        const bidLabel = f2(cur.bid), rivLabel = f2(cur.a.B);
        if (ph === "bid") {
          const k = p01(DUR.bid);
          moneyBag(MAN_X + 18 + (STOP_X - 12 - (MAN_X + 18)) * k, LINE_Y - 34 + 14 * k - 30 * Math.sin(Math.PI * k), bidLabel, colors.bid, colors.surface);
        } else if (["rival"].includes(ph)) {
          moneyBag(STOP_X - 12, LINE_Y - 20, bidLabel, colors.bid, colors.surface);
          const k = ease((st.pt - 0.6) / 0.5);
          if (st.pt > 0.6) moneyBag(RIVAL_X - 16 + (STOP_X + 12 - (RIVAL_X - 16)) * k, LINE_Y - 34 + 14 * k - 24 * Math.sin(Math.PI * k), rivLabel, colors.rival, colors.surface, 1, colors.accentInk);
        } else if (ph === "absorb") {
          const k = p01(DUR.absorb);
          moneyBag(STOP_X - 12 * (1 - k), LINE_Y - 20 + 20 * k, bidLabel, colors.bid, colors.surface, 1 - k);
          moneyBag(STOP_X + 12 * (1 - k), LINE_Y - 20 + 20 * k, rivLabel, colors.rival, colors.surface, 1 - k, colors.accentInk);
        }

        // the gavel taps, then WIN / LOST, payment
        if (["gavel", "result", "click", "leave"].includes(ph)) {
          const k = ph === "gavel" ? st.pt / DUR.gavel : 1;
          const ang = ph === "gavel" ? (k < 0.45 ? -0.9 + 0.9 * ease(k / 0.45) * 1.15 : k < 0.6 ? 0.12 : 0.12 - 0.5 * ease((k - 0.6) / 0.4)) : -0.38;
          gavel(GAVEL_X, 158, ang);
          if (ph === "gavel" && k > 0.42 && k < 0.7) {
            g.strokeStyle = colors.ink2; g.lineWidth = 2;
            for (let i = -1; i <= 1; i++) { g.beginPath(); g.moveTo(GAVEL_X - 26 + i * 16, 162); g.lineTo(GAVEL_X - 30 + i * 22, 150); g.stroke(); }
          }
          if (ph !== "gavel") {
            text(cur.win ? "WIN" : "LOST", GAVEL_X - 5, 202, 22, cur.win ? colors.bidInk : colors.muted, { weight: 700, align: "center" });
            const sub = cur.win ? "pays " + f2(cur.pay) + (st.c.auction === "second" ? " (2nd price)" : " (1st price)") : "pays nothing";
            text(sub, GAVEL_X - 5, 219, 11, colors.muted, { mono: true, align: "center" });
          }
        }
        if (["observe", "bid", "rival", "absorb", "gavel", "result", "click", "leave"].includes(ph)) {
          g.globalAlpha = ph === "observe" ? p01(0.4) : ph === "leave" ? 1 - p01(DUR.leave) : 1;
          obsCard(cur);
          g.globalAlpha = 1;
        }
      }

      hud();
      if (st.popup) screenPopup(st.popup);
      if (st.waiting) {
        g.fillStyle = colors.accent; rrect(W / 2 + 60, 236, 250, 34, 17); g.fill();
        text("Take your time · press Continue ▸", W / 2 + 185, 258, 13, colors.surface, { align: "center", weight: 600 });
      }

      // footer of the canvas: clock and policy
      text("t = " + st.t.toFixed(2) + "   ·   campaign end T hidden from the bidder", 16, H - 14, 11, colors.muted, { mono: true });
      const pol = st.c.policy === "fipla" ? "FiPla: bids V₀" : st.c.policy === "shading" ? "Shading: bids " + st.c.alpha.toFixed(1) + " × V₀" : "Constant: bids " + st.c.cst.toFixed(2);
      text(pol, W - 16, H - 14, 11, colors.ink2, { mono: true, align: "right" });

      if (ph === "end") {
        g.fillStyle = colors.surface; g.globalAlpha = 0.92; g.fillRect(0, 0, W, H); g.globalAlpha = 1;
        text("End of the campaign", W / 2, 150, 26, colors.ink, { weight: 700, align: "center" });
        const net = st.convN - st.paid;
        text(st.auctions.length + " auctions · " + st.done.filter((d) => d.win).length + " won · paid " + f2(st.paid) + " · " + st.convN + " conversion" + (st.convN === 1 ? "" : "s"),
          W / 2, 188, 14, colors.ink2, { mono: true, align: "center" });
        text("net reward " + (net >= 0 ? "+" : "") + f2(net), W / 2, 226, 22, colors.accent, { weight: 700, align: "center", mono: true });
        if (st.lost) text(st.lost + " click(s) after the last auction were not credited", W / 2, 258, 12, colors.muted, { mono: true, align: "center" });
        text("Change the policy and replay the same campaign to compare.", W / 2, 300, 13, colors.muted, { align: "center" });
      } else if (!running && st.t === 0 && st.phase === "travel") {
        text("Choose the primitives and the policy, then press Start.", W / 2, 150, 15, colors.muted, { align: "center" });
      }
    }

    /* ---------------- wiring ---------------- */
    ["auction", "competition", "arrivals", "horizon", "context", "obs", "conv", "policy"].forEach((k) =>
      els[k].addEventListener("change", () => { syncControls(); reset(); }));
    ["alpha", "cst"].forEach((k) => els[k].addEventListener("input", () => { syncControls(); reset(); }));
    function proceed() { S.waiting = false; go("bid"); setRunning(true); }
    canvas.addEventListener("click", () => { if (S && S.waiting) proceed(); });
    els.step.addEventListener("change", () => { if (!els.step.checked && S.waiting) proceed(); });
    els.play.addEventListener("click", () => {
      if (S.waiting) { proceed(); return; }
      if (S.phase === "end") { reset(); setRunning(true); return; }
      setRunning(!running);
    });
    els.reset.addEventListener("click", reset);
    els.fresh.addEventListener("click", () => { seed = 1 + Math.floor(Math.random() * 99999); syncControls(); reset(); });
    window.addEventListener("resize", draw);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(draw);

    readColors();
    syncControls();
    reset();
  }

  window.DynamDemo = { buildCampaign, makeContext, contextValue, contextWin, convProb, DELTA };
  function initToc() {
    const links = Array.from(document.querySelectorAll(".sidetoc a"));
    if (!links.length || !("IntersectionObserver" in window)) return;
    const targets = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
    const visible = new Map();
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => visible.set(e.target.id, e.isIntersecting ? e.intersectionRatio : 0));
      // the active entry: the last target whose top has passed the upper third of the screen
      let active = targets[0].id;
      targets.forEach((t) => { if (t.getBoundingClientRect().top < window.innerHeight / 3) active = t.id; });
      links.forEach((a) => a.classList.toggle("is-active", a.getAttribute("href") === "#" + active));
    }, { threshold: [0, 0.25, 0.5, 1] });
    targets.forEach((t) => io.observe(t));
    window.addEventListener("scroll", () => {
      let active = targets[0].id;
      targets.forEach((t) => { if (t.getBoundingClientRect().top < window.innerHeight / 3) active = t.id; });
      links.forEach((a) => a.classList.toggle("is-active", a.getAttribute("href") === "#" + active));
    }, { passive: true });
  }
  if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", () => { initPipeline(); initDemo(); initToc(); });
})();
