/* The Knight's Merchant: a small planning game for the DynamAuction website.
 *
 * Model. The knight orders K = 5 of the 8 pieces below and pays `value` for each. The market has N = 20 stands,
 * visited in order. Stand k sells one piece still missing, chosen uniformly, at an integer price uniform in
 * [round(0.3 v), round(1.2 v)]. Buying removes the piece from all later stands. Profit = sum over bought pieces of
 * (value - price); a missing piece earns 0.
 * A market is a seed: the order and two uniforms per stand (which piece, which price quantile). The player, the
 * myopic merchant and the planner all play the same uniforms (common random numbers), so they meet the same market,
 * although not always the same pieces once their purchases differ.
 * Planner: the exact optimal policy, by dynamic programming over (pieces still missing S, stands left n):
 *   V(S, n) = mean_{i in S} E_p[ max(v_i - p + V(S \ i, n - 1), V(S, n - 1)) ],
 * so at a stand it buys iff p <= v_i + V(S \ i, n - 1) - V(S, n - 1).
 * Myopic merchant: buys iff p <= v_i. */
(function () {
  "use strict";

  // ------------------------------------------------------------------ model
  const ITEMS = [
    { key: "helmet", name: "Helmet", value: 40 },
    { key: "sword",  name: "Sword",  value: 55 },
    { key: "shield", name: "Shield", value: 35 },
    { key: "horse",  name: "Horse",  value: 90 },
    { key: "bow",    name: "Bow",    value: 30 },
    { key: "armor",  name: "Armor",  value: 70 },
    { key: "boots",  name: "Boots",  value: 20 },
    { key: "ring",   name: "Ring",   value: 60 },
  ];
  const N = 20, K = 5, LO = 0.3, HI = 1.2;

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function makeMarket(seed) {
    const r = mulberry32(seed);
    const keys = ITEMS.map((x) => x.key);
    for (let i = keys.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [keys[i], keys[j]] = [keys[j], keys[i]]; }
    const chosen = new Set(keys.slice(0, K));
    const items = ITEMS.filter((x) => chosen.has(x.key))
      .map((x) => ({ ...x, lo: Math.round(LO * x.value), hi: Math.round(HI * x.value) }));
    const U = Array.from({ length: N }, () => [r(), r()]);
    return { seed, items, U, ...solve(items) };
  }

  // V[n][S]: optimal expected profit; M[n][S]: expected profit of the myopic merchant
  function solve(items) {
    const F = 1 << items.length;
    const V = [new Float64Array(F)], M = [new Float64Array(F)];
    for (let n = 1; n <= N; n++) {
      const v = new Float64Array(F), m = new Float64Array(F);
      for (let S = 1; S < F; S++) {
        let ev = 0, em = 0, c = 0;
        for (let i = 0; i < items.length; i++) {
          if (!((S >> i) & 1)) continue;
          c++;
          const it = items[i], S2 = S & ~(1 << i), np = it.hi - it.lo + 1;
          const stay = V[n - 1][S], go = V[n - 1][S2], mstay = M[n - 1][S], mgo = M[n - 1][S2];
          let a = 0, b = 0;
          for (let p = it.lo; p <= it.hi; p++) {
            a += Math.max(it.value - p + go, stay);
            b += p <= it.value ? it.value - p + mgo : mstay;
          }
          ev += a / np; em += b / np;
        }
        v[S] = ev / c; m[S] = em / c;
      }
      V.push(v); M.push(m);
    }
    return { V, M };
  }

  const missing = (S, K_) => { const r = []; for (let i = 0; i < K_; i++) if ((S >> i) & 1) r.push(i); return r; };

  function offerAt(mk, k, S) {
    const idx = missing(S, mk.items.length);
    const i = idx[Math.floor(mk.U[k][0] * idx.length)];
    const it = mk.items[i];
    return { i, price: it.lo + Math.floor(mk.U[k][1] * (it.hi - it.lo + 1)) };
  }

  // highest price at which the planner buys piece i at stand k (0-based) with missing set S
  function plannerMax(mk, S, k, i) {
    const n = N - k;
    return Math.floor(mk.items[i].value + mk.V[n - 1][S & ~(1 << i)] - mk.V[n - 1][S] + 1e-9);
  }

  function simulate(mk, who) {
    let S = (1 << mk.items.length) - 1, profit = 0;
    for (let k = 0; k < N && S; k++) {
      const o = offerAt(mk, k, S), it = mk.items[o.i];
      const buy = who === "planner" ? o.price <= plannerMax(mk, S, k, o.i) : o.price <= it.value;
      if (buy) { profit += it.value - o.price; S &= ~(1 << o.i); }
    }
    return profit;
  }

  // ------------------------------------------------------------------ assets
  const ASSET = "assets/img/game/";
  const img = {};
  function load(name) {
    return new Promise((res) => { const im = new Image(); im.onload = () => res(); im.onerror = () => res(); im.src = ASSET + name + ".png"; img[name] = im; });
  }

  // ------------------------------------------------------------------ scene geometry (canvas units)
  const W = 960, H = 540;
  const ROAD_Y = 452, STAND_BOTTOM = 470, MERCH_BOTTOM = 516;
  const SPACING = 660, STAND_OFF = 232, MSX = 330;           // merchant screen x, and his stop left of a stand
  const standX = (k) => 1000 + k * SPACING;                   // world x of the centre of stand k
  const ARCH_X = standX(0) - STAND_OFF - 380;                 // market gate
  const START_X = ARCH_X - 250;
  const SW = 258, SH = 325;                                   // stand sprite
  const MS = 0.86;                                            // merchant scale

  // ------------------------------------------------------------------ DOM
  const $ = (id) => document.getElementById(id);
  let canvas, ctx, dpr = 1;
  const fmtSigned = (x) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x);

  // ------------------------------------------------------------------ game state
  let g = null;
  let hint = false;

  function newGame(seed) {
    const mk = makeMarket(seed);
    g = {
      mk, S: (1 << mk.items.length) - 1, k: -1, profit: 0, log: [], phase: "intro",
      mx: START_X, step: 0, walk: null, offer: null, next: null, stands: [], castleX: null,
      coins: [], texts: [], timers: [], bubble: null, hop: -1, time: 0, status: {},
    };
    mk.items.forEach((it, i) => { g.status[i] = "needed"; });
    $("seed").textContent = seed;
    $("intro-items").innerHTML = mk.items.map((it) =>
      `<div><img src="${ASSET}icon-${it.key}.png" alt=""><span>${it.name}</span><b>${it.value}</b></div>`).join("");
    $("intro").hidden = false;
    $("end").hidden = true;
    $("offer").hidden = true;
    renderOrder(); renderHud(); renderHistory();
    $("recap").hidden = true;
  }

  function later(t, fn) { g.timers.push({ t, fn }); }

  function startWalk(to, cb) {
    const from = g.mx;
    g.walk = { from, to, t: 0, dur: Math.max(0.9, Math.abs(to - from) / 470), cb };
    g.phase = "walk";
    if (!g.early) g.bubble = null;
  }

  function goNext() {
    const k = g.k + 1;
    if (g.S === 0 || k >= N) { goCastle(); return; }
    const o = offerAt(g.mk, k, g.S);
    g.next = { k, ...o };
    g.stands.push({ k, key: g.mk.items[o.i].key, x: standX(k) });
    startWalk(standX(k) - STAND_OFF, arrive);
  }

  function arrive() {
    g.k = g.next.k;
    g.offer = g.next;
    g.phase = "offer";
    const it = g.mk.items[g.offer.i];
    g.bubble = { text: `${it.name}: ${g.offer.price} coins!`, x: standX(g.k) };
    showOffer();
    renderOrder(); renderHud();
  }

  function decide(buy) {
    if (!g || g.phase !== "offer") return;
    const o = g.offer, it = g.mk.items[o.i];
    const thr = plannerMax(g.mk, g.S, g.k, o.i);
    g.log.push({ k: g.k, i: o.i, price: o.price, bought: buy, thr, left: N - g.k - 1 });
    renderHistory();
    $("offer").hidden = true;
    if (buy) {
      g.phase = "buy";
      g.bubble = { text: "A fine choice!", x: standX(g.k) };
      const hand = handWorld();
      for (let c = 0; c < 5; c++) g.coins.push({ t: -c * 0.08, from: hand, to: { x: standX(g.k) - 40, y: 372 } });
      later(0.55, () => {
        g.hop = 0;
        g.profit += it.value - o.price;
        g.S &= ~(1 << o.i);
        g.status[o.i] = "bought";
        g.texts.push({ x: g.mx, y: MERCH_BOTTOM - 240, text: fmtSigned(it.value - o.price), t: 0,
                       color: it.value - o.price >= 0 ? "#3c5a2a" : "#9b3b2a" });
        flyIcon(o.i, () => { renderOrder(); });
        renderHud();
      });
      later(1.45, goNext);
    } else {
      g.phase = "pass";
      g.bubble = { text: "Maybe next time…", x: standX(g.k) };
      later(0.45, goNext);
    }
    renderOrder();
  }

  function goCastle() {
    g.offer = null;
    if (g.S === 0 && g.k < N - 1) {
      g.early = N - 1 - g.k;   // stands left unvisited: nothing to buy there any more
      g.bubble = { text: "Order complete! Off to the castle.", follow: true };
      renderHud();
    }
    g.castleX = g.mx + 980;
    startWalk(g.castleX - 150, finish);
  }

  function finish() {
    g.phase = "end";
    g.mk.items.forEach((it, i) => { if (g.status[i] !== "bought") g.status[i] = "missing"; });
    renderOrder(); renderHud();
    g.bubble = null;
    const mk = g.mk;
    const full = (1 << mk.items.length) - 1;
    const planner = simulate(mk, "planner"), myopic = simulate(mk, "myopic");
    const nMissing = mk.items.filter((_, i) => g.status[i] === "missing").length;
    $("e-kicker").textContent = g.early ? `Order complete at stand ${N - g.early} of ${N}` : "At the knight's castle";
    $("e-title").textContent = nMissing === 0 ? "Order complete!"
      : `${nMissing} piece${nMissing > 1 ? "s" : ""} missing`;
    const rows = [
      { who: "You", sub: "", v: g.profit, color: "var(--bid)" },
      { who: "Myopic merchant", sub: "buys whenever price < knight's price", v: myopic, color: "var(--myopic)" },
      { who: "Planner", sub: "the optimal plan", v: planner, color: "var(--accent)" },
    ];
    const top = Math.max(1, ...rows.map((r) => r.v));
    $("e-bars").innerHTML = rows.map((r) =>
      `<div class="bar"><span class="who">${r.who}${r.sub ? `<small>${r.sub}</small>` : ""}</span>` +
      `<span class="track"><span class="fill" style="display:block;width:${Math.max(0, r.v) / top * 100}%;background:${r.color}"></span></span>` +
      `<span class="num">${fmtSigned(r.v)}</span></div>`).join("");
    const gap = planner - g.profit;
    const verdict = gap <= 0 ? "You matched the planner on this market. Well planned!"
      : `The planner made ${gap} more coin${gap > 1 ? "s" : ""} on this market.`;
    const earlyTxt = g.early ? `You had all five pieces, so you skipped the last ${g.early} stand${g.early > 1 ? "s" : ""}: nothing there was worth anything to you. ` : "";
    $("e-note").textContent = `${earlyTxt}${verdict} Same market for all three (same draws). Averaged over all markets with this ` +
      `order, the planner makes ${mk.V[N][full].toFixed(0)} coins and the myopic merchant ${mk.M[N][full].toFixed(0)}.`;
    $("end").hidden = false;
    renderRecap();
    setTimeout(() => $("b-again").focus({ preventScroll: true }), 50);
  }

  // ------------------------------------------------------------------ DOM rendering
  function renderOrder() {
    const cur = g.phase === "offer" && g.offer ? g.offer.i : -1;
    $("order").innerHTML = g.mk.items.map((it, i) => {
      const st = g.status[i];
      const rec = g.log.find((l) => l.i === i && l.bought);
      let right = "";
      if (st === "bought" && rec) right = `✓ ${rec.price} · ${fmtSigned(it.value - rec.price)}`;
      else if (st === "missing") right = "✗ not found";
      else if (i === cur) right = "on offer";
      const cls = [st, i === cur ? "offered" : ""].join(" ");
      return `<li class="${cls}" data-i="${i}"><img src="${ASSET}icon-${it.key}.png" alt="">` +
        `<span class="nm">${it.name}<span class="val">pays ${it.value}</span></span><span class="st">${right}</span></li>`;
    }).join("");
  }

  function renderHud() {
    const seen = g.k + 1;
    $("h-stand").textContent = `${Math.max(0, seen)} / ${N}`;
    $("h-profit").textContent = fmtSigned(g.profit) || "0";
    let pips = "";
    for (let k = 0; k < N; k++) {
      const l = g.log.find((x) => x.k === k);
      const cls = l ? (l.bought ? "buy" : "past") : (k === g.k && g.phase === "offer" ? "now" : g.early ? "skip" : "");
      pips += `<i class="${cls}"></i>`;
    }
    $("pips").innerHTML = pips;
  }

  function showOffer() {
    const o = g.offer, it = g.mk.items[o.i];
    const left = N - g.k - 1;
    $("o-icon").src = `${ASSET}icon-${it.key}.png`;
    $("o-name").textContent = it.name;
    $("o-sub").textContent = `Stand ${g.k + 1} of ${N} · ${left} more after this one`;
    $("o-price").textContent = o.price;
    $("o-value").textContent = it.value;
    $("o-margin").textContent = fmtSigned(it.value - o.price) || "0";
    $("b-buy").innerHTML = `Buy for ${o.price} <kbd>B</kbd>`;
    renderAdvice();
    $("offer").hidden = false;
  }

  function renderAdvice() {
    const box = $("o-advice");
    if (!hint || !g || g.phase !== "offer") { box.hidden = true; return; }
    const o = g.offer, it = g.mk.items[o.i];
    const thr = plannerMax(g.mk, g.S, g.k, o.i);
    let txt;
    if (thr >= it.hi) txt = "The planner would buy at any price here.";
    else if (thr < it.lo) txt = "The planner would walk on, whatever the price.";
    else txt = `The planner pays up to ${thr} here: it would ${o.price <= thr ? "buy" : "walk on"}.`;
    box.textContent = txt;
    box.hidden = false;
  }

  // live strip under the game: every stand seen so far, newest on the right
  function renderHistory() {
    const list = $("hist");
    if (!g.log.length) {
      list.innerHTML = `<li class="hist-empty">none yet: your stands will appear here as you walk</li>`;
      return;
    }
    list.innerHTML = g.log.map((l, j) => {
      const it = g.mk.items[l.i];
      const what = l.bought ? `✓ ${l.price}` : `${l.price} · passed`;
      return `<li class="${l.bought ? "b" : ""}${j === g.log.length - 1 ? " new" : ""}" title="Stand ${l.k + 1}: ${it.name} at ${l.price}, ${l.bought ? "bought" : "walked on"}">` +
        `<img src="${ASSET}icon-${it.key}.png" alt=""><span class="hn">#${l.k + 1} ${it.name}</span><span class="hp">${what}</span></li>`;
    }).join("");
  }

  function renderRecap() {
    $("recap-body").innerHTML = g.log.map((l) => {
      const it = g.mk.items[l.i];
      const pBuy = l.price <= l.thr;
      const thrTxt = l.thr >= it.hi ? "any price" : l.thr < it.lo ? "never" : `up to ${l.thr}`;
      return `<tr class="${pBuy !== l.bought ? "diff" : ""}"><td>${l.k + 1}</td>` +
        `<td><img src="${ASSET}icon-${it.key}.png" alt="">${it.name}</td><td class="n">${l.price}</td><td class="n">${it.value}</td>` +
        `<td>${l.bought ? `<span class="tag b">bought · ${fmtSigned(it.value - l.price)}</span>` : `<span class="tag p">walked on</span>`}</td>` +
        `<td>${pBuy ? "buy" : "walk on"} <span style="color:var(--muted)">(pays ${thrTxt})</span></td></tr>`;
    }).join("");
    $("recap").hidden = false;
  }

  // fly the bought piece from the stand to its line on the parchment
  function flyIcon(i, done) {
    const stage = $("stage"), cr = canvas.getBoundingClientRect(), sr = stage.getBoundingClientRect();
    const row = $("order").querySelector(`li[data-i="${i}"] img`);
    if (!row || matchMedia("(prefers-reduced-motion: reduce)").matches) { done(); return; }
    const sx = (standX(g.k) + 70 - camX()) / W * cr.width + cr.left - sr.left;
    const sy = (STAND_BOTTOM - 130) / H * cr.height + cr.top - sr.top;
    const tr = row.getBoundingClientRect();
    const el = document.createElement("img");
    el.src = `${ASSET}icon-${g.mk.items[i].key}.png`;
    el.className = "fly"; el.alt = "";
    el.style.left = sx - cr.width * 0.02 + "px"; el.style.top = sy - cr.width * 0.02 + "px";
    stage.appendChild(el);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      el.style.left = tr.left - sr.left + "px"; el.style.top = tr.top - sr.top + "px"; el.style.width = tr.width + "px";
    }));
    setTimeout(() => { el.remove(); done(); }, 850);
  }

  // ------------------------------------------------------------------ drawing
  const camX = () => g.mx - MSX;
  const merchW = () => (img.merchant.naturalWidth || 180) * MS, merchH = () => (img.merchant.naturalHeight || 244) * MS;
  const handWorld = () => ({ x: g.mx - merchW() / 2 + 0.89 * merchW(), y: MERCH_BOTTOM - merchH() + 0.31 * merchH() });

  function hash(n) { let x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); }

  function rect(c, x, y, w, h) { ctx.fillStyle = c; ctx.fillRect(Math.round(x), Math.round(y), Math.round(w), Math.round(h)); }

  function drawBackground(cam, progress) {
    rect("#cfe2e0", 0, 0, W, H);
    // clouds
    for (let j = 0; j < 5; j++) {
      const x = ((j * 260 - cam * 0.06) % (W + 260) + W + 260) % (W + 260) - 130, y = 34 + hash(j) * 70;
      rect("#eef5f1", x, y, 110, 14); rect("#eef5f1", x + 18, y - 12, 64, 14); rect("#eef5f1", x + 40, y - 22, 30, 12);
    }
    // far hills, with the knight's castle getting closer
    for (let x = 0; x < W; x += 12) {
      const u = x + cam * 0.12;
      const h = 300 + 26 * Math.sin(u / 150) + 12 * Math.sin(u / 47);
      rect("#a9c3a2", x, h, 12, H - h);
    }
    const cx = W * 1.08 - progress * W * 0.42, base = 318;
    rect("#7d939a", cx - 50, base - 46, 100, 46); rect("#7d939a", cx - 62, base - 76, 26, 76); rect("#7d939a", cx + 36, base - 76, 26, 76);
    rect("#7d939a", cx - 14, base - 92, 28, 92);
    for (let b = 0; b < 3; b++) { rect("#7d939a", cx - 62 + b * 10, base - 84, 6, 8); rect("#7d939a", cx + 36 + b * 10, base - 84, 6, 8); }
    rect("#c4623f", cx - 1, base - 112, 3, 20); rect("#c4623f", cx + 2, base - 112, 14, 8);
    for (let x = 0; x < W; x += 12) {
      const u = x + cam * 0.22;
      const h = 340 + 20 * Math.sin(u / 110 + 1) + 8 * Math.sin(u / 37);
      rect("#93b38a", x, h, 12, H - h);
    }
    // houses behind the stands
    const hc = cam * 0.5, HW = 240;
    for (let j = Math.floor((hc - 260) / HW); j <= Math.floor((hc + W) / HW); j++) {
      const x = j * HW - hc + hash(j) * 30, w = 150 + hash(j + 7) * 60, h = 110 + hash(j + 3) * 70, y = ROAD_Y - h;
      const walls = ["#ecdcb8", "#e3cfa6", "#efe3c8", "#dcc49a"], roofs = ["#a5523a", "#8c4a3a", "#5f6f7a", "#7b5a3c"];
      rect(walls[Math.floor(hash(j + 1) * 4)], x, y, w, h);
      // timber frame
      rect("#6b4a32", x, y, w, 5); rect("#6b4a32", x, y, 5, h); rect("#6b4a32", x + w - 5, y, 5, h); rect("#6b4a32", x, y + h * 0.45, w, 4);
      // roof (stepped)
      const roof = roofs[Math.floor(hash(j + 2) * 4)];
      for (let s = 0; s < 6; s++) rect(roof, x - 12 + s * 9, y - (s + 1) * 9, w + 24 - s * 18, 9);
      // windows and door
      rect("#5b6f78", x + 18, y + 18, 22, 22); rect("#5b6f78", x + w - 40, y + 18, 22, 22);
      rect("#6b4a32", x + w / 2 - 14, y + h - 46, 28, 46);
    }
    // grass edge and road
    rect("#86a76a", 0, ROAD_Y - 8, W, 10);
    rect("#d3bd94", 0, ROAD_Y, W, H - ROAD_Y);
    for (let r = 0; r < 5; r++) {
      const y = ROAD_Y + 10 + r * 17, off = (r % 2) * 22;
      for (let i = Math.floor((cam - off) / 44) - 1; i * 44 + off - cam < W; i++) {
        const x = i * 44 + off - cam;
        rect("#c1a87c", x + 3, y, 34, 9); rect("#dcc8a2", x + 5, y, 28, 3);
      }
    }
  }

  function drawGate(x, label) {
    rect("#6b4a32", x - 120, ROAD_Y - 200, 18, 200); rect("#6b4a32", x + 102, ROAD_Y - 200, 18, 200);
    rect("#5a3d27", x - 132, ROAD_Y - 222, 264, 30);
    rect("#f4e4bc", x - 96, ROAD_Y - 217, 192, 20);
    ctx.fillStyle = "#3b2a1a"; ctx.font = "700 16px 'Pixelify Sans', monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(label, x, ROAD_Y - 206);
    // bunting
    const cols = ["#c4623f", "#f2c14e", "#22566b", "#8fb26f"];
    for (let i = 0; i < 10; i++) {
      ctx.fillStyle = cols[i % 4];
      ctx.beginPath(); ctx.moveTo(x - 110 + i * 22, ROAD_Y - 190); ctx.lineTo(x - 92 + i * 22, ROAD_Y - 190); ctx.lineTo(x - 101 + i * 22, ROAD_Y - 176); ctx.fill();
    }
  }

  function drawCastle(x) {
    const b = ROAD_Y;
    rect("#8e9a9c", x - 170, b - 230, 340, 230);
    rect("#7b878a", x - 230, b - 300, 90, 300); rect("#7b878a", x + 140, b - 300, 90, 300);
    for (let i = 0; i < 4; i++) { rect("#7b878a", x - 230 + i * 24, b - 318, 14, 18); rect("#7b878a", x + 140 + i * 24, b - 318, 14, 18); }
    for (let i = 0; i < 7; i++) rect("#8e9a9c", x - 170 + i * 50, b - 248, 30, 18);
    rect("#4a3524", x - 56, b - 130, 112, 130);
    rect("#3a2a1c", x - 50, b - 124, 100, 124);
    for (let i = 0; i < 5; i++) rect("#5a4430", x - 46 + i * 20, b - 124, 4, 124);
    rect("#22566b", x - 214, b - 270, 46, 80); rect("#f2c14e", x - 196, b - 250, 10, 30);
    rect("#22566b", x + 168, b - 270, 46, 80); rect("#f2c14e", x + 186, b - 250, 10, 30);
    rect("#c4623f", x - 2, b - 300, 4, 54); rect("#c4623f", x + 2, b - 300, 34, 20);
  }

  function drawMerchant(cam) {
    const m = img.merchant, w = merchW(), h = merchH();
    const walking = g.phase === "walk";
    let dy = walking ? -Math.abs(Math.sin(g.step / 26)) * 7 : 0;
    if (g.hop >= 0) dy -= Math.sin(Math.PI * Math.min(1, g.hop / 0.5)) * 24;
    const tilt = walking ? Math.sin(g.step / 26) * 0.035 : 0;
    const breathe = walking ? 1 : 1 + 0.008 * Math.sin(g.time * 2.4);
    const x = g.mx - cam;
    ctx.fillStyle = "rgba(60, 40, 20, .22)";
    ctx.beginPath(); ctx.ellipse(x, MERCH_BOTTOM - 2, w * 0.36 * (1 + dy / 80), 9, 0, 0, Math.PI * 2); ctx.fill();
    ctx.save();
    ctx.translate(x, MERCH_BOTTOM + dy);
    ctx.rotate(tilt);
    ctx.scale(1, breathe);
    if (m.complete && m.naturalWidth) ctx.drawImage(m, -w / 2, -h, w, h);
    ctx.restore();
    // dust puffs while walking
    if (walking) {
      const ph = (g.step / 52) % 1;
      ctx.fillStyle = `rgba(200, 180, 140, ${0.55 * (1 - ph)})`;
      ctx.beginPath(); ctx.arc(x - w * 0.3 - ph * 26, MERCH_BOTTOM - 6 - ph * 8, 5 + ph * 7, 0, Math.PI * 2); ctx.fill();
    }
  }

  function drawBubble(cam) {
    if (!g.bubble) return;
    const x = g.bubble.follow ? g.mx - cam + 70 : g.bubble.x - cam + 30, y = g.bubble.follow ? 200 : 92;
    ctx.font = "700 20px 'IBM Plex Sans', system-ui, sans-serif";
    const tw = ctx.measureText(g.bubble.text).width, bw = tw + 32, bh = 40;
    const bx = Math.max(8, Math.min(W - bw - 8, x - bw / 2));
    rect("#2f1f14", bx - 3, y - 3, bw + 6, bh + 6);
    rect("#fffaf0", bx, y, bw, bh);
    ctx.fillStyle = "#2f1f14";
    ctx.beginPath(); ctx.moveTo(x - 40, y + bh); ctx.lineTo(x - 20, y + bh); ctx.lineTo(x - 50, y + bh + 22); ctx.fill();
    ctx.fillStyle = "#fffaf0";
    ctx.beginPath(); ctx.moveTo(x - 37, y + bh - 1); ctx.lineTo(x - 24, y + bh - 1); ctx.lineTo(x - 45, y + bh + 14); ctx.fill();
    ctx.fillStyle = "#3b2a1a"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(g.bubble.text, bx + bw / 2, y + bh / 2 + 1);
  }

  function drawCoin(x, y, r) {
    ctx.fillStyle = "#8a5a12"; ctx.beginPath(); ctx.arc(x, y, r + 2, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#f2c14e"; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#fff1b8"; ctx.fillRect(Math.round(x - r / 2), Math.round(y - r / 2), 3, 3);
  }

  function draw() {
    const cam = camX();
    const end = g.castleX || standX(N) + 200;
    const progress = Math.max(0, Math.min(1, (g.mx - START_X) / (end - START_X)));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawBackground(cam, progress);
    if (ARCH_X - cam > -200 && ARCH_X - cam < W + 200) drawGate(ARCH_X - cam, "MARKET");
    for (const s of g.stands) {
      const x = s.x - cam;
      if (x < -SW || x > W + SW) continue;
      const im = img["stand-" + s.key];
      if (im && im.complete && im.naturalWidth) ctx.drawImage(im, Math.round(x - SW / 2), STAND_BOTTOM - SH, SW, SH);
    }
    if (g.castleX && g.castleX - cam < W + 300) drawCastle(g.castleX - cam + 140);
    // coins
    for (const c of g.coins) {
      if (c.t < 0 || c.t > 0.55) continue;
      const s = c.t / 0.55;
      const x = c.from.x + (c.to.x - c.from.x) * s - cam;
      const y = c.from.y + (c.to.y - c.from.y) * s - Math.sin(Math.PI * s) * 70;
      drawCoin(x, y, 7);
    }
    drawMerchant(cam);
    drawBubble(cam);
    for (const t of g.texts) {
      ctx.globalAlpha = Math.max(0, 1 - t.t / 1.3);
      ctx.font = "700 28px 'IBM Plex Mono', ui-monospace, monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.lineWidth = 5; ctx.strokeStyle = "#fffaf0"; ctx.strokeText(t.text, t.x - cam, t.y - t.t * 40);
      ctx.fillStyle = t.color; ctx.fillText(t.text, t.x - cam, t.y - t.t * 40);
      ctx.globalAlpha = 1;
    }
  }

  // ------------------------------------------------------------------ loop
  let lastT = 0;
  function frame(now) {
    const dt = Math.min(0.05, (now - (lastT || now)) / 1000);
    lastT = now;
    if (g) {
      g.time += dt;
      if (g.walk) {
        const w = g.walk;
        w.t += dt;
        const s = Math.min(1, w.t / w.dur);
        const e = s < 0.5 ? 2 * s * s : 1 - 2 * (1 - s) * (1 - s);
        const nx = w.from + (w.to - w.from) * e;
        g.step += Math.abs(nx - g.mx);
        g.mx = nx;
        if (s >= 1) { g.walk = null; w.cb(); }
      }
      for (const c of g.coins) c.t += dt;
      g.coins = g.coins.filter((c) => c.t <= 0.6);
      for (const t of g.texts) t.t += dt;
      g.texts = g.texts.filter((t) => t.t < 1.3);
      if (g.hop >= 0) { g.hop += dt; if (g.hop > 0.5) g.hop = -1; }
      const due = g.timers.filter((t) => (t.t -= dt) <= 0);
      g.timers = g.timers.filter((t) => t.t > 0);
      due.forEach((t) => t.fn());
      draw();
    }
    requestAnimationFrame(frame);
  }

  function resize() {
    dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = W * dpr; canvas.height = H * dpr;
  }

  function newSeed() { return 1 + Math.floor(Math.random() * 99999); }

  function init() {
    canvas = $("scene");
    ctx = canvas.getContext("2d");
    resize();
    window.addEventListener("resize", resize);
    try { hint = localStorage.getItem("km-hint") === "1"; } catch (e) { hint = false; }
    $("hint").checked = hint;
    $("hint").addEventListener("change", (e) => {
      hint = e.target.checked;
      try { localStorage.setItem("km-hint", hint ? "1" : "0"); } catch (err) { /* storage unavailable */ }
      renderAdvice();
    });
    $("b-start").addEventListener("click", () => { if (g.phase !== "intro") return; $("intro").hidden = true; goNext(); });
    $("b-buy").addEventListener("click", () => decide(true));
    $("b-pass").addEventListener("click", () => decide(false));
    $("b-again").addEventListener("click", () => newGame(newSeed()));
    $("b-replay").addEventListener("click", () => newGame(g.mk.seed));
    document.addEventListener("keydown", (e) => {
      if (!g || e.target.closest("input, textarea, select")) return;
      if (g.phase === "intro" && e.key === "Enter" && !e.target.closest("button, a")) { e.preventDefault(); $("b-start").click(); }
      if (g.phase === "offer") {
        if (e.key === "b" || e.key === "B") { e.preventDefault(); decide(true); }
        else if (e.key === "ArrowRight" || e.key === " " || e.key === "n" || e.key === "N") { e.preventDefault(); decide(false); }
      }
    });
    const names = ["merchant", ...ITEMS.map((x) => "stand-" + x.key)];
    newGame(newSeed());
    Promise.all(names.map(load)).then(() => document.fonts && document.fonts.load("700 20px 'Pixelify Sans'"));
    requestAnimationFrame(frame);
  }

  window.KnightsMerchant = { makeMarket, simulate, plannerMax, offerAt, N };
  document.addEventListener("DOMContentLoaded", init);
})();
