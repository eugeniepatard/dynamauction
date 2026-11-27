/* The Knight's Merchant: a small planning game for the DynamAuction website.
 *
 * Model. The knight orders K = 5 of the 8 pieces below and pays `value` for each. The market has N = 20 stands,
 * visited in order. Stand k sells one of the K pieces, chosen uniformly (owned or not), at an integer price uniform in
 * the piece's range [lo, hi]. The merchant may buy any number of copies; every copy is paid, and the knight takes one
 * copy of each piece (the cheapest one: it leaves the most profit). Profit = sum over the pieces owned of value - sum of
 * all the prices paid; a missing piece earns 0, an extra copy loses its price. Extra copies are there to show the
 * regret of an early purchase: a later stand may sell the same piece cheaper.
 * A market is a seed: the order and, for each ordered piece, its price range. Ranges differ from piece to piece and
 * from market to market, so playing many markets does not teach one fixed distribution; they are written on the
 * parchment, so the player knows what the planner knows. A day is a second seed: two uniforms per stand (which
 * piece, which price quantile). "Replay" keeps the market and draws a new day. The player, the myopic merchant and
 * the planner all play the same day (common random numbers), so they meet the same stands, although not always the
 * same pieces once their purchases differ.
 * Planner: the exact optimal policy, by dynamic programming over (pieces still missing S, stands left n):
 *   V(S, n) = (1/K) [ sum_{i in S} E_p[ max(v_i - p + V(S \ i, n - 1), V(S, n - 1)) ] + (K - |S|) V(S, n - 1) ],
 * so at a stand it buys a missing piece iff p <= v_i + V(S \ i, n - 1) - V(S, n - 1), and never a second copy (it
 * earns nothing and costs its price).
 * Myopic merchant: buys a missing piece iff p <= v_i, never a second copy. */
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
  const N = 20, K = 5;
  // price range of a piece, in fractions of its value: centre in [C0, C1], half-width in [H0, H1]
  // (the former fixed range [0.3 v, 1.2 v] is centre 0.75, half-width 0.45)
  const C0 = 0.45, C1 = 0.95, H0 = 0.15, H1 = 0.5;

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // market `seed` fixes the order and the price ranges; `day` fixes the stands (defaults to the market seed)
  function makeMarket(seed, day = seed) {
    const r = mulberry32(seed);
    const keys = ITEMS.map((x) => x.key);
    for (let i = keys.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [keys[i], keys[j]] = [keys[j], keys[i]]; }
    const chosen = new Set(keys.slice(0, K));
    const items = ITEMS.filter((x) => chosen.has(x.key)).map((x) => {
      const c = C0 + (C1 - C0) * r(), h = H0 + (H1 - H0) * r();
      const lo = Math.max(1, Math.round((c - h) * x.value));
      return { ...x, lo, hi: Math.max(lo + 2, Math.round((c + h) * x.value)) };
    }).sort((a, b) => b.value - a.value);         // most valuable first, as on the parchment
    const d = mulberry32(day ^ 0x5bd1e995);
    const U = Array.from({ length: N }, () => [d(), d()]);
    return { seed, day, items, U, ...solve(items) };
  }

  // V[n][S]: optimal expected profit; M[n][S]: expected profit of the myopic merchant
  function solve(items) {
    const F = 1 << items.length;
    const V = [new Float64Array(F)], M = [new Float64Array(F)];
    for (let n = 1; n <= N; n++) {
      const v = new Float64Array(F), m = new Float64Array(F);
      for (let S = 1; S < F; S++) {
        let ev = 0, em = 0;
        for (let i = 0; i < items.length; i++) {
          if (!((S >> i) & 1)) { ev += V[n - 1][S]; em += M[n - 1][S]; continue; }   // owned piece: walk on
          const it = items[i], S2 = S & ~(1 << i), np = it.hi - it.lo + 1;
          const stay = V[n - 1][S], go = V[n - 1][S2], mstay = M[n - 1][S], mgo = M[n - 1][S2];
          let a = 0, b = 0;
          for (let p = it.lo; p <= it.hi; p++) {
            a += Math.max(it.value - p + go, stay);
            b += p <= it.value ? it.value - p + mgo : mstay;
          }
          ev += a / np; em += b / np;
        }
        v[S] = ev / items.length; m[S] = em / items.length;
      }
      V.push(v); M.push(m);
    }
    return { V, M };
  }

  // stand k sells piece i (any of the K, owned or not) at a price of its range: fixed by the day's uniforms
  function offerAt(mk, k) {
    const i = Math.floor(mk.U[k][0] * mk.items.length);
    const it = mk.items[i];
    return { i, price: it.lo + Math.floor(mk.U[k][1] * (it.hi - it.lo + 1)) };
  }

  // highest price at which the planner buys piece i at stand k (0-based) with missing set S; -1: never (owned)
  function plannerMax(mk, S, k, i) {
    if (!((S >> i) & 1)) return -1;
    const n = N - k;
    return Math.floor(mk.items[i].value + mk.V[n - 1][S & ~(1 << i)] - mk.V[n - 1][S] + 1e-9);
  }

  function simulate(mk, who) {
    let S = (1 << mk.items.length) - 1, profit = 0;
    for (let k = 0; k < N && S; k++) {
      const o = offerAt(mk, k), it = mk.items[o.i];
      if (!((S >> o.i) & 1)) continue;   // owned: neither buys a second copy
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

  // competition: a market code everyone plays on its first day (day seed = market seed), so the stands are the same
  let comp = null;
  const todayCode = () => {
    const d = new Date(), k = d.getFullYear() * 10000 + (d.getMonth() + 1) * 100 + d.getDate();
    return 1 + (Math.imul(k ^ 0x2c1b3c6d, 0x297a2d39) >>> 0) % 99999;
  };

  function newGame(seed, visit = 1) {
    if (comp) { seed = comp; visit = 1; }
    const mk = makeMarket(seed, visit === 1 ? seed : newSeed());
    g = {
      visit,
      mk, S: (1 << mk.items.length) - 1, k: -1, profit: 0, log: [], phase: "intro",
      mx: START_X, step: 0, walk: null, offer: null, next: null, stands: [], castleX: null,
      coins: [], texts: [], timers: [], bubble: null, hop: -1, time: 0, status: {}, paid: {},
    };
    mk.items.forEach((it, i) => { g.status[i] = "needed"; g.paid[i] = []; });
    $("intro-warn").hidden = true;
    $("seed").textContent = comp ? `Competition #${seed}: everyone meets these same stands` : visit === 1 ? `Market #${seed}`
      : `Market #${seed}, day ${visit}: same order and price ranges, new stands`;
    $("intro-items").innerHTML = mk.items.map((it) =>
      `<div><img src="${ASSET}icon-${it.key}.png" alt=""><span>${it.name}</span><b>${it.value}</b>` +
      `<small>sold ${it.lo}–${it.hi}</small></div>`).join("");
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

  const owned = (i) => !((g.S >> i) & 1);
  const kept = (i) => Math.min(...g.paid[i]);      // the copy the knight takes: the cheapest

  function goNext() {
    const k = g.k + 1;
    if (k >= N) { goCastle(); return; }
    const o = offerAt(g.mk, k);
    g.next = { k, ...o };
    g.stands.push({ k, key: g.mk.items[o.i].key, x: standX(k) });
    startWalk(standX(k) - STAND_OFF, arrive);
  }

  function arrive() {
    g.k = g.next.k;
    g.offer = g.next;
    g.phase = "offer";
    const it = g.mk.items[g.offer.i];
    const regret = owned(g.offer.i) && g.offer.price < kept(g.offer.i);
    g.bubble = { text: regret ? `${it.name}: ${g.offer.price}… I paid ${kept(g.offer.i)}!` : `${it.name}: ${g.offer.price} coins!`,
                 x: standX(g.k) };
    showOffer();
    renderOrder(); renderHud();
  }

  function decide(buy) {
    if (!g || g.phase !== "offer") return;
    const o = g.offer, it = g.mk.items[o.i];
    const thr = plannerMax(g.mk, g.S, g.k, o.i), dup = owned(o.i);
    g.log.push({ k: g.k, i: o.i, price: o.price, bought: buy, thr, left: N - g.k - 1, dup,
                 prev: dup ? kept(o.i) : null });
    renderHistory();
    $("offer").hidden = true;
    if (buy) {
      g.phase = "buy";
      g.bubble = { text: dup ? "One more, just in case…" : "A fine choice!", x: standX(g.k) };
      const hand = handWorld();
      for (let c = 0; c < 5; c++) g.coins.push({ t: -c * 0.08, from: hand, to: { x: standX(g.k) - 40, y: 372 } });
      later(0.55, () => {
        g.hop = 0;
        const gain = dup ? -o.price : it.value - o.price;   // a second copy earns nothing: only its price counts
        g.profit += gain;
        g.paid[o.i].push(o.price);
        g.S &= ~(1 << o.i);
        g.status[o.i] = "bought";
        g.texts.push({ x: g.mx, y: MERCH_BOTTOM - 240, text: fmtSigned(gain), t: 0, color: gain >= 0 ? "#3c5a2a" : "#9b3b2a" });
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
    $("offer").hidden = true;
    if (g.k < N - 1) {
      g.early = N - 1 - g.k;   // the player chose to skip the last stands (order complete)
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
    const earlyTxt = g.early ? `You had all five pieces and skipped the last ${g.early} stand${g.early > 1 ? "s" : ""}. ` : "";
    const extras = g.log.filter((l) => l.bought && l.dup), exCost = extras.reduce((a, l) => a + l.price, 0);
    const extraTxt = extras.length ? `You bought ${extras.length} extra cop${extras.length > 1 ? "ies" : "y"} for ${exCost} coins: ` +
      `the knight takes only one of each piece, the cheapest. ` : "";
    const ps = Math.round(recordGame(g.profit, myopic, planner));
    const psTxt = comp ? "" : `Planning score ${ps} (100 = the planner, 0 ≈ the myopic merchant), `;
    $("e-note").textContent = `${psTxt}${comp ? "Saved" : "saved"} to the scoreboard as ${playerName()}. ${earlyTxt}${extraTxt}${verdict} All three met the same stands. Averaged over many days at this market ` +
      `(same order, same price ranges), the planner makes ${mk.V[N][full].toFixed(0)} coins and the myopic merchant ` +
      `${mk.M[N][full].toFixed(0)}.`;
    $("end").hidden = false;
    renderRecap();
    if (comp) {
      const rows = loadScores().filter((r) => r.comp === comp).sort((a, b) => b.you - a.you);
      const rank = rows.findIndex((r) => r.you === g.profit && r.name === playerName()) + 1;
      $("e-kicker").textContent = `Competition #${comp} · rank ${rank} of ${rows.length}`;
    }
    setTimeout(() => $("b-again").focus({ preventScroll: true }), 50);
  }

  // ------------------------------------------------------------------ DOM rendering
  function renderOrder() {
    const cur = g.phase === "offer" && g.offer ? g.offer.i : -1;
    $("order").innerHTML = g.mk.items.map((it, i) => {
      const st = g.status[i];
      let right = "";
      if (st === "bought" && g.paid[i].length) {
        const k = kept(i), ex = g.paid[i].length - 1, exCost = g.paid[i].reduce((a, p) => a + p, 0) - k;
        right = `✓ ${k} · ${fmtSigned(it.value - k)}` + (ex ? `<span class="ex">+${ex} extra · −${exCost}</span>` : "");
      }
      else if (st === "missing") right = "✗ not found";
      else right = rangeHtml(it, i === cur ? g.offer.price : null);
      const cls = [st, i === cur ? "offered" : ""].join(" ");
      return `<li class="${cls}" data-i="${i}"><img src="${ASSET}icon-${it.key}.png" alt="">` +
        `<span class="nm">${it.name}<span class="val">pays ${it.value}</span></span><span class="st">${right}</span></li>`;
    }).join("");
  }

  // price range of a piece: "lo–hi" over a bar, green where the price is below the knight's, with the offer marked
  function rangeHtml(it, price) {
    const pos = (x) => Math.min(100, Math.max(0, (x - it.lo) / (it.hi - it.lo) * 100)).toFixed(1) + "%";
    const mark = price === null ? "" : `<i class="mk" style="left:${pos(price)}"></i>`;
    return `<span class="rg">${it.lo}–${it.hi}</span>` +
      `<span class="rb" style="--v:${pos(it.value)}"><i class="tk"></i>${mark}</span>`;
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
    $("o-sub").textContent = `Stand ${g.k + 1} of ${N} · ${left} left after`;
    const dup = owned(o.i);
    $("o-price").textContent = o.price;
    $("o-value").textContent = it.value;   // he pays it once: for one copy only
    $("o-margin").textContent = dup ? fmtSigned(-o.price) : fmtSigned(it.value - o.price) || "0";
    $("b-buy").innerHTML = `Buy ${o.price} <kbd>B</kbd>`;
    const rg = $("o-regret");
    if (dup) {
      const k = kept(o.i), d = k - o.price;
      rg.innerHTML = d > 0 ? `Yours cost <b>${k}</b>, this one <b>${d} less</b>. The knight would take it, but your ${k} are lost: <b>${fmtSigned(-o.price)}</b>.`
        : `You have one (paid <b>${k}</b>); the knight pays for one only.`;
      rg.className = "o-regret" + (d > 0 ? " sting" : "");
      rg.hidden = false;
    } else rg.hidden = true;
    $("b-castle").hidden = g.S !== 0;
    renderAdvice();
    $("offer").hidden = false;
  }

  function renderAdvice() {
    const box = $("o-advice");
    if (!hint || !g || g.phase !== "offer" || owned(g.offer.i)) { box.hidden = true; return; }   // owned: the regret line says it
    const o = g.offer, it = g.mk.items[o.i];
    const thr = plannerMax(g.mk, g.S, g.k, o.i);
    let txt;
    if (owned(o.i)) txt = `The planner would walk on: a second ${it.name.toLowerCase()} earns nothing.`;
    else if (thr >= it.hi) txt = "The planner would buy at any price here.";
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
      const what = l.bought ? (l.dup ? `✓ ${l.price} · extra` : `✓ ${l.price}`) : `${l.price} · passed`;
      const cls = l.bought ? (l.dup ? "x" : "b") : l.dup && l.price < l.prev ? "r" : "";
      const tip = `Stand ${l.k + 1}: ${it.name} at ${l.price}, ${l.bought ? (l.dup ? "bought an extra copy" : "bought") : "walked on"}` +
        (l.dup ? ` (you had one at ${l.prev})` : "");
      return `<li class="${cls}${j === g.log.length - 1 ? " new" : ""}" title="${tip}">` +
        `<img src="${ASSET}icon-${it.key}.png" alt=""><span class="hn">#${l.k + 1} ${it.name}</span><span class="hp">${what}</span></li>`;
    }).join("");
  }

  function renderRecap() {
    $("recap-body").innerHTML = g.log.map((l) => {
      const it = g.mk.items[l.i];
      const pBuy = l.price <= l.thr;
      const thrTxt = l.dup ? "nothing: already has one" : l.thr >= it.hi ? "any price" : l.thr < it.lo ? "never" : `up to ${l.thr}`;
      const you = l.bought ? (l.dup ? `<span class="tag x">extra copy · ${fmtSigned(-l.price)}</span>`
        : `<span class="tag b">bought · ${fmtSigned(it.value - l.price)}</span>`)
        : `<span class="tag p">walked on</span>${l.dup && l.price < l.prev ? ` <span class="rg-note">yours cost ${l.prev}</span>` : ""}`;
      return `<tr class="${pBuy !== l.bought ? "diff" : ""}"><td>${l.k + 1}</td>` +
        `<td><img src="${ASSET}icon-${it.key}.png" alt="">${it.name}</td><td class="n">${l.price}</td><td class="n">${it.value}</td>` +
        `<td>${you}</td>` +
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

  // parchment box in canvas units, or null when it does not lie over the canvas (narrow layout)
  function parchBox() {
    const el = document.querySelector(".parch"), c = canvas.getBoundingClientRect();
    if (!el || !c.width) return null;
    const r = el.getBoundingClientRect(), s = W / c.width;
    if (r.top >= c.bottom || r.bottom <= c.top) return null;
    return { right: (r.right - c.left) * s, bottom: (r.bottom - c.top) * s };
  }

  function drawBubble(cam) {
    if (!g.bubble) return;
    const x = g.bubble.follow ? g.mx - cam + 70 : g.bubble.x - cam + 30, y = g.bubble.follow ? 200 : 92;
    ctx.font = "700 20px 'IBM Plex Sans', system-ui, sans-serif";
    const tw = ctx.measureText(g.bubble.text).width, bw = tw + 32, bh = 40;
    let bx = Math.max(8, Math.min(W - bw - 8, x - bw / 2));
    const p = parchBox();                          // keep the bubble clear of the parchment (an HTML layer above the canvas)
    if (p && bx < p.right + 14 && y < p.bottom) bx = Math.min(W - bw - 8, p.right + 14);
    const tx = Math.max(bx + 34, Math.min(bx + bw - 14, x - 20));   // tail: points down-left, kept under the bubble
    rect("#2f1f14", bx - 3, y - 3, bw + 6, bh + 6);
    rect("#fffaf0", bx, y, bw, bh);
    ctx.fillStyle = "#2f1f14";
    ctx.beginPath(); ctx.moveTo(tx - 20, y + bh); ctx.lineTo(tx, y + bh); ctx.lineTo(tx - 30, y + bh + 22); ctx.fill();
    ctx.fillStyle = "#fffaf0";
    ctx.beginPath(); ctx.moveTo(tx - 17, y + bh - 1); ctx.lineTo(tx - 4, y + bh - 1); ctx.lineTo(tx - 25, y + bh + 14); ctx.fill();
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

  // ------------------------------------------------------------------ scoreboard (localStorage, this browser only)
  const SB_KEY = "km-scores", SB_TOP = 15;
  function loadScores() {
    try { const a = JSON.parse(localStorage.getItem(SB_KEY) || "[]"); return Array.isArray(a) ? a : []; } catch (e) { return []; }
  }
  function saveScores(a) { try { localStorage.setItem(SB_KEY, JSON.stringify(a)); } catch (e) { /* storage unavailable */ } }
  function playerName() { return ($("player").value || "").trim().slice(0, 24) || "Anonymous"; }
  const pct = (a, b) => (b > 0 ? a / b * 100 : null);         // your total as a % of theirs: 100% = as good

  function recordGame(you, myopic, planner) {
    const a = loadScores();
    const full = (1 << g.mk.items.length) - 1, edge = g.mk.V[N][full] - g.mk.M[N][full];
    a.push({ name: playerName(), seed: g.mk.seed, day: g.visit, you, myopic, planner, edge, t: Date.now(), ...(comp ? { comp } : {}) });
    saveScores(a);
    renderScoreboard();
    return planScore(a[a.length - 1]);
  }

  // planning score of one game (normal games only; a competition ranks by profit): 100 = the planner's profit on
  // the same stands; each point below is 1% of what planning is worth on this market (the planner's expected edge
  // over the myopic merchant, from the DP), so the myopic merchant scores 0 on average.
  function planScore(r) {
    const e = r.edge > 0 ? r.edge : Math.max(1, r.planner - r.myopic);
    return 100 * (1 - (r.planner - r.you) / e);
  }

  let sbView = "games", psOpen = false;

  function renderScoreboard() {
    const all = loadScores().map((r) => ({ ...r, ps: planScore(r) }));
    $("ps-box").hidden = !psOpen || (sbView === "comp" && !!comp);   // explanation opens from the column header
    const psHead = (t) => `<th class="n"><button class="th-info" data-ps aria-expanded="${psOpen}" aria-controls="ps-box">${t} <span aria-hidden="true">ⓘ</span></button></th>`;
    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    // against a reference (myopic merchant or planner): a bar diverging from its score (centre line), right/green
    // when you made more, left/red when you made less; full half-width at ±50%. Matching the planner earns a gold star.
    const vs = (you, ref, star) => {
      const x = pct(you, ref);
      if (x === null) return `<td class="n">—</td>`;
      const d = x - 100, tie = Math.abs(d) < 0.5, w = Math.min(50, Math.abs(d)) / 50 * 50;
      const bar = tie ? "" : `<i class="${d > 0 ? "up" : "down"}" style="width:${w.toFixed(1)}%;${d > 0 ? "left" : "right"}:50%"></i>`;
      const txt = tie ? (star ? "★ tie" : "tie") : `${d > 0 ? "▲ +" : "▼ −"}${Math.abs(d).toFixed(0)}%`;
      const cls = tie ? (star ? "top" : "even") : d > 0 ? "up" : "down";
      return `<td class="n my ${cls}"><span class="mbar">${bar}</span><span class="mt">${txt}</span></td>`;
    };
    const psCell = (x) => {
      const r = Math.round(x), top = r >= 100;
      return `<td class="n ps${top ? " top" : r < 0 ? " down" : ""}"><span class="psbar"><i style="width:${Math.max(0, Math.min(100, x)).toFixed(1)}%"></i></span>` +
        `<b>${top ? "★ " : ""}${r < 0 ? "−" + Math.abs(r) : r}</b></td>`;
    };
    const empty = (n) => `<tr><td class="empty" colspan="${n}">No market finished yet: play one and your score appears here.</td></tr>`;
    let head, body;
    if (sbView === "comp" && comp) {
      // same market and stands for all: rank by profit directly
      const rows = all.filter((r) => r.comp === comp).sort((a, b) => b.you - a.you || (a.t || 0) - (b.t || 0));
      const last = rows.reduce((m, r) => Math.max(m, r.t || 0), 0);
      head = `<tr><th>#</th><th>Player</th><th class="n">Profit</th><th class="n">vs myopic</th><th class="n">vs planner</th></tr>`;
      body = rows.length ? rows.map((p, i) =>
        `<tr class="${p.t === last ? "me" : ""}"><td class="rk">${i === 0 ? "🏆" : i + 1}</td><td>${esc(p.name)}</td>` +
        `<td class="n"><b>${fmtSigned(p.you) || "0"}</b></td>${vs(p.you, p.myopic, false)}${vs(p.you, p.planner, true)}</tr>`).join("")
        : `<tr><td class="empty" colspan="5">Nobody has finished competition #${comp} on this device yet.</td></tr>`;
    } else if (sbView === "players") {
      const by = new Map();
      all.forEach((r) => {
        const key = String(r.name).trim().toLowerCase();
        const p = by.get(key) || { name: r.name, n: 0, sum: 0, best: -1e9, you: 0, myopic: 0, planner: 0 };
        p.n++; p.sum += r.ps; p.best = Math.max(p.best, r.ps); p.you += r.you; p.myopic += r.myopic; p.planner += r.planner;
        by.set(key, p);
      });
      const rows = [...by.values()].map((p) => ({ ...p, avg: p.sum / p.n }))
        .sort((a, b) => b.avg - a.avg || b.n - a.n).slice(0, SB_TOP);
      const me = playerName().toLowerCase();
      head = `<tr><th>#</th><th>Player</th>${psHead("Planning score (average)")}<th class="n">Games</th>` +
        `<th class="n">Best game</th><th class="n">vs myopic</th><th class="n">vs planner</th></tr>`;
      body = rows.length ? rows.map((p, i) =>
        `<tr class="${p.name.trim().toLowerCase() === me ? "me" : ""}"><td class="rk">${i + 1}</td><td>${esc(p.name)}</td>` +
        `${psCell(p.avg)}<td class="n">${p.n}</td><td class="n">${Math.round(p.best)}</td>` +
        `${vs(p.you, p.myopic, false)}${vs(p.you, p.planner, true)}</tr>`).join("") : empty(7);
    } else {
      // one entry per market played (the same player on another market, or another day, is a new row)
      const last = all.reduce((m, r) => Math.max(m, r.t || 0), 0);
      const rows = all.slice().sort((a, b) => b.ps - a.ps || b.you - a.you || (a.t || 0) - (b.t || 0)).slice(0, SB_TOP);
      head = `<tr><th>#</th><th>Player</th>${psHead("Planning score")}<th>Market</th><th class="n">Profit</th>` +
        `<th class="n">vs myopic</th><th class="n">vs planner</th></tr>`;
      body = rows.length ? rows.map((p, i) =>
        `<tr class="${p.t && p.t === last ? "me" : ""}"><td class="rk">${i + 1}</td><td>${esc(p.name)}</td>` +
        `${psCell(p.ps)}<td class="mk">${p.seed ? `#${p.seed}${p.day > 1 ? ` · day ${p.day}` : ""}` : "—"}</td>` +
        `<td class="n">${fmtSigned(p.you) || "0"}</td>${vs(p.you, p.myopic, false)}${vs(p.you, p.planner, true)}</tr>`).join("") : empty(7);
    }
    $("sb-head").innerHTML = head;
    $("sb-body").innerHTML = body;
  }

  let setViewRef = null;
  function showWarn(t) { const w = $("intro-warn"); w.textContent = t; w.hidden = false; }

  // enter or leave a competition: UI state, URL hash (#c12345, shareable), scoreboard tab, new game
  function setComp(c) {
    comp = c;
    $("comp").classList.toggle("on", !!c);
    $("comp-off").hidden = !!c; $("comp-on").hidden = !c;
    $("sb-v-comp").hidden = !c;
    $("c-restart").disabled = !!c;
    $("b-replay").hidden = !!c;
    $("b-again").textContent = c ? "Next player" : "New market";
    $("c-new").textContent = c ? "Leave and play a new market" : "New market";
    if (c) { $("comp-name").textContent = c; $("comp-share").textContent = c; }
    try { history.replaceState(null, "", c ? "#c" + c : location.pathname + location.search); } catch (e) { /* ignore */ }
    if (setViewRef) setViewRef(c ? "comp" : (sbView === "comp" ? "games" : sbView));
    if (c) newGame(c);
  }

  function armReset() {
    const b = $("sb-reset");
    if (b.classList.contains("armed")) {
      saveScores([]); renderScoreboard();
      b.classList.remove("armed"); b.textContent = "Reset the scoreboard"; clearTimeout(b._t);
      return;
    }
    b.classList.add("armed"); b.textContent = "Click again to erase all scores";
    b._t = setTimeout(() => { b.classList.remove("armed"); b.textContent = "Reset the scoreboard"; }, 3500);
  }

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
    $("b-start").addEventListener("click", () => {
      if (g.phase !== "intro") return;
      if (comp) {   // one try per player and competition (on this device)
        const name = playerName(), key = name.toLowerCase();
        if (name === "Anonymous") { showWarn("Enter your name to take part in the competition."); $("player").focus(); return; }
        if (loadScores().some((r) => r.comp === comp && String(r.name).trim().toLowerCase() === key)) {
          showWarn(`${name} already played competition #${comp}: one try per player.`); $("player").focus(); return;
        }
      }
      $("intro").hidden = true; goNext();
    });
    $("b-buy").addEventListener("click", () => decide(true));
    $("b-pass").addEventListener("click", () => decide(false));
    $("b-castle").addEventListener("click", () => { if (g && g.phase === "offer" && g.S === 0) goCastle(); });
    $("b-again").addEventListener("click", () => newGame(newSeed()));   // in a competition: next player, same market
    $("b-replay").addEventListener("click", () => newGame(g.mk.seed, g.visit + 1));
    $("c-restart").addEventListener("click", () => newGame(g.mk.seed, g.visit + 1));   // usable mid-game
    $("c-new").addEventListener("click", () => { if (comp) setComp(null); newGame(newSeed()); });
    $("comp-today").addEventListener("click", () => setComp(todayCode()));
    $("comp-join").addEventListener("click", () => {
      const c = parseInt($("comp-code").value, 10);
      if (c >= 1 && c <= 99999) setComp(c); else $("comp-code").focus();
    });
    $("comp-code").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); $("comp-join").click(); } });
    $("comp-leave").addEventListener("click", () => { setComp(null); newGame(newSeed()); });
    $("comp-board").addEventListener("click", () => $("scoreboard").scrollIntoView({ behavior: "smooth" }));
    $("comp-copy").addEventListener("click", () => {
      const url = location.href.split("#")[0] + "#c" + comp, b = $("comp-copy");
      const done = () => { b.textContent = "Copied ✓"; setTimeout(() => { b.textContent = "Copy link"; }, 1600); };
      if (navigator.clipboard) navigator.clipboard.writeText(url).then(done, () => { b.textContent = url; });
      else b.textContent = url;
    });
    try { $("player").value = localStorage.getItem("km-player") || ""; } catch (e) { /* storage unavailable */ }
    $("player").addEventListener("input", () => {
      try { localStorage.setItem("km-player", $("player").value.trim()); } catch (e) { /* storage unavailable */ }
      renderScoreboard();
    });
    $("player").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); $("b-start").click(); } });
    $("sb-reset").addEventListener("click", armReset);
    $("sb-head").addEventListener("click", (e) => { if (e.target.closest("[data-ps]")) { psOpen = !psOpen; renderScoreboard(); } });
    $("ps-close").addEventListener("click", () => { psOpen = false; renderScoreboard(); });
    try { if (localStorage.getItem("km-sb-view") === "players") sbView = "players"; } catch (e) { /* storage unavailable */ }
    const setView = (v) => {
      sbView = v;
      $("sb-v-games").setAttribute("aria-pressed", v === "games"); $("sb-v-players").setAttribute("aria-pressed", v === "players");
      $("sb-v-comp").setAttribute("aria-pressed", v === "comp");
      try { localStorage.setItem("km-sb-view", v); } catch (e) { /* storage unavailable */ }
      renderScoreboard();
    };
    $("sb-v-games").addEventListener("click", () => setView("games"));
    $("sb-v-players").addEventListener("click", () => setView("players"));
    $("sb-v-comp").addEventListener("click", () => setView("comp"));
    setViewRef = setView;
    if (sbView === "comp") sbView = "games";
    setView(sbView);
    renderScoreboard();
    document.addEventListener("keydown", (e) => {
      if (!g || e.target.closest("input, textarea, select")) return;
      if (g.phase === "intro" && e.key === "Enter" && !e.target.closest("button, a")) { e.preventDefault(); $("b-start").click(); }
      if (g.phase === "offer") {
        if (e.key === "b" || e.key === "B") { e.preventDefault(); decide(true); }
        else if (e.key === "ArrowRight" || e.key === " " || e.key === "n" || e.key === "N") { e.preventDefault(); decide(false); }
      }
    });
    const names = ["merchant", ...ITEMS.map((x) => "stand-" + x.key)];
    const hc = /^#c(\d{1,5})$/.exec(location.hash);
    if (hc && +hc[1] >= 1) setComp(+hc[1]); else newGame(newSeed());
    Promise.all(names.map(load)).then(() => document.fonts && document.fonts.load("700 20px 'Pixelify Sans'"));
    requestAnimationFrame(frame);
  }

  window.KnightsMerchant = { makeMarket, simulate, plannerMax, offerAt, N };
  document.addEventListener("DOMContentLoaded", init);
})();
