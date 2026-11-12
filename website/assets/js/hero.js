/* DynamAuction website: main animation of the "problem" section.
 * Traces the value of the next display over one campaign. Each won auction multiplies it by a recovering fatigue
 * factor (1 - c e^{-(t - t_i)/s}), so it drops at the win and recovers between wins; lost auctions leave it unchanged.
 * The blue line is drawn in time (steady pace, short pause at each drop), the auctions light up when reached, and the
 * animation loops while it is on screen. With prefers-reduced-motion, the whole trajectory is shown at once. */
(function () {
  "use strict";

  const X0 = 20, X1 = 340, Y_TOP = 40, Y_AXIS = 160;
  const C = 0.5, SCALE = 40;                      // fatigue strength and recovery scale (in x units)
  const AUCTIONS = [[45, false], [70, true], [95, false], [120, true], [160, true],
                    [190, false], [225, false], [265, false], [305, false]];   // [x, won]
  const MS_PER_X = 17, DROP_MS = 320, PAUSE_MS = 160, HOLD_MS = 1900, FADE_MS = 500, STEP = 2;

  const wins = AUCTIONS.filter((a) => a[1]).map((a) => a[0]);
  const value = (x, afterWin) => wins.filter((w) => w < x || (afterWin && w === x))
    .reduce((p, w) => p * (1 - C * Math.exp(-(x - w) / SCALE)), 1);
  const yOf = (v) => Y_AXIS - v * (Y_AXIS - Y_TOP);

  // timeline: points (t, x, y), linear in between
  const keys = [];
  const reach = {};                               // x of an auction -> time it is reached
  let t = 0;
  const xs = new Set();
  for (let x = X0; x <= X1; x += STEP) xs.add(x);
  AUCTIONS.forEach((a) => xs.add(a[0]));
  [...xs].sort((a, b) => a - b).forEach((x, i, arr) => {
    if (i > 0) t += (x - arr[i - 1]) * MS_PER_X;
    keys.push({ t, x, y: yOf(value(x, false)) });
    if (AUCTIONS.some((a) => a[0] === x)) reach[x] = t;
    if (wins.includes(x)) {
      t += PAUSE_MS; keys.push({ t, x, y: yOf(value(x, false)) });
      t += DROP_MS;  keys.push({ t, x, y: yOf(value(x, true)) });
    }
  });
  const T_END = t;

  function init() {
    const svg = document.getElementById("hero-anim");
    if (!svg) return;
    const curve = svg.querySelector("#ha-curve"), area = svg.querySelector("#ha-area"), head = svg.querySelector("#ha-head");
    const group = svg.querySelector("#ha-auctions");
    const NS = "http://www.w3.org/2000/svg";
    const dots = AUCTIONS.map(([x, won]) => {
      const c = document.createElementNS(NS, "circle");
      c.setAttribute("cx", x); c.setAttribute("cy", Y_AXIS); c.setAttribute("r", 6);
      c.setAttribute("class", "au future lost");   // every auction looks pending until the line reaches it
      group.appendChild(c);
      return { el: c, x, won, done: false };
    });

    function render(tau) {
      let i = 0;
      while (i + 1 < keys.length && keys[i + 1].t <= tau) i++;
      const pts = keys.slice(0, i + 1).map((k) => [k.x, k.y]);
      if (i + 1 < keys.length && tau > keys[i].t) {
        const a = keys[i], b = keys[i + 1], s = (tau - a.t) / (b.t - a.t);
        pts.push([a.x + (b.x - a.x) * s, a.y + (b.y - a.y) * s]);
      }
      const line = pts.map((p, j) => (j ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
      const last = pts[pts.length - 1];
      curve.setAttribute("d", line);
      area.setAttribute("d", `${line} L${last[0].toFixed(1)} ${Y_AXIS} L${X0} ${Y_AXIS} Z`);
      head.setAttribute("cx", last[0].toFixed(1)); head.setAttribute("cy", last[1].toFixed(1));
      dots.forEach((d) => {
        if (!d.done && tau >= reach[d.x]) {
          d.done = true;
          d.el.setAttribute("class", "au " + (d.won ? "won pop" : "lost"));
          if (d.won) d.el.setAttribute("r", 7);
        }
      });
    }

    function reset() {
      dots.forEach((d) => { d.done = false; d.el.setAttribute("class", "au future lost"); d.el.setAttribute("r", 6); });
      [curve, area, head].forEach((el) => { el.style.opacity = ""; });
    }

    if (matchMedia("(prefers-reduced-motion: reduce)").matches) { render(T_END); return; }

    let start = null, raf = 0, visible = false, paused = 0;
    function frame(now) {
      if (start === null) start = now - paused;
      const tau = now - start;
      if (tau <= T_END) render(tau);
      else if (tau <= T_END + HOLD_MS) render(T_END);
      else if (tau <= T_END + HOLD_MS + FADE_MS) {
        const o = 1 - (tau - T_END - HOLD_MS) / FADE_MS;
        [curve, area, head].forEach((el) => { el.style.opacity = o.toFixed(2); });
        dots.forEach((d) => { d.el.style.opacity = o.toFixed(2); });
      } else {
        reset(); dots.forEach((d) => { d.el.style.opacity = ""; });
        start = now; render(0);
      }
      raf = visible ? requestAnimationFrame(frame) : 0;
    }

    render(0);
    const io = new IntersectionObserver((entries) => {
      visible = entries[0].isIntersecting;
      if (visible && !raf) { start = null; raf = requestAnimationFrame(frame); }
      if (!visible && start !== null) { paused = performance.now() - start; }
    }, { threshold: 0.15 });
    io.observe(svg);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
