#!/usr/bin/env python3
"""
fluid.py — a Navier–Stokes smoke tank for your terminal.

Jos Stam's "Stable Fluids" (semi-Lagrangian advection + pressure projection)
on a numpy grid, with vorticity confinement for curly turbulence, buoyant
smoke and multi-coloured dye, rendered with curses in 256 colours.

    python3 fluid.py          # starts in demo mode – touch any key to take over

Keys
  arrows / wasd   move the nozzle (and aim it)      , .   rotate the aim
  space           puff of dye + momentum            f     toggle a jet from the nozzle
  e               plant a permanent emitter         E     remove all emitters
  x               explosion                         v / V vortex spin (cw / ccw)
  1-7, 0          dye colour, 0 = rainbow           b     buoyancy on/off
  g               vorticity confinement strength    [ ]   smoke fade rate
  - =             viscosity                         r     render mode
  m               glyph set                         t     demo autopilot
  c               clear       p  pause              h     help        q  quit
  mouse drag      stir and paint
"""
import curses
import locale
import math
import random
import time

import numpy as np

ASPECT = 2.0  # a terminal cell is roughly twice as tall as it is wide

DYES = [
    ("ember",   (1.00, 0.35, 0.05)),
    ("cyan",    (0.05, 0.65, 1.00)),
    ("magenta", (1.00, 0.10, 0.75)),
    ("lime",    (0.45, 1.00, 0.10)),
    ("gold",    (1.00, 0.80, 0.10)),
    ("violet",  (0.50, 0.20, 1.00)),
    ("ghost",   (0.85, 0.85, 0.92)),
]

GLYPHS = [
    ("ascii",   " .,:;=+*oxO#%@"),
    ("blocks",  " ░░▒▒▓▓██"),
    ("braille", " ⠁⠂⠃⠇⠏⠟⠿⡿⣿"),
    ("dots",    " ·∙•●●"),
]
ARROWS = "→↘↓↙←↖↑↗"
RENDER_MODES = ["smoke", "velocity", "vorticity", "pressure"]
VORT_LEVELS = [0.0, 1.0, 2.5, 5.0, 9.0]


def hsv(h, s=1.0, v=1.0):
    h = (h % 1.0) * 6
    i = int(h)
    f = h - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]


def hsv_np(h, v):
    """Vectorised hue (0..1) + value -> r,g,b arrays."""
    h6 = (h % 1.0) * 6
    i = h6.astype(int) % 6
    f = h6 - np.floor(h6)
    q, t = v * (1 - f), v * f
    z = np.zeros_like(v)
    r = np.choose(i, [v, q, z, z, t, v])
    g = np.choose(i, [t, v, v, q, z, z])
    b = np.choose(i, [z, z, t, v, v, q])
    return r, g, b


def bnd(b, x):
    """Solid walls: mirror normal velocity, copy everything else."""
    x[0, :] = -x[1, :] if b == 2 else x[1, :]
    x[-1, :] = -x[-2, :] if b == 2 else x[-2, :]
    x[:, 0] = -x[:, 1] if b == 1 else x[:, 1]
    x[:, -1] = -x[:, -2] if b == 1 else x[:, -2]


class Fluid:
    def __init__(self, h, w):
        self.h, self.w = h, w
        self.u = np.zeros((h, w))       # x velocity (cells / unit time)
        self.v = np.zeros((h, w))       # y velocity, + is down, same units as u
        self.p = np.zeros((h, w))
        self.curl = np.zeros((h, w))
        self.dye = np.zeros((3, h, w))
        self.Y, self.X = np.mgrid[0:h, 0:w].astype(float)

    def splat(self, x, y, fx, fy, color=None, radius=2.5, amount=1.0):
        r = int(radius * 3) + 2
        x0, x1 = max(0, int(x) - r), min(self.w, int(x) + r + 1)
        y0, y1 = max(0, int(y - r / ASPECT) - 1), min(self.h, int(y + r / ASPECT) + 2)
        if x0 >= x1 or y0 >= y1:
            return
        X, Y = self.X[y0:y1, x0:x1], self.Y[y0:y1, x0:x1]
        g = np.exp(-((X - x) ** 2 + ((Y - y) * ASPECT) ** 2) / (radius * radius))
        self.u[y0:y1, x0:x1] += fx * g
        self.v[y0:y1, x0:x1] += fy * g
        if color is not None:
            self.dye[:, y0:y1, x0:x1] += amount * g * np.asarray(color)[:, None, None]

    def swirl(self, x, y, strength, radius=8.0):
        dx, dy = self.X - x, (self.Y - y) * ASPECT
        g = np.exp(-(dx * dx + dy * dy) / (radius * radius)) * strength
        self.u += -dy * g / radius
        self.v += dx * g / radius

    def blast(self, x, y, strength, radius=6.0, color=None):
        dx, dy = self.X - x, (self.Y - y) * ASPECT
        d = np.sqrt(dx * dx + dy * dy) + 1e-3
        g = np.exp(-(d / radius) ** 2) * strength
        self.u += dx / d * g
        self.v += dy / d * g
        if color is not None:
            self.dye += np.exp(-(d / (radius * 0.6)) ** 2) * 1.6 * np.asarray(color)[:, None, None]

    def advect(self, f, u, v, dt):
        x = np.clip(self.X - dt * u, 0.5, self.w - 1.501)
        y = np.clip(self.Y - dt * v / ASPECT, 0.5, self.h - 1.501)
        i0, j0 = x.astype(np.int32), y.astype(np.int32)
        s1, t1 = x - i0, y - j0
        s0, t0 = 1 - s1, 1 - t1
        i1, j1 = i0 + 1, j0 + 1
        return (s0 * (t0 * f[..., j0, i0] + t1 * f[..., j1, i0]) +
                s1 * (t0 * f[..., j0, i1] + t1 * f[..., j1, i1]))

    def diffuse(self, x, a, iters=8):
        x0 = x.copy()
        for _ in range(iters):
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[1:-1, 2:] + x[1:-1, :-2] +
                                                  x[2:, 1:-1] + x[:-2, 1:-1])) / (1 + 4 * a)

    def project(self, iters=24):
        u, v, p = self.u, self.v, self.p
        ia2 = 1.0 / (ASPECT * ASPECT)
        div = np.zeros_like(u)
        div[1:-1, 1:-1] = (0.5 * (u[1:-1, 2:] - u[1:-1, :-2]) +
                           0.5 * (v[2:, 1:-1] - v[:-2, 1:-1]) / ASPECT)
        denom = 2 + 2 * ia2
        for _ in range(iters):  # warm-started Jacobi
            p[1:-1, 1:-1] = ((p[1:-1, 2:] + p[1:-1, :-2]) +
                             (p[2:, 1:-1] + p[:-2, 1:-1]) * ia2 - div[1:-1, 1:-1]) / denom
            bnd(0, p)
        u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2])
        v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / ASPECT
        bnd(1, u)
        bnd(2, v)

    def confine(self, eps, dt):
        u, v = self.u, self.v
        w = np.zeros_like(u)
        w[1:-1, 1:-1] = (0.5 * (v[1:-1, 2:] - v[1:-1, :-2]) -
                         0.5 * (u[2:, 1:-1] - u[:-2, 1:-1]) / ASPECT)
        self.curl = w
        if not eps:
            return
        aw = np.abs(w)
        nx, ny = np.zeros_like(u), np.zeros_like(u)
        nx[1:-1, 1:-1] = 0.5 * (aw[1:-1, 2:] - aw[1:-1, :-2])
        ny[1:-1, 1:-1] = 0.5 * (aw[2:, 1:-1] - aw[:-2, 1:-1]) / ASPECT
        mag = np.sqrt(nx * nx + ny * ny) + 1e-5
        u += dt * eps * (ny / mag) * w
        v -= dt * eps * (nx / mag) * w

    def step(self, dt, visc, eps, buoy, fade):
        if buoy:
            self.v -= dt * buoy * self.dye.sum(0)
        self.confine(eps, dt)
        if visc:
            self.diffuse(self.u, dt * visc)
            self.diffuse(self.v, dt * visc)
        self.project()
        u0, v0 = self.u, self.v
        self.u = self.advect(u0, u0, v0, dt)
        self.v = self.advect(v0, u0, v0, dt)
        bnd(1, self.u)
        bnd(2, self.v)
        self.project()
        self.u *= 0.9995
        self.v *= 0.9995
        self.dye = self.advect(self.dye, self.u, self.v, dt) * fade
        np.minimum(self.dye, 6.0, out=self.dye)


class Emitter:
    def __init__(self, x, y, angle, dye, phase=None):
        self.x, self.y, self.angle, self.dye = x, y, angle, dye
        self.phase = random.random() * 6.28 if phase is None else phase


class App:
    def __init__(self, scr):
        self.scr = scr
        self.vort = 2
        self.buoy = True
        self.fade = 0.992
        self.visc = 0.0
        self.mode = 0
        self.glyph = 0
        self.dye = 0                 # index into DYES, -1 = rainbow
        self.demo = True
        self.paused = False
        self.help = False
        self.jet = False
        self.emitters = []
        self.angle = -math.pi / 2    # pointing up
        self.mouse_last = None
        self.t = 0.0
        self.fps = 0.0
        self.msg, self.msg_t = "", 0
        self.next_event = 2.0
        self.setup_colors()
        self.resize()

    # ---------------------------------------------------------------- setup
    def setup_colors(self):
        curses.start_color()
        try:
            curses.use_default_colors()
            bg = -1
        except curses.error:
            bg = curses.COLOR_BLACK
        self.c256 = curses.COLORS >= 256
        n = min(255, curses.COLOR_PAIRS - 1) if self.c256 else 7
        for c in range(1, n + 1):
            curses.init_pair(c, c, bg)
        self.npairs = n

    def resize(self):
        self.H, self.W = self.scr.getmaxyx()
        self.gh, self.gw = max(8, self.H - 1), max(16, self.W - 1)
        self.fl = Fluid(self.gh, self.gw)
        self.cx, self.cy = self.gw / 2, self.gh * 0.75
        self.emitters = [e for e in self.emitters if e.x < self.gw and e.y < self.gh]
        self.scr.clear()

    def color_of(self):
        if self.dye < 0:
            return hsv(self.t * 0.07)
        return DYES[self.dye][1]

    def say(self, m):
        self.msg, self.msg_t = m, time.time()

    # ---------------------------------------------------------------- input
    def handle(self, k):
        fl = self.fl
        took_over = True
        step = 1.0
        if k in (ord('q'), 27):
            return False
        elif k in (curses.KEY_LEFT, ord('a')):
            self.cx -= step * 2; self.angle = math.pi
        elif k in (curses.KEY_RIGHT, ord('d')):
            self.cx += step * 2; self.angle = 0
        elif k in (curses.KEY_UP, ord('w')):
            self.cy -= step; self.angle = -math.pi / 2
        elif k in (curses.KEY_DOWN, ord('s')):
            self.cy += step; self.angle = math.pi / 2
        elif k == ord(','):
            self.angle -= math.pi / 12
        elif k == ord('.'):
            self.angle += math.pi / 12
        elif k == ord(' '):
            c = self.color_of()
            f = 40.0
            fl.splat(self.cx, self.cy, f * math.cos(self.angle), f * math.sin(self.angle),
                     c, radius=3.0, amount=1.4)
        elif k == ord('f'):
            self.jet = not self.jet
            self.say("jet " + ("on" if self.jet else "off"))
        elif k == ord('e'):
            self.emitters.append(Emitter(self.cx, self.cy, self.angle, self.dye))
            self.say(f"emitter planted ({len(self.emitters)})")
        elif k == ord('E'):
            self.emitters.clear()
            self.say("emitters removed")
        elif k == ord('x'):
            fl.blast(self.cx, self.cy, 60.0, color=self.color_of())
        elif k == ord('v'):
            fl.swirl(self.cx, self.cy, 45.0)
        elif k == ord('V'):
            fl.swirl(self.cx, self.cy, -45.0)
        elif ord('1') <= k <= ord('7'):
            self.dye = k - ord('1')
            self.say("dye: " + DYES[self.dye][0])
        elif k == ord('0'):
            self.dye = -1
            self.say("dye: rainbow")
        else:
            took_over = False
            if k == ord('b'):
                self.buoy = not self.buoy
                self.say("buoyancy " + ("on" if self.buoy else "off"))
            elif k == ord('g'):
                self.vort = (self.vort + 1) % len(VORT_LEVELS)
                self.say(f"vorticity confinement {VORT_LEVELS[self.vort]}")
            elif k == ord('['):
                self.fade = max(0.95, self.fade - 0.002)
                self.say(f"fade {self.fade:.3f}")
            elif k == ord(']'):
                self.fade = min(1.0, self.fade + 0.002)
                self.say(f"fade {self.fade:.3f}")
            elif k == ord('-'):
                self.visc = max(0.0, self.visc - 0.5)
                self.say(f"viscosity {self.visc:.1f}")
            elif k == ord('='):
                self.visc = min(10.0, self.visc + 0.5)
                self.say(f"viscosity {self.visc:.1f}")
            elif k == ord('r'):
                self.mode = (self.mode + 1) % len(RENDER_MODES)
                self.say("render: " + RENDER_MODES[self.mode])
            elif k == ord('m'):
                self.glyph = (self.glyph + 1) % len(GLYPHS)
                self.say("glyphs: " + GLYPHS[self.glyph][0])
            elif k == ord('t'):
                self.demo = not self.demo
                self.say("demo " + ("on" if self.demo else "off"))
            elif k == ord('c'):
                self.fl = Fluid(self.gh, self.gw)
                self.say("cleared")
            elif k == ord('p'):
                self.paused = not self.paused
            elif k in (ord('h'), ord('?')):
                self.help = not self.help
            elif k == curses.KEY_RESIZE:
                self.resize()
            elif k == curses.KEY_MOUSE:
                self.mouse()
                took_over = True
        if took_over and self.demo and k != curses.KEY_RESIZE:
            self.demo = False
            self.say("you have the controls (t = demo)")
        self.cx = min(max(self.cx, 1), self.gw - 2)
        self.cy = min(max(self.cy, 1), self.gh - 2)
        return True

    def mouse(self):
        try:
            _, mx, my, _, bstate = curses.getmouse()
        except curses.error:
            return
        my -= 1
        now = time.time()
        c = self.color_of()
        if self.mouse_last and now - self.mouse_last[2] < 0.25:
            lx, ly, _ = self.mouse_last
            dx, dy = mx - lx, (my - ly) * ASPECT
            self.fl.splat(mx, my, dx * 14, dy * 14, c, radius=2.5, amount=0.8)
        else:
            self.fl.splat(mx, my, 0, 0, c, radius=2.5, amount=1.0)
        self.cx, self.cy = mx, my
        if bstate & (curses.BUTTON1_RELEASED | getattr(curses, "BUTTON1_CLICKED", 0)):
            self.mouse_last = None
        else:
            self.mouse_last = (mx, my, now)

    # ---------------------------------------------------------------- sim
    def autopilot(self):
        fl, t = self.fl, self.t
        w, h = self.gw, self.gh
        # two wandering jets from the floor, aimed inward, slowly changing colour
        for side, hue0 in ((0, 0.02), (1, 0.55)):
            x = w * (0.2 + 0.6 * side) + math.sin(t * 0.31 + side * 2) * w * 0.08
            a = -math.pi / 2 + (0.55 if side == 0 else -0.55) + 0.45 * math.sin(t * 0.9 + side * 3.1)
            c = hsv(hue0 + 0.08 * math.sin(t * 0.13 + side))
            fl.splat(x, h - 3, 26 * math.cos(a), 26 * math.sin(a), c, radius=2.2, amount=0.32)
        # a side jet now and then
        if math.sin(t * 0.21) > 0.6:
            fl.splat(2, h * 0.4 + math.sin(t) * 3, 30, -2, hsv(0.83), radius=2.0, amount=0.3)
        if t > self.next_event:
            self.next_event = t + random.uniform(2.5, 5.0)
            x, y = random.uniform(w * 0.2, w * 0.8), random.uniform(h * 0.2, h * 0.7)
            if random.random() < 0.5:
                fl.blast(x, y, 40.0, color=hsv(random.random()))
            else:
                fl.swirl(x, y, random.choice((-1, 1)) * 40.0, radius=random.uniform(5, 10))

    def tick(self, dt):
        self.t += dt
        fl = self.fl
        if self.demo:
            self.autopilot()
        for e in self.emitters:
            a = e.angle + 0.3 * math.sin(self.t * 1.3 + e.phase)
            c = hsv(self.t * 0.07 + e.phase) if e.dye < 0 else DYES[e.dye][1]
            fl.splat(e.x, e.y, 24 * math.cos(a), 24 * math.sin(a), c, radius=2.0, amount=0.35)
        if self.jet:
            a = self.angle + 0.15 * math.sin(self.t * 4)
            fl.splat(self.cx, self.cy, 28 * math.cos(a), 28 * math.sin(a), self.color_of(),
                     radius=2.2, amount=0.4)
        fl.step(0.12, self.visc, VORT_LEVELS[self.vort], 1.6 if self.buoy else 0.0, self.fade)

    # ---------------------------------------------------------------- render
    def to_pairs(self, r, g, b):
        """rgb arrays in 0..1 -> curses pair numbers."""
        if self.c256:
            ri = np.clip(np.rint(r * 5), 0, 5).astype(int)
            gi = np.clip(np.rint(g * 5), 0, 5).astype(int)
            bi = np.clip(np.rint(b * 5), 0, 5).astype(int)
            k = 16 + 36 * ri + 6 * gi + bi
            return np.minimum(k, self.npairs)
        k = (r > 0.4).astype(int) * 1 + (g > 0.4) * 2 + (b > 0.4) * 4
        return np.where(k == 0, 7, k)

    def frame(self):
        fl = self.fl
        ramp = GLYPHS[self.glyph][1]
        n = len(ramp) - 1
        mode = RENDER_MODES[self.mode]
        if mode == "smoke":
            d = fl.dye
            m = d.max(0)
            bright = 1 - np.exp(-1.5 * m)
            norm = d / (m + 1e-6)
            lift = 0.35 + 0.65 * bright
            pairs = self.to_pairs(norm[0] * lift, norm[1] * lift, norm[2] * lift)
            gi = np.clip((bright * n + 0.6).astype(int), 0, n)
            gi[bright < 0.03] = 0
            chars = np.array(list(ramp))[gi]
        elif mode == "velocity":
            sp = np.sqrt(fl.u ** 2 + fl.v ** 2)
            ang = np.arctan2(fl.v, fl.u)
            val = 1 - np.exp(-sp / 12)
            r, g, b = hsv_np(ang / (2 * math.pi), 0.3 + 0.7 * val)
            pairs = self.to_pairs(r, g, b)
            ai = (np.rint(ang / (math.pi / 4)).astype(int)) % 8
            chars = np.array(list(ARROWS))[ai]
            gi = (val * n).astype(int)
            chars[val < 0.06] = " "
        else:
            f = fl.curl if mode == "vorticity" else fl.p
            scale = 3.0 if mode == "vorticity" else 6.0
            val = np.tanh(f / scale)
            a = np.abs(val)
            r = np.where(val > 0, 1.0, 0.15) * (0.3 + 0.7 * a)
            g = 0.25 * (0.3 + 0.7 * a) * np.ones_like(a)
            b = np.where(val < 0, 1.0, 0.15) * (0.3 + 0.7 * a)
            pairs = self.to_pairs(r, g, b)
            gi = np.clip((a * n + 0.5).astype(int), 0, n)
            gi[a < 0.04] = 0
            chars = np.array(list(ramp))[gi]
        pairs = np.where(chars == " ", 0, pairs)

        scr = self.scr
        for y in range(self.gh):
            rc = chars[y].tolist()
            rk = pairs[y].tolist()
            x = 0
            while x < self.gw:
                k = rk[x]
                e = x + 1
                while e < self.gw and rk[e] == k:
                    e += 1
                try:
                    scr.addstr(y + 1, x, "".join(rc[x:e]), curses.color_pair(k))
                except curses.error:
                    pass
                x = e

        # emitters and nozzle
        for em in self.emitters:
            self.put(em.y, em.x, "◆", curses.A_DIM)
        ax = self.cx + 2.5 * math.cos(self.angle)
        ay = self.cy + 2.5 * math.sin(self.angle) / ASPECT
        ai = int(round(self.angle / (math.pi / 4))) % 8
        self.put(ay, ax, ARROWS[ai], curses.A_BOLD)
        self.put(self.cy, self.cx, "◎" if not self.jet else "✺", curses.A_BOLD | curses.A_REVERSE)
        self.status()
        if self.help:
            self.draw_help()

    def put(self, y, x, s, attr=0):
        y, x = int(round(y)), int(round(x))
        if 0 <= y < self.gh and 0 <= x < self.gw:
            try:
                self.scr.addstr(y + 1, x, s, attr)
            except curses.error:
                pass

    def status(self):
        dye = "rainbow" if self.dye < 0 else DYES[self.dye][0]
        parts = [
            " ≋ FLUID ",
            f" {RENDER_MODES[self.mode]} ",
            f" dye {dye} ",
            f" vort {VORT_LEVELS[self.vort]:g} ",
            f" buoy {'on' if self.buoy else 'off'} ",
            f" fade {self.fade:.3f} ",
            f" visc {self.visc:g} ",
            f" {self.fps:4.1f} fps ",
        ]
        if self.demo:
            parts.append(" DEMO ")
        if self.paused:
            parts.append(" PAUSED ")
        line = "│".join(parts)
        if self.msg and time.time() - self.msg_t < 2.5:
            line += "  » " + self.msg
        else:
            line += "  h: help"
        line = line[: self.W - 1].ljust(self.W - 1)
        try:
            self.scr.addstr(0, 0, line, curses.A_REVERSE)
        except curses.error:
            pass

    def draw_help(self):
        lines = [l for l in __doc__.strip("\n").splitlines()[7:] if l.strip()]
        w = max(len(l) for l in lines) + 4
        h = len(lines) + 2
        y0, x0 = max(1, (self.H - h) // 2), max(0, (self.W - w) // 2)
        for i in range(h):
            if i == 0 or i == h - 1:
                s = ("╭" if i == 0 else "╰") + "─" * (w - 2) + ("╮" if i == 0 else "╯")
            else:
                s = "│ " + lines[i - 1].ljust(w - 4) + " │"
            try:
                self.scr.addstr(y0 + i, x0, s[: self.W - x0 - 1], curses.A_BOLD)
            except curses.error:
                pass

    # ---------------------------------------------------------------- loop
    def run(self):
        last = time.time()
        acc_t, acc_n = 0.0, 0
        while True:
            k = self.scr.getch()
            while k != -1:
                if not self.handle(k):
                    return
                k = self.scr.getch()
            t0 = time.time()
            if not self.paused:
                self.tick(1 / 30)
            self.frame()
            self.scr.refresh()
            now = time.time()
            dt = now - last
            last = now
            acc_t += dt
            acc_n += 1
            if acc_t > 0.5:
                self.fps, acc_t, acc_n = acc_n / acc_t, 0.0, 0
            spare = 1 / 45 - (time.time() - t0)
            if spare > 0:
                time.sleep(spare)


def main(scr):
    curses.curs_set(0)
    scr.nodelay(True)
    scr.keypad(True)
    curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
    curses.mouseinterval(0)
    print("\033[?1002h", end="", flush=True)  # report motion while a button is held
    try:
        App(scr).run()
    finally:
        print("\033[?1002l", end="", flush=True)


if __name__ == "__main__":
    locale.setlocale(locale.LC_ALL, "")
    curses.wrapper(main)
