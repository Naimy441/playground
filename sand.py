#!/usr/bin/env python3
"""
sand.py — a terminal sand & calligraphy physics engine.

Grains of coloured sand drift in on the wind, settle into crisp letterforms
or fractals, glint for a moment, and then crumble, blow away or explode into a
pile that drains out through the floor like an hourglass.

Each terminal cell holds two "pixels" (▀▄ half blocks), so the grain grid is
twice as tall as your terminal.  Letterforms come from your system's fonts
(Zapfino, Snell Roundhand, Savoye…) via Pillow, with a built-in pixel font
as a fallback.

    python3 sand.py                       # the default playlist
    python3 sand.py "hello" "world"       # your own words (fractals between)
    python3 sand.py --only-text "Ink"     # no fractals
    python3 sand.py --font zapfino        # pin a font

Keys
  space       dissolve now / next shape      t   type new words
  a           assembly style                 d   dissolve style
  f           font                           c   colour palette
  + -         speed                          p   pause
  h           help                           q   quit
  mouse       poke the sculpture and knock grains loose
"""
import argparse
import curses
import locale
import math
import os
import random
import time

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # letterforms fall back to the built-in pixel font
    Image = None

# --------------------------------------------------------------------------- palettes
PALETTES = {  # 8 xterm-256 colours each, dark -> light
    "desert":  [94, 130, 136, 172, 178, 179, 186, 223],
    "ocean":   [17, 18, 25, 31, 38, 45, 87, 159],
    "ember":   [52, 88, 124, 160, 196, 202, 214, 220],
    "jade":    [22, 28, 29, 35, 42, 49, 85, 158],
    "rose":    [53, 89, 125, 161, 168, 175, 182, 225],
    "ash":     [236, 239, 242, 245, 248, 251, 253, 231],
    "rainbow": [196, 202, 226, 46, 51, 33, 93, 201],
    "dusk":    [54, 55, 91, 127, 163, 205, 210, 223],
}
K = 8

# --------------------------------------------------------------------------- fonts
FONT_DIRS = ["/System/Library/Fonts/Supplemental", "/System/Library/Fonts", "/Library/Fonts",
             os.path.expanduser("~/Library/Fonts"), "/usr/share/fonts", "/usr/local/share/fonts"]
FONTS = [  # (name, candidate file names)
    ("zapfino",   ["Zapfino.ttf"]),
    ("snell",     ["SnellRoundhand.ttc"]),
    ("savoye",    ["Savoye LET.ttc"]),
    ("chancery",  ["Apple Chancery.ttf"]),
    ("brush",     ["Brush Script.ttf"]),
    ("didot",     ["Didot.ttc"]),
    ("bodoni",    ["Bodoni 72.ttc"]),
    ("futura",    ["Futura.ttc"]),
    ("luminari",  ["Luminari.ttf"]),
    ("trattatello", ["Trattatello.ttf"]),
    ("dejavu",    ["DejaVuSerif-Bold.ttf", "DejaVuSans-Bold.ttf"]),
    ("pixel",     []),
]


def find_font(names):
    for d in FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for n in names:
                if n in files:
                    return os.path.join(root, n)
    return None


def available_fonts():
    out = []
    for name, files in FONTS:
        if name == "pixel":
            out.append((name, None))
        elif Image is not None:
            p = find_font(files)
            if p:
                out.append((name, p))
    return out


# 3x5 pixel font, rows top->bottom
PIXEL_FONT = {
    "A": "010101111101101", "B": "110101110101110", "C": "011100100100011",
    "D": "110101101101110", "E": "111100110100111", "F": "111100110100100",
    "G": "011100101101011", "H": "101101111101101", "I": "111010010010111",
    "J": "001001001101010", "K": "101101110101101", "L": "100100100100111",
    "M": "101111111101101", "N": "110101101101101", "O": "010101101101010",
    "P": "110101110100100", "Q": "010101101110011", "R": "110101110101101",
    "S": "011100010001110", "T": "111010010010010", "U": "101101101101111",
    "V": "101101101101010", "W": "101101111111101", "X": "101101010101101",
    "Y": "101101010010010", "Z": "111001010100111", "0": "111101101101111",
    "1": "010110010010111", "2": "110001010100111", "3": "110001010001110",
    "4": "101101111001001", "5": "111100110001110", "6": "011100111101111",
    "7": "111001010010010", "8": "111101111101111", "9": "111101111001110",
    " ": "000000000000000", "!": "010010010000010", "?": "110001010000010",
    ".": "000000000000010", "-": "000000111000000", ",": "000000000010100",
    "'": "010010000000000", ":": "000010000010000", "<": "001010100010001",
    ">": "100010001010100", "*": "000101010101000", "+": "000010111010000",
}


def fit_into(img_mask, PH, PW, fill_w=0.92, fill_h=0.8):
    """Scale a boolean mask (nearest) to fit the grid and centre it."""
    ih, iw = img_mask.shape
    s = min(PW * fill_w / iw, PH * fill_h / ih)
    nh, nw = max(1, int(ih * s)), max(1, int(iw * s))
    yi = (np.arange(nh) / s).astype(int).clip(0, ih - 1)
    xi = (np.arange(nw) / s).astype(int).clip(0, iw - 1)
    small = img_mask[yi][:, xi]
    out = np.zeros((PH, PW), bool)
    y0, x0 = (PH - nh) // 2, (PW - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


def pixel_text(text, PH, PW):
    lines = split_lines(text.upper(), PW / PH, 3 / 5)[0]
    w = max(len(l) for l in lines) * 4 - 1
    h = len(lines) * 6 - 1
    m = np.zeros((h, w), bool)
    for li, line in enumerate(lines):
        for ci, ch in enumerate(line):
            bits = PIXEL_FONT.get(ch, PIXEL_FONT["?"])
            for r in range(5):
                for c in range(3):
                    if bits[r * 3 + c] == "1":
                        m[li * 6 + r, ci * 4 + c] = True
    return fit_into(m, PH, PW, 0.9, 0.7)


def split_lines(text, grid_aspect, glyph_aspect):
    """Choose 1-3 lines of roughly equal length that best fill the grid."""
    words = text.split()
    best = ([text], -1)
    for n in range(1, min(3, len(words)) + 1):
        target = len(text) / n
        lines, cur = [], ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > target * 1.25 and len(lines) < n - 1:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        width = max(len(l) for l in lines) * glyph_aspect
        score = min(grid_aspect / width, 1 / len(lines))
        if score > best[1]:
            best = (lines, score)
    return best


def font_text(text, path, PH, PW):
    font = ImageFont.truetype(path, 140)
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    choices = []
    for n in range(1, 4):
        words = text.split()
        if len(words) < n:
            break
        per = math.ceil(len(words) / n)
        lines = [" ".join(words[i:i + per]) for i in range(0, len(words), per)]
        txt = "\n".join(lines)
        l, t, r, b = probe.multiline_textbbox((0, 0), txt, font=font, align="center", spacing=20)
        s = min(PW * 0.92 / max(1, r - l), PH * 0.8 / max(1, b - t))
        choices.append((s, txt, (l, t, r, b)))
    s, txt, (l, t, r, b) = max(choices)
    l, t, r, b = math.floor(l), math.floor(t), math.ceil(r), math.ceil(b)
    pad = 40
    img = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad))
    ImageDraw.Draw(img).multiline_text((pad - l, pad - t), txt, fill=255, font=font,
                                       align="center", spacing=20)
    bbox = img.getbbox()
    if not bbox:
        return pixel_text(text, PH, PW)
    img = img.crop(bbox)
    iw, ih = img.size
    s = min(PW * 0.92 / iw, PH * 0.8 / ih)
    img = img.resize((max(1, int(iw * s)), max(1, int(ih * s))), Image.LANCZOS)
    m = np.array(img) > 90
    out = np.zeros((PH, PW), bool)
    nh, nw = m.shape
    y0, x0 = (PH - nh) // 2, (PW - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = m
    return out


# --------------------------------------------------------------------------- fractals
def chaos_game(maps, PH, PW, n=60000, iters=28, flip=True):
    maps = np.array(maps, float)
    p = maps[:, 6] / maps[:, 6].sum()
    pts = np.random.rand(n, 2)
    for _ in range(iters):
        k = np.random.choice(len(maps), n, p=p)
        a, b, c, d, e, f = (maps[k, i] for i in range(6))
        x, y = pts[:, 0], pts[:, 1]
        pts = np.stack([a * x + b * y + e, c * x + d * y + f], 1)
    x, y = pts[:, 0], pts[:, 1]
    if flip:
        y = -y
    w, h = x.max() - x.min(), y.max() - y.min()
    s = min(PW * 0.9 / max(w, 1e-9), PH * 0.86 / max(h, 1e-9))
    xi = ((x - x.min()) * s + (PW - w * s) / 2).astype(int).clip(0, PW - 1)
    yi = ((y - y.min()) * s + (PH - h * s) / 2).astype(int).clip(0, PH - 1)
    m = np.zeros((PH, PW), bool)
    m[yi, xi] = True
    return m


def fern(PH, PW):
    return chaos_game([(0, 0, 0, .16, 0, 0, .01), (.85, .04, -.04, .85, 0, 1.6, .85),
                       (.2, -.26, .23, .22, 0, 1.6, .07), (-.15, .28, .26, .24, 0, .44, .07)], PH, PW)


def sierpinski(PH, PW):
    v = [(0, 0), (1, 0), (.5, .866)]
    return chaos_game([(.5, 0, 0, .5, x * .5, y * .5, 1) for x, y in v], PH, PW)


def carpet(PH, PW):
    offs = [(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)]
    return chaos_game([(1 / 3, 0, 0, 1 / 3, i / 3, j / 3, 1) for i, j in offs], PH, PW)


def vicsek(PH, PW):
    offs = [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)]
    return chaos_game([(1 / 3, 0, 0, 1 / 3, i / 3, j / 3, 1) for i, j in offs], PH, PW)


def dragon(PH, PW):
    return chaos_game([(.5, -.5, .5, .5, 0, 0, 1), (-.5, -.5, .5, -.5, 1, 0, 1)], PH, PW, n=90000)


def levy(PH, PW):
    return chaos_game([(.5, -.5, .5, .5, 0, 0, 1), (.5, .5, -.5, .5, .5, .5, 1)], PH, PW, n=90000)


def mandelbrot(PH, PW):
    w, h = 3.0, 2.4
    s = min(PW * 0.92 / w, PH * 0.9 / h)
    xs = (np.arange(PW) - PW / 2) / s - 0.65
    ys = (np.arange(PH) - PH / 2) / s
    c = xs[None, :] + 1j * ys[:, None]
    z = np.zeros_like(c)
    it = np.zeros(c.shape, int)
    for i in range(60):
        live = np.abs(z) <= 2
        z[live] = z[live] ** 2 + c[live]
        it[live] += 1
    return it >= 18


def snowflake(PH, PW):
    def koch(a, b, depth):
        if depth == 0:
            return [a]
        (x1, y1), (x2, y2) = a, b
        dx, dy = (x2 - x1) / 3, (y2 - y1) / 3
        p1, p2 = (x1 + dx, y1 + dy), (x1 + 2 * dx, y1 + 2 * dy)
        px = p1[0] + dx * .5 - dy * .866
        py = p1[1] + dy * .5 + dx * .866
        return (koch(a, p1, depth - 1) + koch(p1, (px, py), depth - 1) +
                koch((px, py), p2, depth - 1) + koch(p2, b, depth - 1))
    tri = [(0, 0), (.5, .866), (1, 0)]
    pts = []
    for i in range(3):
        pts += koch(tri[i], tri[(i + 1) % 3], 4)
    pts = np.array(pts)
    pts[:, 1] = -pts[:, 1]
    return polygon_mask(pts, PH, PW)


def polygon_mask(pts, PH, PW):
    x, y = pts[:, 0], pts[:, 1]
    w, h = x.max() - x.min(), y.max() - y.min()
    s = min(PW * 0.9 / w, PH * 0.88 / h)
    px = (x - x.min()) * s + (PW - w * s) / 2
    py = (y - y.min()) * s + (PH - h * s) / 2
    # even-odd scanline fill, vectorised over pixel centres
    Y, X = np.mgrid[0:PH, 0:PW] + 0.5
    inside = np.zeros((PH, PW), bool)
    x0, y0 = px, py
    x1, y1 = np.roll(px, -1), np.roll(py, -1)
    for a, b, c, d in zip(x0, y0, x1, y1):
        if b == d:
            continue
        lo, hi = min(b, d), max(b, d)
        r0, r1 = int(max(0, math.floor(lo))), int(min(PH, math.ceil(hi)))
        if r0 >= r1:
            continue
        yy = Y[r0:r1, :]
        cross = (yy >= lo) & (yy < hi)
        xint = a + (yy - b) * (c - a) / (d - b)
        inside[r0:r1, :] ^= cross & (X[r0:r1, :] < xint)
    return inside


def rose(PH, PW):
    R = min(PH, PW) * 0.46
    Y, X = np.mgrid[0:PH, 0:PW]
    dx, dy = X - PW / 2 + 0.5, Y - PH / 2 + 0.5
    r, th = np.hypot(dx, dy), np.arctan2(dy, dx)
    petals = r < R * np.abs(np.cos(4 * th)) ** 0.6
    inner = r < R * (0.32 + 0.06 * np.cos(12 * th))
    ring = (r > R * 0.2) & (r < R * 0.25)
    return (petals | inner) & ~ring


FRACTALS = {"fern": fern, "sierpinski": sierpinski, "mandelbrot": mandelbrot,
            "dragon": dragon, "snowflake": snowflake, "carpet": carpet,
            "rose": rose, "levy": levy, "vicsek": vicsek}
DEFAULT_WORDS = ["sand", "Calligraphy", "drift", "ephemeral", "hello, world", "dust & ink",
                 "impermanence", "wind"]

# --------------------------------------------------------------------------- simulation
WAIT, FLY, LOCK, FALL, WIND, DEAD = range(6)
ASSEMBLY = ["pour", "write", "swirl"]
DISSOLVE = ["crumble", "wind", "explode"]


class App:
    def __init__(self, scr, args):
        self.scr = scr
        self.fonts = available_fonts()
        self.font_i = None  # None = rotate
        if args.font:
            for i, (n, _) in enumerate(self.fonts):
                if n.startswith(args.font.lower()):
                    self.font_i = i
        words = args.words or DEFAULT_WORDS
        fr = [] if args.only_text else list(FRACTALS)
        self.playlist = []
        for i in range(max(len(words), len(fr))):
            if i < len(words):
                self.playlist.append(("text", words[i]))
            if i < len(fr):
                self.playlist.append(("fractal", fr[i]))
        self.pi = 0
        self.pal_name = None      # None = rotate
        self.asm_i = None
        self.dis_i = None
        self.speed = 1
        self.paused = False
        self.help = False
        self.queue = []           # user-typed words jump the queue
        self.frame_no = 0
        self.msg, self.msg_t = "", 0
        self.rot = 0
        self.setup_colors()
        self.resize()

    # ------------------------------------------------------------------ setup
    def setup_colors(self):
        curses.start_color()
        try:
            curses.use_default_colors()
            self.bg = -1
        except curses.error:
            self.bg = curses.COLOR_BLACK
        self.c256 = curses.COLORS >= 256 and curses.COLOR_PAIRS > 90

    def apply_palette(self, name):
        self.palette = name
        cols = PALETTES[name]
        if not self.c256:
            cols = [3, 3, 3, 3, 7, 7, 7, 7] if name != "ocean" else [4, 4, 4, 6, 6, 6, 7, 7]
        for fg in range(1, K + 1):
            for bg in range(0, K + 1):
                curses.init_pair(1 + fg * (K + 1) + bg, cols[fg - 1], self.bg if bg == 0 else cols[bg - 1])

    def resize(self):
        self.H, self.W = self.scr.getmaxyx()
        self.PH, self.PW = max(10, (self.H - 1) * 2), max(20, self.W - 1)
        self.G = np.zeros((self.PH, self.PW), np.int8)  # the settled sand pile
        self.scr.clear()
        self.n = 0
        self.next_shape()

    def say(self, m):
        self.msg, self.msg_t = m, time.time()

    # ------------------------------------------------------------------ shapes
    def next_shape(self):
        self.rot += 1
        if self.queue:
            kind, what = "text", self.queue.pop(0)
        else:
            kind, what = self.playlist[self.pi % len(self.playlist)]
            self.pi += 1
        fname = ""
        if kind == "text":
            fi = self.font_i if self.font_i is not None else self.rot % len(self.fonts)
            fname, path = self.fonts[fi]
            mask = pixel_text(what, self.PH, self.PW) if path is None else font_text(what, path, self.PH, self.PW)
            self.label = f"“{what}” · {fname}"
        else:
            mask = FRACTALS[what](self.PH, self.PW)
            self.label = what
        pal = self.pal_name or list(PALETTES)[self.rot % len(PALETTES)]
        self.apply_palette(pal)
        self.asm = ASSEMBLY[self.asm_i] if self.asm_i is not None else ASSEMBLY[self.rot % 3]
        self.dis = DISSOLVE[self.dis_i] if self.dis_i is not None else DISSOLVE[(self.rot // 3 + self.rot) % 3]
        self.load(mask, pal == "rainbow")

    def load(self, mask, by_x):
        ty, tx = np.nonzero(mask)
        n = self.n = len(tx)
        PH, PW = self.PH, self.PW
        self.tx, self.ty = tx.astype(np.float32), ty.astype(np.float32)
        if n == 0:
            self.phase, self.phase_t = "drain", 0
            return
        # colour: a gradient across the shape plus grain-to-grain jitter for texture
        if by_x:
            rel = (tx - tx.min()) / max(1, np.ptp(tx))
        else:
            rel = 1 - (ty - ty.min()) / max(1, np.ptp(ty))
        jitter = np.random.randint(-1, 2, n)
        self.col = np.clip((rel * 5.99).astype(int) + 1 + jitter + (0 if by_x else 1), 1, K).astype(np.int8)
        self.st = np.full(n, WAIT, np.int8)
        self.age = np.zeros(n, np.int32)
        noise = np.random.rand(n)
        cx, cy = PW / 2, PH / 2
        if self.asm == "pour":
            order = -(ty + noise * 6)                       # fill the mould from the bottom up
            self.x = tx + np.random.normal(0, 12, n)
            self.y = -np.random.rand(n) * 6 - 1
            self.vx = np.zeros(n)
            self.vy = np.random.rand(n) * 0.6 + 0.4
        elif self.asm == "write":
            order = tx + ty * 0.15 + noise * 4              # a brush moving left to right
            self.x = -np.random.rand(n) * 25 - 2
            self.y = ty + np.random.normal(0, PH * 0.25, n)
            self.vx = np.random.rand(n) * 2 + 1.5
            self.vy = np.random.normal(0, 0.4, n)
        else:  # swirl
            ang = np.arctan2(ty - cy, tx - cx)
            order = (ang + math.pi) + noise * 0.5           # sweep round like a clock hand
            R = max(PH, PW) * 0.7
            a = ang + 1.4
            self.x = cx + R * np.cos(a)
            self.y = cy + R * np.sin(a) * 0.6
            self.vx = -np.sin(a) * 2.5
            self.vy = np.cos(a) * 2.5
        rank = np.empty(n, int)
        rank[np.argsort(order)] = np.arange(n)
        rate = max(4.0, n / (30 * 3.2))                     # assemble in ~3 seconds
        self.rel = (rank / rate).astype(np.int32)
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)
        self.phase, self.phase_t = "assemble", 0
        self.glint = -1e9

    # ------------------------------------------------------------------ physics
    def wind(self, x, y, t):
        vx = 0.7 * np.sin(y * 0.11 + t * 1.3) + 0.35 * np.sin(x * 0.05 - t * 0.7)
        vy = 0.5 * np.sin(x * 0.09 + t * 1.1) + 0.25 * np.cos(y * 0.2 - t)
        return vx, vy

    def step(self):
        self.phase_t += 1
        t = self.frame_no / 30
        n = self.n
        if n:
            st = self.st
            # release waiting grains
            if self.phase == "assemble":
                st[(st == WAIT) & (self.rel <= self.phase_t)] = FLY
            fly = st == FLY
            if fly.any():
                self.step_fly(fly, t)
            fall = st == FALL
            if fall.any():
                self.step_fall(fall)
            wind = st == WIND
            if wind.any():
                self.step_wind(wind, t)
            if self.phase == "dissolve":
                due = (st == LOCK) & (self.rel <= self.phase_t)
                if due.any():
                    self.launch(due)
        self.sand_ca()
        self.advance_phase()

    def step_fly(self, m, t):
        x, y, vx, vy, tx, ty = self.x[m], self.y[m], self.vx[m], self.vy[m], self.tx[m], self.ty[m]
        age = self.age[m] + 1
        if self.asm == "pour":
            vy = np.minimum(vy + 0.09, 2.2)
            frac = np.clip((ty - y) / (ty + 8), 0, 1)
            wx, _ = self.wind(x, y, t)
            x = x + (tx - x) * 0.07 + wx * frac * 1.5
            y = y + vy
            done = (y >= ty) | (age > 240)
        else:
            dx, dy = tx - x, ty - y
            dist = np.hypot(dx, dy)
            wx, wy = self.wind(x, y, t)
            g = np.minimum(1, dist / 25)
            ax = dx * 0.018 - vx * 0.26 + wx * 0.25 * g
            ay = dy * 0.018 - vy * 0.26 + wy * 0.25 * g
            if self.asm == "swirl":
                k = 0.06 * np.minimum(1, dist / 12)
                ax += -dy / (dist + 1e-3) * k * 6
                ay += dx / (dist + 1e-3) * k * 6
            vx, vy = vx + ax, vy + ay
            x, y = x + vx, y + vy
            done = ((dist < 0.8) & (np.hypot(vx, vy) < 1.0)) | (age > 150)
        x = np.where(done, tx, x)
        y = np.where(done, ty, y)
        self.x[m], self.y[m], self.vx[m], self.vy[m], self.age[m] = x, y, vx, vy, age
        idx = np.nonzero(m)[0]
        self.st[idx[done]] = LOCK

    def launch(self, m):
        idx = np.nonzero(m)[0]
        k = len(idx)
        if self.dis == "wind":
            self.st[idx] = WIND
            self.vx[idx] = -np.random.rand(k) * 0.5
            self.vy[idx] = -np.random.rand(k) * 0.3
        else:
            self.st[idx] = FALL
            if self.dis == "explode":
                dx = self.x[idx] - self.PW / 2
                dy = self.y[idx] - self.PH * 0.55
                d = np.hypot(dx, dy) + 1e-3
                sp = np.random.rand(k) * 2.4 + 0.8
                self.vx[idx] = dx / d * sp
                self.vy[idx] = dy / d * sp - 1.2
            else:
                self.vx[idx] = np.random.normal(0, 0.15, k)
                self.vy[idx] = np.random.rand(k) * 0.3

    def step_wind(self, m, t):
        x, y = self.x[m], self.y[m]
        wx, wy = self.wind(x, y, t)
        fx = -1.6 + wx * 1.2
        fy = wy * 1.4 - 0.12
        vx = self.vx[m] + (fx - self.vx[m]) * 0.06 + np.random.normal(0, 0.05, len(x))
        vy = self.vy[m] + (fy - self.vy[m]) * 0.06 + np.random.normal(0, 0.05, len(x))
        x, y = x + vx, y + vy
        self.x[m], self.y[m], self.vx[m], self.vy[m] = x, y, vx, vy
        gone = (x < -2) | (x > self.PW + 2) | (y < -4) | (y > self.PH + 2)
        idx = np.nonzero(m)[0]
        self.st[idx[gone]] = DEAD

    def step_fall(self, m):
        PH, PW, G = self.PH, self.PW, self.G
        vx = self.vx[m] * 0.995
        vy = np.minimum(self.vy[m] + 0.11, 2.5)
        x, y = self.x[m] + vx, self.y[m] + vy
        bounce = (x < 0) | (x > PW - 1)
        vx = np.where(bounce, -vx * 0.5, vx)
        x = np.clip(x, 0, PW - 1)
        self.x[m], self.y[m], self.vx[m], self.vy[m] = x, y, vx, vy
        xi = np.rint(x).astype(int)
        yi = np.floor(y).astype(int)
        onscreen = yi >= 0
        yc = np.clip(yi, 0, PH - 1)
        below = np.clip(yi + 1, 0, PH - 1)
        land = onscreen & ((yi >= PH - 1) | (G[below, xi] != 0) | (G[yc, xi] != 0))
        if not land.any():
            return
        idx = np.nonzero(m)[0][land]
        for i in idx[np.argsort(-self.y[idx])]:
            c = int(round(self.x[i]))
            r = min(int(self.y[i]), PH - 1)
            while r >= 0 and G[r, c]:
                r -= 1
            if r >= 0:
                G[r, c] = self.col[i]
            self.st[i] = DEAD

    def sand_ca(self):
        G = self.G
        passes = 2
        hole = getattr(self, "hole", 0)
        moved = 0
        for p in range(passes):
            if hole > 0:
                c = self.PW // 2
                lo, hi = max(0, int(c - hole)), min(self.PW, int(c + hole) + 1)
                G[-1, lo:hi] = 0
            occ = G != 0
            # straight down
            src = np.zeros_like(occ)
            src[:-1] = occ[:-1] & ~occ[1:]
            if src.any():
                dst = np.zeros_like(src)
                dst[1:] = src[:-1]
                vals = G[src]
                G[src] = 0
                G[dst] = vals
                moved += len(vals)
                occ = G != 0
            # diagonal, alternating sides so two grains never want the same cell
            d = 1 if (self.frame_no + p) % 2 else -1
            src = np.zeros_like(occ)
            rnd = np.random.rand(self.PH - 1, self.PW - 1) < 0.85
            if d == 1:
                src[:-1, :-1] = occ[:-1, :-1] & occ[1:, :-1] & ~occ[1:, 1:] & ~occ[:-1, 1:] & rnd
                dst = np.zeros_like(src)
                dst[1:, 1:] = src[:-1, :-1]
            else:
                src[:-1, 1:] = occ[:-1, 1:] & occ[1:, 1:] & ~occ[1:, :-1] & ~occ[:-1, :-1] & rnd
                dst = np.zeros_like(src)
                dst[1:, :-1] = src[:-1, 1:]
            if src.any():
                vals = G[src]
                G[src] = 0
                G[dst] = vals
                moved += len(vals)
        self.moved = moved

    def advance_phase(self):
        st = self.st if self.n else np.zeros(0)
        if self.phase == "assemble":
            if self.n and not ((st == WAIT) | (st == FLY)).any():
                self.phase, self.phase_t = "hold", 0
        elif self.phase == "hold":
            if self.phase_t == 20:
                self.glint = self.frame_no
            if self.phase_t > 30 * 3.2:
                self.dissolve()
        elif self.phase == "dissolve":
            if not ((st == LOCK) | (st == FALL) | (st == WIND) | (st == FLY)).any():
                self.phase, self.phase_t = "settle", 0
        elif self.phase == "settle":
            if (self.moved == 0 and self.phase_t > 10) or self.phase_t > 90:
                self.phase, self.phase_t = "drain", 0
        elif self.phase == "drain":
            self.hole = 1 + self.phase_t * 0.6
            if not self.G.any() or self.phase_t > 600:
                self.G[:] = 0
                self.hole = 0
                self.next_shape()

    def dissolve(self):
        if self.phase not in ("assemble", "hold"):
            return
        n = self.n
        st = self.st
        st[(st == WAIT)] = DEAD
        st[(st == FLY)] = LOCK
        noise = np.random.rand(n)
        if self.dis == "wind":
            delay = (self.PW - self.tx) / self.PW * 50 + noise * 12 + (self.ty / self.PH) * 6
        elif self.dis == "crumble":
            delay = (self.PH - self.ty) / self.PH * 25 + noise * 8
        else:
            delay = noise * 3
        self.rel = delay.astype(np.int32)
        self.phase, self.phase_t = "dissolve", 0

    def poke(self, px, py):
        if not self.n:
            return
        d = np.hypot(self.x - px, self.y - py)
        hit = (d < 5) & ((self.st == LOCK) | (self.st == FLY))
        idx = np.nonzero(hit)[0]
        if len(idx):
            self.st[idx] = FALL
            dd = d[idx] + 0.5
            self.vx[idx] = (self.x[idx] - px) / dd * 1.5
            self.vy[idx] = (self.y[idx] - py) / dd * 1.5 - 0.8

    # ------------------------------------------------------------------ render
    def render(self):
        PH, PW = self.PH, self.PW
        canvas = self.G.copy()
        if self.n:
            vis = (self.st != WAIT) & (self.st != DEAD)
            xi = np.rint(self.x[vis]).astype(int)
            yi = np.rint(self.y[vis]).astype(int)
            col = self.col[vis].astype(np.int8).copy()
            st = self.st[vis]
            # flying grains catch the light; held grains glint along a sweeping band
            col[st == FLY] = np.minimum(col[st == FLY] + 1, K)
            if self.phase in ("hold", "dissolve"):
                band = (self.frame_no - self.glint) * 3.0 - 20
                pos = xi + (PH - yi) * 0.6
                g = np.abs(pos - band) < 5
                col[g] = np.minimum(col[g] + 3, K)
                sp = np.random.rand(len(col)) < 0.006
                col[sp] = K
            ok = (xi >= 0) & (xi < PW) & (yi >= 0) & (yi < PH)
            canvas[yi[ok], xi[ok]] = col[ok]
        if PH % 2:
            canvas = canvas[:-1]
        top, bot = canvas[0::2].astype(int), canvas[1::2].astype(int)
        both = (top > 0) & (bot > 0)
        char = np.full(top.shape, " ", "<U1")
        char[(top > 0) & (bot == 0)] = "▀"
        char[(top == 0) & (bot > 0)] = "▄"
        char[both] = "▀"
        char[both & (top == bot)] = "█"
        fg = np.where(top > 0, top, bot)
        bg = np.where(both & (top != bot), bot, 0)
        pair = np.where(fg > 0, 1 + fg * (K + 1) + bg, 0)
        scr = self.scr
        for r in range(char.shape[0]):
            rc, rk = char[r].tolist(), pair[r].tolist()
            x = 0
            while x < PW:
                k = rk[x]
                e = x + 1
                while e < PW and rk[e] == k:
                    e += 1
                try:
                    scr.addstr(r + 1, x, "".join(rc[x:e]), curses.color_pair(k))
                except curses.error:
                    pass
                x = e
        self.status()
        if self.help:
            self.draw_help()

    def status(self):
        alive = int(((self.st != DEAD) & (self.st != WAIT)).sum()) if self.n else 0
        pile = int((self.G != 0).sum())
        parts = [" ⏳ SAND ", f" {self.label} ", f" {self.asm} → {self.dis} ",
                 f" {self.palette} ", f" {alive + pile} grains ", f" {self.phase} "]
        if self.speed > 1:
            parts.append(f" ×{self.speed} ")
        if self.paused:
            parts.append(" PAUSED ")
        line = "│".join(parts)
        line += ("  » " + self.msg) if self.msg and time.time() - self.msg_t < 2.5 else "  h: help"
        try:
            self.scr.addstr(0, 0, line[: self.W - 1].ljust(self.W - 1), curses.A_REVERSE)
        except curses.error:
            pass

    def draw_help(self):
        lines = [l for l in __doc__.split("Keys", 1)[1].splitlines() if l.strip()]
        lines = ["Keys"] + lines
        w = max(len(l) for l in lines) + 4
        h = len(lines) + 2
        y0, x0 = max(1, (self.H - h) // 2), max(0, (self.W - w) // 2)
        for i in range(h):
            if i in (0, h - 1):
                s = ("╭" if i == 0 else "╰") + "─" * (w - 2) + ("╮" if i == 0 else "╯")
            else:
                s = "│ " + lines[i - 1].ljust(w - 4) + " │"
            try:
                self.scr.addstr(y0 + i, x0, s[: self.W - x0 - 1], curses.A_BOLD)
            except curses.error:
                pass

    # ------------------------------------------------------------------ input
    def prompt(self):
        scr = self.scr
        scr.nodelay(False)
        curses.echo()
        curses.curs_set(1)
        label = " words to sculpt › "
        try:
            scr.addstr(0, 0, label.ljust(self.W - 1), curses.A_REVERSE | curses.A_BOLD)
            scr.move(0, len(label))
            text = scr.getstr(0, len(label), max(1, self.W - len(label) - 2)).decode("utf-8", "replace")
        except curses.error:
            text = ""
        curses.noecho()
        curses.curs_set(0)
        scr.nodelay(True)
        text = text.strip()
        if text:
            self.queue.append(text)
            self.say(f"next up: “{text}”")
            self.dissolve()

    def handle(self, k):
        if k in (ord('q'), 27):
            return False
        if k == ord(' '):
            if self.phase in ("assemble", "hold"):
                self.dissolve()
            else:
                self.say("already dissolving…")
        elif k == ord('t'):
            self.prompt()
        elif k == ord('a'):
            self.asm_i = 0 if self.asm_i is None else (self.asm_i + 1) % (len(ASSEMBLY) + 1)
            if self.asm_i == len(ASSEMBLY):
                self.asm_i = None
            self.say("assembly: " + ("rotate" if self.asm_i is None else ASSEMBLY[self.asm_i]) + " (next shape)")
        elif k == ord('d'):
            self.dis_i = 0 if self.dis_i is None else (self.dis_i + 1) % (len(DISSOLVE) + 1)
            if self.dis_i == len(DISSOLVE):
                self.dis_i = None
            if self.dis_i is not None and self.phase in ("assemble", "hold"):
                self.dis = DISSOLVE[self.dis_i]
            self.say("dissolve: " + ("rotate" if self.dis_i is None else DISSOLVE[self.dis_i]))
        elif k == ord('f'):
            self.font_i = 0 if self.font_i is None else self.font_i + 1
            if self.font_i >= len(self.fonts):
                self.font_i = None
            self.say("font: " + ("rotate" if self.font_i is None else self.fonts[self.font_i][0]) + " (next words)")
        elif k == ord('c'):
            names = list(PALETTES)
            i = names.index(self.palette) + 1
            self.pal_name = names[i % len(names)]
            self.apply_palette(self.pal_name)
            self.say("palette: " + self.pal_name)
        elif k in (ord('+'), ord('=')):
            self.speed = min(4, self.speed + 1)
        elif k == ord('-'):
            self.speed = max(1, self.speed - 1)
        elif k == ord('p'):
            self.paused = not self.paused
        elif k in (ord('h'), ord('?')):
            self.help = not self.help
        elif k == curses.KEY_RESIZE:
            self.resize()
        elif k == curses.KEY_MOUSE:
            try:
                _, mx, my, _, _ = curses.getmouse()
                self.poke(mx, (my - 1) * 2 + 1)
            except curses.error:
                pass
        return True

    def run(self):
        while True:
            t0 = time.time()
            k = self.scr.getch()
            while k != -1:
                if not self.handle(k):
                    return
                k = self.scr.getch()
            if not self.paused:
                for _ in range(self.speed):
                    self.frame_no += 1
                    self.step()
            self.render()
            self.scr.refresh()
            spare = 1 / 30 - (time.time() - t0)
            if spare > 0:
                time.sleep(spare)


def main(scr, args):
    curses.curs_set(0)
    scr.nodelay(True)
    scr.keypad(True)
    curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
    curses.mouseinterval(0)
    print("\033[?1002h", end="", flush=True)
    try:
        App(scr, args).run()
    finally:
        print("\033[?1002l", end="", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Terminal sand & calligraphy physics engine")
    ap.add_argument("words", nargs="*", help="words to sculpt (default: a built-in playlist)")
    ap.add_argument("--font", help="pin a font: " + ", ".join(n for n, _ in FONTS))
    ap.add_argument("--only-text", action="store_true", help="skip the fractals")
    args = ap.parse_args()
    locale.setlocale(locale.LC_ALL, "")
    curses.wrapper(main, args)
