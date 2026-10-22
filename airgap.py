#!/usr/bin/env python3
"""
airgap.py — an acoustic air-gap messenger.

Sends text and small files between two nearby computers through their
speakers and microphones.  16-tone FSK on near-ultrasonic (18–20 kHz),
audible or low carriers; a chirp preamble for sync, a tone-training sweep to
learn each room's frequency response, Reed–Solomon error correction and a
CRC-32 seal.  A live spectrogram waterfall shows everything the mic hears.

    python3 airgap.py                       # chat UI: type, Enter to send
    python3 airgap.py --profile audible     # chirpy & audible (more robust)
    python3 airgap.py --profile low         # low hum, for cheap speakers
    python3 airgap.py --volume 0.3          # transmit volume 0..1
    python3 airgap.py --hear-self           # decode your own transmissions
    python3 airgap.py send "hello there"    # one-shot, no UI
    python3 airgap.py send --file note.txt
    python3 airgap.py wav "hi" -o hi.wav    # render a transmission to WAV
    python3 airgap.py decode hi.wav         # decode a recording
    python3 airgap.py selftest              # simulate a noisy room, no audio

Chat commands
  /file PATH      send a small file (lands in ./received on the other side)
  /vol 0.4        transmit volume
  /profile NAME   ultra | audible | low   (both sides must match)
  /quit           leave (Esc and Ctrl-C work too)
"""
import argparse
import curses
import locale
import os
import queue
import struct
import sys
import time
import wave
import zlib
from collections import deque

import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None

SR = 48000
VERSION = 1
NSYM = 20            # RS parity bytes per data block (fixes 10 bad bytes per block)
HSYM = 8             # RS parity bytes on the header
BLOCK = 255 - NSYM
MAX_LEN = 16384


# ============================================================================ Reed–Solomon over GF(256)
class RSError(Exception):
    pass


GF_EXP = [0] * 512
GF_LOG = [0] * 256
_x = 1
for _i in range(255):
    GF_EXP[_i] = _x
    GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    GF_EXP[_i] = GF_EXP[_i - 255]


def gf_mul(x, y):
    if x == 0 or y == 0:
        return 0
    return GF_EXP[GF_LOG[x] + GF_LOG[y]]


def gf_div(x, y):
    if y == 0:
        raise ZeroDivisionError
    if x == 0:
        return 0
    return GF_EXP[(GF_LOG[x] + 255 - GF_LOG[y]) % 255]


def gf_pow(x, p):
    return GF_EXP[(GF_LOG[x] * p) % 255]


def gf_inv(x):
    return GF_EXP[255 - GF_LOG[x]]


def poly_scale(p, x):
    return [gf_mul(c, x) for c in p]


def poly_add(p, q):
    r = [0] * max(len(p), len(q))
    for i, c in enumerate(p):
        r[i + len(r) - len(p)] = c
    for i, c in enumerate(q):
        r[i + len(r) - len(q)] ^= c
    return r


def poly_mul(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for j, b in enumerate(q):
        for i, a in enumerate(p):
            r[i + j] ^= gf_mul(a, b)
    return r


def poly_eval(p, x):
    y = p[0]
    for c in p[1:]:
        y = gf_mul(y, x) ^ c
    return y


def poly_div(dividend, divisor):
    out = list(dividend)
    for i in range(len(dividend) - len(divisor) + 1):
        c = out[i]
        if c:
            for j in range(1, len(divisor)):
                if divisor[j]:
                    out[i + j] ^= gf_mul(divisor[j], c)
    sep = -(len(divisor) - 1)
    return out[:sep], out[sep:]


_GEN = {}


def rs_generator(nsym):
    if nsym not in _GEN:
        g = [1]
        for i in range(nsym):
            g = poly_mul(g, [1, gf_pow(2, i)])
        _GEN[nsym] = g
    return _GEN[nsym]


def rs_encode(msg, nsym):
    gen = rs_generator(nsym)
    res = list(msg) + [0] * nsym
    for i in range(len(msg)):
        c = res[i]
        if c:
            for j in range(1, len(gen)):
                res[i + j] ^= gf_mul(gen[j], c)
    return bytes(list(msg) + res[len(msg):])


def rs_decode(msg, nsym):
    """Returns (data, number_of_corrected_bytes)."""
    msg = list(msg)
    synd = [0] + [poly_eval(msg, gf_pow(2, i)) for i in range(nsym)]
    if max(synd) == 0:
        return bytes(msg[:-nsym]), 0
    # Berlekamp–Massey
    err_loc, old_loc = [1], [1]
    for i in range(nsym):
        k = i + 1
        delta = synd[k]
        for j in range(1, len(err_loc)):
            delta ^= gf_mul(err_loc[-(j + 1)], synd[k - j])
        old_loc = old_loc + [0]
        if delta:
            if len(old_loc) > len(err_loc):
                new_loc = poly_scale(old_loc, delta)
                old_loc = poly_scale(err_loc, gf_inv(delta))
                err_loc = new_loc
            err_loc = poly_add(err_loc, poly_scale(old_loc, delta))
    while err_loc and err_loc[0] == 0:
        err_loc.pop(0)
    errs = len(err_loc) - 1
    if errs * 2 > nsym:
        raise RSError("too many errors")
    # Chien search
    rev = err_loc[::-1]
    n = len(msg)
    err_pos = [n - 1 - i for i in range(n) if poly_eval(rev, gf_pow(2, i)) == 0]
    if len(err_pos) != errs:
        raise RSError("could not locate errors")
    # Forney
    coef_pos = [n - 1 - p for p in err_pos]
    e_loc = [1]
    for i in coef_pos:
        e_loc = poly_mul(e_loc, poly_add([1], [gf_pow(2, i), 0]))
    _, rem = poly_div(poly_mul(synd[::-1], e_loc), [1] + [0] * len(e_loc))
    err_eval = rem[::-1]
    X = [gf_pow(2, -(255 - p)) for p in coef_pos]
    E = [0] * n
    for i, Xi in enumerate(X):
        Xi_inv = gf_inv(Xi)
        prime = 1
        for j, Xj in enumerate(X):
            if j != i:
                prime = gf_mul(prime, 1 ^ gf_mul(Xi_inv, Xj))
        y = gf_mul(Xi, poly_eval(err_eval[::-1], Xi_inv))
        if prime == 0:
            raise RSError("bad error locator")
        E[err_pos[i]] = gf_div(y, prime)
    msg = poly_add(msg, E)
    if max(poly_eval(msg, gf_pow(2, i)) for i in range(nsym)):
        raise RSError("could not correct")
    return bytes(msg[:-nsym]), errs


# ============================================================================ modem
class Profile:
    def __init__(self, name, f0, df, sym, guard, desc):
        self.name, self.desc = name, desc
        self.M = 16
        self.freqs = f0 + df * np.arange(self.M)
        self.fa, self.fb = f0 - df * 0.8, self.freqs[-1] + df * 0.8
        self.Ns = int(sym * SR)
        self.Ng = int(guard * SR)
        self.Nt = int(0.004 * SR)                  # raised-cosine taper on each symbol
        self.Nw = self.Ns - self.Ng - self.Nt      # analysis window
        self.Lc = int(0.12 * SR)                   # chirp
        self.gap = int(0.03 * SR)
        self.bps = 4 / sym
        n = np.arange(self.Nw)
        self.basis = (np.hanning(self.Nw)[:, None] *
                      np.exp(-2j * np.pi * self.freqs[None, :] * n[:, None] / SR))
        t = np.arange(self.Lc) / SR
        T = self.Lc / SR
        ch = np.sin(2 * np.pi * (self.fa * t + (self.fb - self.fa) * t * t / (2 * T)))
        self.chirp = (ch * tukey(self.Lc, 0.15)).astype(np.float32)
        ts = np.arange(self.Ns) / SR
        env = np.ones(self.Ns)
        rc = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, self.Nt))
        env[:self.Nt], env[-self.Nt:] = rc, rc[::-1]
        self.tones = [(np.sin(2 * np.pi * f * ts) * env).astype(np.float32) for f in self.freqs]
        self.view = (max(50, self.fa - 700), self.fb + 700)

    def band_mask(self, nfft):
        f = np.fft.rfftfreq(nfft, 1 / SR)
        return ((f > self.fa - 250) & (f < self.fb + 250)).astype(np.float32)


def tukey(n, a):
    w = np.ones(n)
    k = int(a * n / 2)
    if k:
        r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
        w[:k], w[-k:] = r, r[::-1]
    return w


PROFILES = {
    "ultra":   Profile("ultra", 18000, 120, 0.040, 0.010, "near-ultrasonic 18.0–19.8 kHz"),
    "audible": Profile("audible", 1200, 160, 0.035, 0.008, "audible 1.2–3.6 kHz"),
    "low":     Profile("low", 300, 60, 0.070, 0.012, "low 300–1200 Hz"),
}


def to_nibbles(data):
    out = []
    for b in data:
        out += [b >> 4, b & 15]
    return out


def from_nibbles(nb):
    return bytes((nb[i] << 4) | nb[i + 1] for i in range(0, len(nb) - 1, 2))


def make_payload(text=None, path=None):
    if path:
        name = os.path.basename(path).encode()[:200]
        with open(path, "rb") as f:
            data = f.read()
        return b"F" + bytes([len(name)]) + name + data
    return b"T" + text.encode("utf-8")


def frame_bytes(payload):
    frame = payload + struct.pack(">I", zlib.crc32(payload))
    if len(frame) > MAX_LEN:
        raise ValueError(f"too big: {len(frame)} bytes (max {MAX_LEN})")
    hdr = rs_encode(struct.pack(">HB", len(frame), VERSION), HSYM)
    body = b"".join(rs_encode(frame[i:i + BLOCK], NSYM) for i in range(0, len(frame), BLOCK))
    return hdr, body


def modulate(payload, prof, volume=0.5):
    hdr, body = frame_bytes(payload)
    syms = list(range(prof.M)) + to_nibbles(hdr) + to_nibbles(body)
    parts = [np.zeros(int(0.05 * SR), np.float32), prof.chirp, np.zeros(prof.gap, np.float32)]
    parts += [prof.tones[s] for s in syms]
    parts.append(np.zeros(int(0.1 * SR), np.float32))
    w = np.concatenate(parts) * volume
    return w.astype(np.float32)


def body_len(frame_len):
    return sum(min(BLOCK, frame_len - i) + NSYM for i in range(0, frame_len, BLOCK))


class Receiver:
    """Streaming demodulator: feed() it samples, it yields events."""
    THRESH = 0.38

    def __init__(self, prof):
        self.p = prof
        self.buf = np.zeros(0, np.float32)
        self.base = 0
        self.state = "search"
        self.search_from = 0
        self.ignore_until = 0
        self.last_corr = 0.0
        self.mask_cache = {}

    @property
    def end(self):
        return self.base + len(self.buf)

    def at(self, a, n):
        i = a - self.base
        return self.buf[i:i + n]

    def feed(self, x):
        self.buf = np.concatenate([self.buf, x.astype(np.float32)])
        ev = []
        while True:
            fn = getattr(self, "st_" + self.state)
            r = fn(ev)
            if not r:
                break
        self.trim()
        return ev

    def trim(self):
        keep = self.search_from - self.p.Lc if self.state == "search" else self.sync - SR // 10
        keep = max(keep, self.end - SR * 60)
        cut = keep - self.base
        if cut > SR:
            self.buf = self.buf[cut:]
            self.base += cut

    # -- searching for the chirp ------------------------------------------------
    def ncorr(self, seg):
        Lc = self.p.Lc
        nfft = 1 << int(np.ceil(np.log2(len(seg) + Lc)))
        if nfft not in self.mask_cache:
            self.mask_cache[nfft] = (self.p.band_mask(nfft), np.conj(np.fft.rfft(self.p.chirp, nfft)))
        mask, C = self.mask_cache[nfft]
        S = np.fft.rfft(seg, nfft) * mask
        bp = np.fft.irfft(S, nfft)[:len(seg)]
        corr = np.fft.irfft(S * C, nfft)[:len(seg) - Lc + 1]
        cs = np.concatenate([[0.0], np.cumsum(bp.astype(np.float64) ** 2)])
        E = cs[Lc:] - cs[:-Lc]
        ec = float(np.sum(self.p.chirp.astype(np.float64) ** 2))
        return corr / (np.sqrt(np.maximum(E, 1e-12) * ec) + 1e-9)

    def st_search(self, ev):
        Lc = self.p.Lc
        start = max(self.search_from, self.base, self.ignore_until)
        if self.end - start < 2 * Lc:
            return False
        nc = self.ncorr(self.at(start, self.end - start))
        limit = len(nc) - Lc // 2
        if limit <= 0:
            return False
        self.last_corr = float(nc[:limit].max())
        hits = np.nonzero(nc[:limit] > self.THRESH)[0]
        if not len(hits):
            self.search_from = start + limit
            return False
        j = hits[0]
        if j + Lc // 2 > len(nc):
            return False
        i = j + int(np.argmax(nc[j:j + Lc // 2]))
        self.sync = start + i
        self.corr = float(nc[i])
        self.state = "train"
        ev.append(("sync", self.corr))
        return True

    # -- tone training: learn timing and per-tone gain -----------------------
    def energies(self, start):
        p = self.p
        seg = self.at(start + p.Ng, p.Nw)
        if len(seg) < p.Nw:
            return None
        return np.abs(seg @ p.basis) ** 2

    def st_train(self, ev):
        p = self.p
        t0 = self.sync + p.Lc + p.gap
        slack = p.Ng // 2
        if self.end < t0 + (p.M + 1) * p.Ns + slack:
            return False
        best = None
        for off in range(-slack, slack + 1, max(1, SR // 2000)):
            E = np.array([self.energies(t0 + off + k * p.Ns) for k in range(p.M)])
            diag = np.diag(E) / (E.sum(1) + 1e-20)
            score = diag.mean()
            if best is None or score > best[0]:
                best = (score, off, E)
        score, off, E = best
        if score < 0.35:
            ev.append(("false", f"training failed ({score:.2f})"))
            return self.reset(self.sync + p.Lc // 2)
        g = np.diag(E)
        self.amp = np.sqrt(np.maximum(g, g.mean() * 1e-3))   # expected amplitude of each tone
        off_diag = (E.sum() - g.sum()) / (p.M * (p.M - 1))
        self.snr = 10 * np.log10(g.mean() / (off_diag + 1e-20))
        self.t0 = t0 + off + p.M * p.Ns
        self.state = "header"
        return True

    def decide(self, k0, n):
        # High-SNR approximation of the ML metric for tones whose channel gains we
        # learned during training; s tracks slow fading of the overall level.
        out, a, s = [], self.amp, 1.0
        for k in range(k0, k0 + n):
            r = np.sqrt(self.energies(self.t0 + k * self.p.Ns))
            j = int(np.argmax(r * s * a - 0.5 * (s * a) ** 2))
            s = 0.85 * s + 0.15 * float(np.clip(r[j] / a[j], 0.2, 5))
            out.append(j)
        return out

    def st_header(self, ev):
        p = self.p
        n = (3 + HSYM) * 2
        if self.end < self.t0 + (n + 1) * p.Ns:
            return False
        raw = from_nibbles(self.decide(0, n))
        try:
            hdr, fixed = rs_decode(raw, HSYM)
        except RSError:
            ev.append(("false", "header unreadable"))
            return self.reset(self.sync + p.Lc // 2)
        flen, ver = struct.unpack(">HB", hdr)
        if ver != VERSION or not 5 <= flen <= MAX_LEN:
            ev.append(("false", "header nonsense"))
            return self.reset(self.sync + p.Lc // 2)
        self.flen = flen
        self.k_body = n
        self.n_body = body_len(flen) * 2
        self.hdr_fixed = fixed
        self.state = "body"
        self.t_start = time.time()
        ev.append(("header", flen, self.snr))
        return True

    def st_body(self, ev):
        p = self.p
        have = (self.end - self.t0) // p.Ns - self.k_body - 1
        if have < self.n_body:
            k = max(0, min(have, self.n_body))
            preview = from_nibbles(self.decide(self.k_body, min(k, 120) & ~1))
            ev.append(("progress", k / self.n_body, preview))
            return False
        raw = from_nibbles(self.decide(self.k_body, self.n_body))
        frame, fixed, ok = b"", self.hdr_fixed, True
        i = 0
        for start in range(0, self.flen, BLOCK):
            size = min(BLOCK, self.flen - start) + NSYM
            blk = raw[i:i + size]
            i += size
            try:
                d, f = rs_decode(blk, NSYM)
                fixed += f
            except RSError:
                d, ok = blk[:-NSYM], False
            frame += d
        payload, crc = frame[:-4], frame[-4:]
        good = ok and struct.pack(">I", zlib.crc32(payload)) == crc
        ev.append(("message", payload, good, fixed, self.snr))
        end = self.t0 + (self.k_body + self.n_body) * p.Ns
        return self.reset(end)

    def reset(self, frm):
        self.state = "search"
        self.search_from = frm
        return True


def parse_payload(payload):
    if payload[:1] == b"F" and len(payload) >= 2:
        n = payload[1]
        return "file", payload[2:2 + n].decode("utf-8", "replace"), payload[2 + n:]
    return "text", None, payload[1:].decode("utf-8", "replace")


def save_file(name, data):
    os.makedirs("received", exist_ok=True)
    name = os.path.basename(name) or "file.bin"
    path = os.path.join("received", name)
    root, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(path):
        path = f"{root}-{i}{ext}"
        i += 1
    with open(path, "wb") as f:
        f.write(data)
    return path


# ============================================================================ wav helpers
def write_wav(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def read_wav(path):
    with wave.open(path, "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    dt = {1: np.uint8, 2: "<i2", 4: "<i4"}[sw]
    x = np.frombuffer(raw, dt).astype(np.float32)
    if sw == 1:
        x = (x - 128) / 128
    else:
        x /= float(2 ** (8 * sw - 1))
    x = x.reshape(-1, ch).mean(1)
    if sr != SR:
        t = np.arange(int(len(x) * SR / sr)) * sr / SR
        x = np.interp(t, np.arange(len(x)), x).astype(np.float32)
    return x


# ============================================================================ chat UI
HEAT = [16, 17, 18, 19, 20, 56, 92, 128, 164, 200, 198, 202, 208, 214, 220, 226, 229, 231]
HEAT8 = [0, 4, 4, 5, 5, 1, 1, 3, 3, 7]


class Chat:
    def __init__(self, scr, args):
        self.scr = scr
        self.args = args
        self.volume = args.volume
        self.set_profile(args.profile)
        self.q = queue.Queue()
        self.tx = None
        self.tx_pos = 0
        self.tx_end_abs = 0
        self.log = deque(maxlen=200)
        self.line = ""
        self.status = "listening"
        self.status_t = 0
        self.preview = b""
        self.prog = None
        self.level = 0.0
        self.mic_abs = 0
        self.tail = np.zeros(4096, np.float32)
        self.rows = deque(maxlen=200)
        self.floor = None
        self.last_tone = None
        self.colors()
        self.add("✦", f"airgap ready · {self.p.desc} · {self.p.bps:.0f} baud raw · "
                      f"type a message and press Enter · /file PATH sends a file", 4)

    def set_profile(self, name):
        self.p = PROFILES[name]
        self.rx = Receiver(self.p)
        self.rx.base = getattr(self, "mic_abs", 0)
        self.rx.search_from = self.rx.base

    def colors(self):
        curses.start_color()
        try:
            curses.use_default_colors()
            bg = -1
        except curses.error:
            bg = 0
        self.c256 = curses.COLORS >= 256
        heat = HEAT if self.c256 else HEAT8
        for i, c in enumerate(heat):
            curses.init_pair(10 + i, 0, c)
        self.nheat = len(heat)
        curses.init_pair(1, 51 if self.c256 else 6, bg)    # them
        curses.init_pair(2, 213 if self.c256 else 5, bg)   # you
        curses.init_pair(3, 196 if self.c256 else 1, bg)   # error
        curses.init_pair(4, 245 if self.c256 else 7, bg)   # dim
        curses.init_pair(5, 226 if self.c256 else 3, bg)   # markers
        curses.init_pair(6, 46 if self.c256 else 2, bg)    # good

    def add(self, who, text, cp):
        self.log.append((time.strftime("%H:%M:%S"), who, text, cp))

    # -- audio ------------------------------------------------------------------
    def mic_cb(self, indata, frames, ti, status):
        self.q.put(indata[:, 0].copy())

    def spk_cb(self, out, frames, ti, status):
        out.fill(0)
        tx = self.tx
        if tx is not None:
            n = min(frames, len(tx) - self.tx_pos)
            out[:n, 0] = tx[self.tx_pos:self.tx_pos + n]
            self.tx_pos += n
            if self.tx_pos >= len(tx):
                self.tx = None
        if out.shape[1] > 1:
            out[:, 1] = out[:, 0]

    def send(self, payload, label):
        if self.tx is not None:
            self.add("!", "still transmitting — wait a moment", 3)
            return
        try:
            w = modulate(payload, self.p, self.volume)
        except ValueError as e:
            self.add("!", str(e), 3)
            return
        if not self.args.hear_self:
            self.rx.ignore_until = self.mic_abs + len(w) + SR // 2
        self.tx_pos = 0
        self.tx = w
        self.tx_len = len(w)
        self.add("▸", f"{label}   ({len(w) / SR:.1f} s on air)", 2)

    # -- input --------------------------------------------------------------------
    def command(self, s):
        parts = s.split(maxsplit=1)
        cmd, arg = parts[0], parts[1] if len(parts) > 1 else ""
        if cmd == "/quit":
            return False
        if cmd == "/file":
            path = os.path.expanduser(arg.strip().strip("'\""))
            if not os.path.isfile(path):
                self.add("!", f"no such file: {path}", 3)
            else:
                size = os.path.getsize(path)
                if size > 4096:
                    self.add("!", f"{size} bytes will take ~{size * 2 / self.p.bps * 1.2 * 4:.0f} s… sending anyway", 4)
                self.send(make_payload(path=path), f"file {os.path.basename(path)} ({size} bytes)")
        elif cmd == "/vol":
            try:
                self.volume = min(1.0, max(0.0, float(arg)))
                self.add("✦", f"transmit volume {self.volume:.2f}", 4)
            except ValueError:
                self.add("!", "usage: /vol 0.4", 3)
        elif cmd == "/profile":
            if arg in PROFILES:
                self.set_profile(arg)
                self.floor = None
                self.rows.clear()
                self.add("✦", f"profile {arg}: {self.p.desc}", 4)
            else:
                self.add("!", "profiles: " + ", ".join(PROFILES), 3)
        else:
            self.add("!", "commands: /file PATH · /vol 0.4 · /profile ultra|audible|low · /quit", 3)
        return True

    def key(self, ch):
        if ch in ("\x1b",):
            return False
        if ch in ("\n", "\r", curses.KEY_ENTER):
            s, self.line = self.line.strip(), ""
            if not s:
                return True
            if s.startswith("/"):
                return self.command(s)
            self.send(make_payload(text=s), s)
        elif ch in (curses.KEY_BACKSPACE, "\x7f", "\b", 127, 8):
            self.line = self.line[:-1]
        elif ch == curses.KEY_RESIZE:
            self.scr.clear()
        elif isinstance(ch, str) and ch.isprintable():
            self.line += ch
        return True

    # -- receive ------------------------------------------------------------------
    def pump(self):
        got = []
        while True:
            try:
                got.append(self.q.get_nowait())
            except queue.Empty:
                break
        if not got:
            return
        x = np.concatenate(got)
        self.mic_abs += len(x)
        self.level = 0.8 * self.level + 0.2 * float(np.sqrt(np.mean(x * x)))
        self.tail = np.concatenate([self.tail, x])[-len(self.tail):]
        for e in self.rx.feed(x):
            self.event(e)

    def event(self, e):
        kind = e[0]
        if kind == "sync":
            self.status, self.status_t = f"⟟ chirp locked (corr {e[1]:.2f})", time.time()
        elif kind == "false":
            self.status, self.status_t = f"… {e[1]}", time.time()
            self.prog = None
        elif kind == "header":
            self.status = f"receiving {e[1]} bytes · SNR {e[2]:.0f} dB"
            self.status_t = time.time()
            self.prog = 0.0
        elif kind == "progress":
            self.prog, self.preview = e[1], e[2]
        elif kind == "message":
            payload, good, fixed, snr = e[1:]
            self.prog = None
            note = f"   (RS fixed {fixed} byte{'s' if fixed != 1 else ''} · SNR {snr:.0f} dB)"
            try:
                typ, name, data = parse_payload(payload)
            except Exception:
                typ, name, data = "text", None, repr(payload)
            if typ == "file" and good:
                path = save_file(name, data)
                self.add("◂", f"file {name} ({len(data)} bytes) → {path}{note}", 1)
            elif good:
                self.add("◂", data + note, 1)
            else:
                shown = data if typ == "text" else f"file {name}"
                self.add("✗", f"garbled: {shown[:200]!s}   (checksum failed)", 3)
            self.status, self.status_t = "listening", time.time()

    # -- drawing ------------------------------------------------------------------
    def spectrum_row(self, W):
        n = len(self.tail)
        X = np.abs(np.fft.rfft(self.tail * np.hanning(n))) + 1e-9
        db = 20 * np.log10(X)
        f = np.fft.rfftfreq(n, 1 / SR)
        lo, hi = self.p.view
        edges = np.linspace(lo, hi, W + 1)
        idx = np.searchsorted(f, edges)
        cols = np.array([db[idx[i]:max(idx[i] + 1, idx[i + 1])].max() for i in range(W)])
        if self.floor is None or len(self.floor) != W:
            self.floor = cols.copy()
        self.floor = np.minimum(self.floor * 0.995 + cols * 0.005, cols + 3)
        self.floor = np.where(cols < self.floor, 0.9 * self.floor + 0.1 * cols, self.floor)
        lv = np.clip((cols - self.floor - 3) / 35, 0, 1)
        return lv

    def freq_col(self, f, W):
        lo, hi = self.p.view
        return int((f - lo) / (hi - lo) * W)

    def draw(self):
        scr = self.scr
        H, W = scr.getmaxyx()
        Wd = W - 1
        scr.erase()

        def put(y, x, s, a=0):
            if 0 <= y < H and 0 <= x < Wd:
                try:
                    scr.addstr(y, x, s[: Wd - x], a)
                except curses.error:
                    pass

        meter = int(min(1, self.level * 40) * 12)
        head = (f" ◉ AIRGAP · {self.p.desc} · 16-FSK {self.p.bps:.0f} baud · RS+CRC32 · "
                f"tx vol {self.volume:.2f} · mic {'▮' * meter}{'▯' * (12 - meter)} ")
        put(0, 0, head.ljust(Wd), curses.A_REVERSE | curses.A_BOLD)

        wf_h = max(5, min(H // 2, H - 10))
        self.rows.appendleft(self.spectrum_row(Wd))
        for r, row in enumerate(list(self.rows)[:wf_h]):
            lv = (row * (self.nheat - 1)).astype(int)
            x = 0
            while x < Wd:
                k = lv[x]
                e = x + 1
                while e < Wd and lv[e] == k:
                    e += 1
                put(1 + r, x, " " * (e - x), curses.color_pair(10 + k))
                x = e
        # axis with tone markers
        ay = 1 + wf_h
        axis = [" "] * Wd
        lo, hi = self.p.view
        step = 500 if hi - lo < 6000 else 1000
        if hi - lo < 2500:
            step = 200
        f = (int(lo) // step + 1) * step
        while f < hi:
            c = self.freq_col(f, Wd)
            lab = f"{f / 1000:.1f}k" if f >= 1000 else f"{f}"
            for i, ch in enumerate("┴" + lab):
                if 0 <= c + i < Wd:
                    axis[c + i] = ch
            f += step
        put(ay, 0, "".join(axis), curses.color_pair(4))
        for t in self.p.freqs:
            put(ay, self.freq_col(t, Wd), "▾", curses.color_pair(5))

        # status / progress
        sy = ay + 1
        if self.tx is not None:
            fr = self.tx_pos / max(1, self.tx_len)
            bw = 30
            put(sy, 1, f"transmitting ▕{'█' * int(fr * bw)}{'░' * (bw - int(fr * bw))}▏ {fr * 100:3.0f}%",
                curses.color_pair(2) | curses.A_BOLD)
        elif self.prog is not None:
            bw = 30
            pv = "".join(chr(b) if 32 <= b < 127 else "·" for b in self.preview[1:60])
            put(sy, 1, f"{self.status}  ▕{'█' * int(self.prog * bw)}{'░' * (bw - int(self.prog * bw))}▏ "
                f"{self.prog * 100:3.0f}%   raw: {pv}", curses.color_pair(1) | curses.A_BOLD)
        else:
            st = self.status if time.time() - self.status_t < 3 else "listening"
            spin = "◐◓◑◒"[int(time.time() * 4) % 4]
            put(sy, 1, f"{spin} {st}   (best chirp match {self.rx.last_corr:.2f}, need {Receiver.THRESH})",
                curses.color_pair(4))

        # log
        top = sy + 2
        avail = H - top - 2
        lines = []
        for ts, who, text, cp in self.log:
            first = f"{ts} {who} "
            width = max(10, Wd - len(first) - 1)
            chunks = [text[i:i + width] for i in range(0, max(1, len(text)), width)] or [""]
            lines.append((first + chunks[0], cp))
            for c in chunks[1:]:
                lines.append((" " * len(first) + c, cp))
        for i, (s, cp) in enumerate(lines[-avail:] if avail > 0 else []):
            put(top + i, 1, s, curses.color_pair(cp))
        # input
        prompt = "› "
        put(H - 1, 0, (prompt + self.line)[-(Wd - 1):].ljust(Wd), curses.A_BOLD)
        try:
            scr.move(H - 1, min(Wd - 1, len(prompt) + len(self.line)))
        except curses.error:
            pass
        scr.refresh()

    def run(self):
        if sd is None:
            raise SystemExit("sounddevice is required: pip install sounddevice")
        mic = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=1200,
                             callback=self.mic_cb, device=self.args.input)
        spk = sd.OutputStream(samplerate=SR, channels=1, dtype="float32", blocksize=1200,
                              callback=self.spk_cb, device=self.args.output)
        with mic, spk:
            while True:
                try:
                    while True:
                        ch = self.scr.get_wch()
                        if not self.key(ch):
                            return
                except curses.error:
                    pass
                except KeyboardInterrupt:
                    return
                self.pump()
                self.draw()
                time.sleep(1 / 30)


# ============================================================================ CLI modes
def cli_send(args):
    if sd is None:
        raise SystemExit("sounddevice is required: pip install sounddevice")
    prof = PROFILES[args.profile]
    payload = make_payload(path=args.file) if args.file else make_payload(text=" ".join(args.text))
    w = modulate(payload, prof, args.volume)
    dur = len(w) / SR
    print(f"◉ {prof.desc} · {len(payload)} byte payload · {dur:.1f} s on air")
    sd.play(w, SR, device=args.output)
    t0 = time.time()
    while time.time() - t0 < dur:
        fr = (time.time() - t0) / dur
        print(f"\r  ▕{'█' * int(fr * 40)}{'░' * (40 - int(fr * 40))}▏ {fr * 100:3.0f}%", end="", flush=True)
        time.sleep(0.05)
    sd.wait()
    print(f"\r  ▕{'█' * 40}▏ 100%  sent.")


def decode_signal(x, prof, chunk=2400):
    rx = Receiver(prof)
    out = []
    for i in range(0, len(x), chunk):
        for e in rx.feed(x[i:i + chunk]):
            if e[0] == "message":
                out.append(e)
    for e in rx.feed(np.zeros(SR, np.float32)):
        if e[0] == "message":
            out.append(e)
    return out


def report(msgs):
    if not msgs:
        print("  nothing decoded.")
    for _, payload, good, fixed, snr in msgs:
        typ, name, data = parse_payload(payload)
        tag = "✓" if good else "✗ (checksum failed)"
        if typ == "file":
            path = save_file(name, data) if good else "-"
            print(f"  {tag} file {name} ({len(data)} bytes) → {path}   RS fixed {fixed}, SNR {snr:.0f} dB")
        else:
            print(f"  {tag} “{data}”   RS fixed {fixed}, SNR {snr:.0f} dB")


def channel(w, rng, gain=0.08, noise=0.004, drift=40e-6, echo=True, burst=True, tilt=True):
    """A cruel simulated room: lead-in, echoes, tilt, noise, a clap, clock drift."""
    x = np.concatenate([np.zeros(int(rng.uniform(0.2, 1.0) * SR)), w, np.zeros(SR // 2)])
    if echo:
        h = np.zeros(int(0.009 * SR))
        h[0], h[int(0.003 * SR)], h[-1] = 1.0, 0.45, 0.25
        x = np.convolve(x, h)[:len(x)]
    if tilt:  # treble roll-off like a tiny laptop speaker
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(len(x), 1 / SR)
        X *= 1 / (1 + (f / 9000) ** 2)
        x = np.fft.irfft(X, len(x))
    x *= gain
    t = np.arange(len(x)) / SR
    x += rng.normal(0, noise, len(x)) + 0.02 * np.sin(2 * np.pi * 120 * t)
    if burst:
        b = int(len(x) * 0.55)
        x[b:b + 600] += rng.normal(0, 0.3, min(600, len(x) - b))
    if drift:  # band-limited resample, as if the two sound cards' clocks disagree
        n = int(len(x) / (1 + drift))
        X = np.fft.rfft(x)[: n // 2 + 1]
        x = np.fft.irfft(X, n) * n / len(x)
    return x.astype(np.float32)


def selftest(args):
    rng = np.random.default_rng(7)
    print("Reed–Solomon:", end=" ")
    for trial in range(200):
        msg = bytes(rng.integers(0, 256, rng.integers(1, BLOCK)).tolist())
        enc = bytearray(rs_encode(msg, NSYM))
        k = int(rng.integers(0, NSYM // 2 + 1))
        for pos in rng.choice(len(enc), k, replace=False):
            enc[pos] ^= int(rng.integers(1, 256))
        dec, fixed = rs_decode(bytes(enc), NSYM)
        assert dec == msg and fixed == k, (trial, k, fixed)
    print("200 random blocks with up to 10 byte errors each — all repaired ✓")
    text = "The quick brown fox whispers across the air gap at 19 kHz. 🦊"
    for name in (args.profile_list or list(PROFILES)):
        prof = PROFILES[name]
        w = modulate(make_payload(text=text), prof, 0.5)
        x = channel(w, rng)
        t0 = time.time()
        msgs = decode_signal(x, prof)
        dt = time.time() - t0
        ok = msgs and msgs[0][2] and parse_payload(msgs[0][1])[2] == text
        print(f"{name:>8}: {len(w) / SR:4.1f} s of audio, decoded in {dt:.2f} s →", "✓" if ok else "✗")
        if msgs:
            report(msgs)


def main():
    ap = argparse.ArgumentParser(description="Acoustic air-gap messenger (FSK + Reed–Solomon)")
    ap.add_argument("mode", nargs="?", default="chat", choices=["chat", "send", "wav", "decode", "selftest"])
    ap.add_argument("text", nargs="*", help="message (send/wav) or WAV path (decode)")
    ap.add_argument("--file", help="send this file instead of text")
    ap.add_argument("--profile", default="ultra", choices=list(PROFILES))
    ap.add_argument("--volume", type=float, default=0.5, help="transmit volume 0..1")
    ap.add_argument("--hear-self", action="store_true", help="decode your own transmissions too")
    ap.add_argument("-o", "--out", default="airgap.wav")
    ap.add_argument("--input", type=int, default=None, help="input device index")
    ap.add_argument("--output", type=int, default=None, help="output device index")
    args = ap.parse_args()
    args.profile_list = None
    if args.mode == "send":
        if not args.text and not args.file:
            ap.error("send needs text or --file")
        cli_send(args)
    elif args.mode == "wav":
        payload = make_payload(path=args.file) if args.file else make_payload(text=" ".join(args.text))
        w = modulate(payload, PROFILES[args.profile], args.volume)
        write_wav(args.out, w)
        print(f"wrote {args.out} ({len(w) / SR:.1f} s, {PROFILES[args.profile].desc})")
    elif args.mode == "decode":
        if not args.text:
            ap.error("decode needs a WAV path")
        report(decode_signal(read_wav(args.text[0]), PROFILES[args.profile]))
    elif args.mode == "selftest":
        selftest(args)
    else:
        locale.setlocale(locale.LC_ALL, "")

        def ui(scr):
            curses.curs_set(1)
            scr.nodelay(True)
            scr.keypad(True)
            Chat(scr, args).run()
        try:
            curses.wrapper(ui)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
