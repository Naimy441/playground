#!/usr/bin/env python3
"""
airgap.py — an acoustic air-gap messenger.

Sends text and files between two nearby computers through their speakers
and microphones.  Parallel multi-tone FSK (several groups of tones sound at
once) on near-ultrasonic (18–20 kHz), audible or low carriers; a chirp
preamble for sync, a tone-training sweep that learns each room's frequency
response *and its echo*, a detector that subtracts the echo it predicts,
interleaved Reed–Solomon with soft-decision (GMD) decoding, and CRC-32 seals.
Payloads are split into frames that each decompress on their own.  In chat
mode the receiver asks for frames that failed, at most MAX_RESENDS times;
every copy is scrambled differently and the receiver adds up the soft bits of
all copies, so two copies that are each too damaged to read usually decode
together.  If that still isn't enough it saves what it could, marks the gaps,
and tells the sender to stop.  A live spectrogram waterfall shows everything
the mic hears.

    python3 airgap.py                       # chat UI: type, Enter to send
    python3 airgap.py --profile audible     # chirpy & audible (more robust)
    python3 airgap.py --profile low         # low hum, for cheap speakers
    python3 airgap.py --profile all         # every band at once: fastest, loudest
    python3 airgap.py --volume 0.3          # transmit volume 0..1
    python3 airgap.py --hear-self           # decode your own transmissions
    python3 airgap.py send "hello there"    # one-shot, no UI
    python3 airgap.py send --file note.txt
    python3 airgap.py wav "hi" -o hi.wav    # render a transmission to WAV
    python3 airgap.py decode hi.wav         # decode a recording
    python3 airgap.py selftest              # simulate a noisy room, no audio

Chat commands
  /file PATH      send a file (lands in ./received on the other side)
  /vol 0.4        transmit volume
  /profile NAME   ultra | audible | low | all   (both sides must match)
  /quit           leave (Esc and Ctrl-C work too)
"""
import argparse
import curses
import hashlib
import locale
import lzma
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
VERSION = 3
NSYM = 32            # RS parity bytes per data block (fixes 16 errors, or up to 32 erasures)
HSYM = 12            # RS parity bytes on the header (it can't be soft-combined across copies)
BLOCK = 255 - NSYM
HDR = struct.Struct(">HHHHBB")         # message id, frame seq, frame count, frame length, flags, version
HDR_LEN = HDR.size + HSYM
CTL = 0x80           # flags bit: an ack / resend request, which is never resent; the low bits are the copy number
MAX_FRAMES = 65535
NAK_WAIT = 4.0       # quiet seconds before a receiver asks for missing frames
ACK_WAIT = 15.0      # quiet seconds before a sender resends a message nobody acknowledged
REPLY_GAP = 1.0      # seconds to hold an ack / request back, so the sender has stopped and is listening
MAX_RESENDS = 2      # extra copies a sender plays of any frame, asked for or not; then both sides stop
TRACK_GAIN = 2.0     # how hard the early/late gate pulls the symbol window back into place
LLR_CAP = 30.0       # most a single copy may vouch for one bit (≈ 1e-13 odds of being wrong)
CLIP_PCT = 99.0
LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "dict_size": 1 << 20}]


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


def rs_decode(msg, nsym, erase_pos=()):
    """Errors-and-erasures decode. Returns (data, number_of_corrected_bytes)."""
    orig = list(msg)
    msg = list(msg)
    erase_pos = list(erase_pos)
    if len(erase_pos) > nsym:
        raise RSError("too many erasures")
    for p in erase_pos:
        msg[p] = 0
    n = len(msg)
    synd = [0] + [poly_eval(msg, gf_pow(2, i)) for i in range(nsym)]
    if max(synd) == 0:
        return bytes(msg[:-nsym]), sum(a != b for a, b in zip(msg, orig))
    # Forney syndromes: strip the erasures out so Berlekamp–Massey only sees errors
    fsynd = synd[1:]
    for p in erase_pos:
        x = gf_pow(2, n - 1 - p)
        for j in range(len(fsynd) - 1):
            fsynd[j] = gf_mul(fsynd[j], x) ^ fsynd[j + 1]
    # Berlekamp–Massey
    err_loc, old_loc = [1], [1]
    for i in range(nsym - len(erase_pos)):
        delta = fsynd[i]
        for j in range(1, len(err_loc)):
            delta ^= gf_mul(err_loc[-(j + 1)], fsynd[i - j])
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
    if errs * 2 + len(erase_pos) > nsym:
        raise RSError("too many errors")
    # Chien search
    rev = err_loc[::-1]
    err_pos = [n - 1 - i for i in range(n) if poly_eval(rev, gf_pow(2, i)) == 0]
    if len(err_pos) != errs:
        raise RSError("could not locate errors")
    # Forney, over errors and erasures together
    coef_pos = [n - 1 - p for p in erase_pos + err_pos]
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
        E[n - 1 - coef_pos[i]] = gf_div(y, prime)
    msg = poly_add(msg, E)
    if max(poly_eval(msg, gf_pow(2, i)) for i in range(nsym)):
        raise RSError("could not correct")
    return bytes(msg[:-nsym]), sum(a != b for a, b in zip(msg, orig))


def rs_decode_soft(blk, conf, nsym):
    """Generalized minimum-distance decoding: erase 0, 2, 4 … of the least confident bytes and, of all the
    codewords that come out, keep the one that disagrees with what we heard only where we were least sure."""
    weakest = [int(i) for i in np.argsort(conf, kind="stable")]
    rx = np.frombuffer(bytes(blk), np.uint8)
    best, err = None, RSError("undecodable")
    for k in range(0, nsym - 1, 2):
        try:
            d, f = rs_decode(blk, nsym, sorted(weakest[:k]))
        except RSError as e:
            err = e
            continue
        if k == 0 and f <= nsym // 4:          # few errors, no erasures: no doubt about it
            return d, f
        diff = np.frombuffer(rs_encode(d, nsym), np.uint8) != rx
        cost = float(np.sum(np.asarray(conf)[diff]))
        if best is None or cost < best[0]:
            best = (cost, d, f)
    if best is None:
        raise err
    return best[1], best[2]


# ============================================================================ modem
class Profile:
    """G groups of 2**K tones sound at once; each group carries K bits per symbol."""

    def __init__(self, name, f0, df, sym, guard, G, K, blocks, desc, chirp=None, emph=0.0):
        self.name, self.desc = name, desc
        self.G, self.K = G, K
        self.M = 1 << K                            # tones per group
        self.N = G * self.M
        self.freqs = f0 + df * np.arange(self.N)
        self.fa, self.fb = f0 - df * 0.8, self.freqs[-1] + df * 0.8
        # the sync chirp can use a narrower, cleaner slice than the data tones
        self.ca, self.cb = chirp or (self.fa, self.fb)
        self.Ns = int(sym * SR)
        self.Ng = int(guard * SR)
        self.Nt = int(0.004 * SR)                  # raised-cosine taper on each symbol
        self.Nw = self.Ns - self.Ng - self.Nt      # analysis window
        self.Lc = int(0.12 * SR)                   # chirp
        self.gap = int(0.03 * SR)
        self.bits = G * K                          # bits per symbol
        self.bps = self.bits / sym
        self.chunk = blocks * BLOCK - 4            # payload bytes per frame (CRC-32 fills the rest)
        n = np.arange(self.Nw)
        self.basis = (np.hanning(self.Nw)[:, None] *
                      np.exp(-2j * np.pi * self.freqs[None, :] * n[:, None] / SR))
        t = np.arange(self.Lc) / SR
        T = self.Lc / SR
        ch = np.sin(2 * np.pi * (self.ca * t + (self.cb - self.ca) * t * t / (2 * T)))
        self.chirp = (ch * tukey(self.Lc, 0.15)).astype(np.float32)
        ts = np.arange(self.Ns) / SR
        env = np.ones(self.Ns)
        rc = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, self.Nt))
        env[:self.Nt], env[-self.Nt:] = rc, rc[::-1]
        # pre-emphasis: tilt tone power toward the top, where small speakers roll off
        gain = (1 + (self.freqs / 9000) ** 2) ** (emph / 2)
        gain /= np.sqrt(np.mean(gain ** 2) * G)
        self.tones = [(np.sin(2 * np.pi * f * ts) * env * k).astype(np.float32)
                      for f, k in zip(self.freqs, gain)]
        self.view = (max(50, self.fa - 700), self.fb + 700)

    def symbol(self, vals):
        return sum(self.tones[g * self.M + int(v)] for g, v in enumerate(vals))

    def band_mask(self, nfft):
        f = np.fft.rfftfreq(nfft, 1 / SR)
        return ((f > self.ca - 250) & (f < self.cb + 250)).astype(np.float32)


def tukey(n, a):
    w = np.ones(n)
    k = int(a * n / 2)
    if k:
        r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k))
        w[:k], w[-k:] = r, r[::-1]
    return w


PROFILES = {   # name, f0, df, symbol s, guard s, groups, bits/group, RS blocks/frame
    "ultra":   Profile("ultra", 18200, 120, 0.040, 0.010, 4, 2, 3, "near-ultrasonic 18.2–20.0 kHz"),
    "audible": Profile("audible", 1000, 120, 0.035, 0.008, 4, 4, 8, "audible 1.0–8.6 kHz"),
    "low":     Profile("low", 300, 60, 0.070, 0.012, 2, 4, 2, "low 300–2200 Hz"),
    "all":     Profile("all", 300, 120, 0.035, 0.008, 10, 4, 12, "full band 0.3–19.4 kHz",
                       chirp=(1000, 8000), emph=0.75),
}


def n_syms(nbytes, prof):
    return -(-nbytes * 8 // prof.bits)


def to_syms(data, prof):
    """Bytes → one row of G tone indices (K bits each) per symbol, zero padded."""
    bits = np.unpackbits(np.frombuffer(bytes(data), np.uint8))
    bits = np.concatenate([bits, np.zeros(-len(bits) % prof.bits, np.uint8)])
    vals = bits.reshape(-1, prof.K).astype(int) @ (1 << np.arange(prof.K - 1, -1, -1))
    return vals.reshape(-1, prof.G)


def log_i0(x):
    """log of the modified Bessel function I0, without overflowing."""
    small = np.log(np.i0(np.minimum(x, 30)))
    xl = np.maximum(x, 30)
    return np.where(x < 30, small, xl - 0.5 * np.log(2 * np.pi * xl) + 1 / (8 * xl))


def bit_llrs(L, K):
    """Tone log-likelihoods (symbols, G, 2**K) → one LLR per bit in to_syms order; > 0 means a 1."""
    j = np.arange(1 << K)
    out = []
    for b in range(K):
        one = (j >> (K - 1 - b)) & 1 == 1
        out.append(np.logaddexp.reduce(L[..., one], -1) - np.logaddexp.reduce(L[..., ~one], -1))
    return np.clip(np.stack(out, -1), -LLR_CAP, LLR_CAP).ravel()


def hard_bytes(llr):
    """Bit LLRs → (bytes, each byte's confidence = its least certain bit)."""
    llr = np.asarray(llr).reshape(-1, 8)
    return np.packbits(llr > 0, axis=1).tobytes(), np.abs(llr).min(1)


def make_payload(text=None, path=None):
    if path:
        name = os.path.basename(path).encode()[:200]
        with open(path, "rb") as f:
            data = f.read()
        return b"F" + bytes([len(name)]) + name + data
    return b"T" + text.encode("utf-8")


def squeeze(data):
    """Stored, raw deflate or raw LZMA, whichever is smallest, behind a method byte."""
    z = zlib.compressobj(9, zlib.DEFLATED, -15)
    return min([b"\0" + data,
                b"\1" + z.compress(data) + z.flush(),
                b"\2" + lzma.compress(data, lzma.FORMAT_RAW, filters=LZMA_FILTERS)], key=len)


def unsqueeze(seg):
    """Inverse of squeeze → (data, intact).  Damaged data yields everything that decodes before the damage."""
    kind, body = seg[:1], seg[1:]
    if kind == b"\3":           # first frame: the payload's kind and file name, stored plain, then the rest
        n = body[0] if body else 0
        head, crc = body[1:1 + n], body[1 + n:3 + n]
        data, ok = unsqueeze(body[3 + n:])
        if crc != struct.pack(">H", zlib.crc32(head) & 0xFFFF):
            return b"", False   # a garbled name is worse than none
        return head + data, ok
    if kind == b"\0":
        return body, True
    if kind == b"\1":
        d = zlib.decompressobj(-15)
    elif kind == b"\2":
        d = lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    else:
        return b"", False
    out = []
    try:
        for i in range(0, len(body), 64):
            out.append(d.decompress(body[i:i + 64]))
            if d.eof:
                break
    except (zlib.error, lzma.LZMAError, EOFError):
        return b"".join(out), False
    return b"".join(out), d.eof


def fit(data, i, room, guess):
    """Roughly the longest stretch data[i:i+n] that squeezes into `room` bytes (secant steps on the ratio)."""
    left = len(data) - i
    best, n = 0, min(left, max(1, guess))
    for _ in range(8):
        size = len(squeeze(data[i:i + n]))
        if size <= room:
            best = max(best, n)
            if n == left or size > 0.97 * room:
                break
        n2 = min(left, max(1, int(n * 0.99 * room / size)))
        if n2 == n or n2 <= best:
            break
        n = n2
    return best or room - 1       # stored always fits room - 1 bytes


def new_message(payload, prof):
    """Returns (message id, frame chunks).  Each chunk is squeezed on its own, so a frame that never
    arrives costs only its own piece of the message, not everything after it.  The kind and file name go
    first and uncompressed, so even a garbled first frame still says what (and whose name) it was."""
    head = payload[:2 + payload[1]] if payload[:1] == b"F" else payload[:1]
    i, n = len(head), 3 * prof.chunk
    room = prof.chunk - 4 - len(head)
    n = fit(payload, i, room, n)
    seal = struct.pack(">H", zlib.crc32(head) & 0xFFFF)
    chunks = [b"\3" + bytes([len(head)]) + head + seal + squeeze(payload[i:i + n])]
    i += n
    while i < len(payload):
        n = fit(payload, i, prof.chunk, n)
        chunks.append(squeeze(payload[i:i + n]))
        i += n
    if len(chunks) > MAX_FRAMES:
        raise ValueError(f"too big: {len(chunks)} frames (max {MAX_FRAMES})")
    return int.from_bytes(os.urandom(2), "big"), chunks


def block_sizes(flen):
    return [min(BLOCK, flen - i) + NSYM for i in range(0, flen, BLOCK)]


def interleave_order(sizes):
    """On-air byte order: column by column across the RS blocks, so a burst is spread thin."""
    offs = np.cumsum([0] + sizes[:-1])
    return np.array([o + c for c in range(max(sizes)) for o, n in zip(offs, sizes) if c < n])


def scramble(mid, seq, rv, nbits, prof):
    """Each copy (rv) of a frame is XORed with its own key stream and rotated by its own amount, so a bit
    that landed on a dead tone, a weak tone group or an echo the first time lands somewhere else the
    next.  Returns (whitening bits, rotation)."""
    key = hashlib.shake_128(struct.pack(">HHB", mid, seq, rv)).digest(-(-nbits // 8))
    step = max(1, prof.G // 3) * prof.K                      # shift by whole tone groups, too
    rot = rv * ((nbits // 3) // prof.bits * prof.bits + step) % nbits
    return np.unpackbits(np.frombuffer(key, np.uint8))[:nbits], rot


def frame_bytes(mid, seq, total, chunk, prof, rv=0, flags=0):
    frame = chunk + struct.pack(">I", zlib.crc32(chunk))
    hdr = rs_encode(HDR.pack(mid, seq, total, len(frame), rv | flags, VERSION), HSYM)
    blocks = [rs_encode(frame[i:i + BLOCK], NSYM) for i in range(0, len(frame), BLOCK)]
    flat = np.frombuffer(b"".join(blocks), np.uint8)[interleave_order([len(b) for b in blocks])]
    key, rot = scramble(mid, seq, rv, len(flat) * 8, prof)
    return hdr, np.packbits(np.roll(np.unpackbits(flat), rot) ^ key).tobytes()


def unscramble(llr, mid, seq, rv, sizes, prof):
    """On-air bit LLRs of one copy → LLRs of the frame's RS codeword bits, the same for every copy."""
    nbits = sum(sizes) * 8
    key, rot = scramble(mid, seq, rv, nbits, prof)
    llr = np.roll(llr[:nbits] * (1 - 2.0 * key), -rot).reshape(-1, 8)
    out = np.empty_like(llr)
    out[interleave_order(sizes)] = llr
    return out.ravel()


def decode_frame(llr, flen):
    """Codeword bit LLRs → (best-guess chunk, CRC good, bytes repaired, per-block repairs or -1)."""
    sizes = block_sizes(flen)
    flat, conf = hard_bytes(llr)
    frame, fixed, ok, i, blocks = b"", 0, True, 0, []
    for size in sizes:
        blk = flat[i:i + size]
        try:
            d, f = rs_decode_soft(blk, conf[i:i + size], NSYM)
            fixed += f
            blocks.append(f)
        except RSError:
            d, ok = blk[:-NSYM], False
            blocks.append(-1)
        frame += d
        i += size
    chunk, crc = frame[:-4], frame[-4:]
    return chunk, ok and struct.pack(">I", zlib.crc32(chunk)) == crc, fixed, blocks


def modulate_frame(mid, seq, total, chunk, prof, rv=0, flags=0):
    hdr, body = frame_bytes(mid, seq, total, chunk, prof, rv, flags)
    train = np.repeat(np.arange(prof.M)[:, None], prof.G, 1)
    syms = np.concatenate([train, to_syms(hdr, prof), to_syms(body, prof)])
    parts = [np.zeros(int(0.05 * SR), np.float32), prof.chirp, np.zeros(prof.gap, np.float32)]
    parts += [prof.symbol(v) for v in syms]
    parts.append(np.zeros(int(0.1 * SR), np.float32))
    return np.concatenate(parts)


def frame_samples(n, prof):
    """Length in samples of modulate_frame() for an n-byte chunk."""
    syms = prof.M + n_syms(HDR_LEN, prof) + n_syms(sum(block_sizes(n + 4)), prof)
    return int(0.05 * SR) + prof.Lc + prof.gap + syms * prof.Ns + int(0.1 * SR)


def modulate(mid, chunks, prof, volume=0.5, seqs=None, rv=0, flags=0):
    """Render frames back to back: all of them, or just `seqs` for a resend (copy number rv)."""
    seqs = range(len(chunks)) if seqs is None else seqs
    w = np.concatenate([modulate_frame(mid, s, len(chunks), chunks[s], prof, rv, flags) for s in seqs])
    # many simultaneous tones make rare tall peaks; scale to the 99th percentile
    # and clip those few samples rather than turning the whole signal down
    peak = max(1.0, float(np.percentile(np.abs(w), CLIP_PCT)))
    return np.clip(w * volume / peak, -volume, volume).astype(np.float32)


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
        keep = max(keep, self.end - SR * 120)
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

    # -- tone training: learn timing, per-tone gain, noise and the room's echo -------
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
            # E[i, g, j]: energy of group g's tone j while every group plays its tone i
            E = np.array([self.energies(t0 + off + k * p.Ns) for k in range(p.M)]).reshape(p.M, p.G, p.M)
            diag = np.einsum("igi->gi", E)
            score = (diag / (E.sum(2).T + 1e-20)).mean()
            if best is None or score > best[0]:
                best = (score, off, E, diag)
        score, off, E, diag = best
        if score < 0.35:
            ev.append(("false", f"training failed ({score:.2f})"))
            return self.reset(self.sync + p.Lc // 2)
        self.a2 = np.maximum(diag, diag.mean() * 1e-3)     # (G, M) energy each tone arrives with
        self.amp = np.sqrt(self.a2)
        g = diag.ravel()
        off_diag = (E.sum() - g.sum()) / (p.G * p.M * (p.M - 1))
        self.snr = 10 * np.log10(g.mean() / (off_diag + 1e-20))
        # The sweep plays tone 0, 1, 2 … so each tone is silent before its turn (→ noise floor)
        # and ringing after it (→ how much of a tone the room still echoes d symbols later).
        Eg = E.transpose(1, 0, 2)                           # [g, symbol, tone]
        k, i = np.indices((p.M, p.M))
        self.noise = np.maximum(Eg[:, k < i].mean(1), 1e-4 * self.a2.mean(1))
        tail = (Eg - self.noise[:, None, None]) / self.a2[:, None, :]
        # per group, since rooms swallow treble faster than bass: R[g, d-1] = echo left d symbols on
        R = np.maximum(np.stack([tail[:, k - i == d].mean(1) for d in range(1, p.M)], 1), 1e-4)
        n = min(8, R.shape[1])
        slope = np.polyfit(np.arange(n), np.log(R[:, :n]).T, 1)[0] if n > 1 else np.full(p.G, np.log(0.3))
        self.rho = np.clip(R[:, 0], 0, 4)[:, None]          # (G, 1)
        self.q = np.clip(np.exp(slope), 0, 0.95)[:, None]
        self.B = np.zeros((p.G, p.M))                       # echo energy expected on each tone right now
        for j in range(p.M):
            self.B *= self.q
            self.B[:, j] += self.rho[:, 0] * self.a2[:, j]
        self.s = 1.0                                        # slow level drift since training
        self.tau = 0.0                                      # timing drift since training, in samples
        self.t0 = t0 + off + p.M * p.Ns
        self.state = "header"
        ev.append(("train", self.amp.ravel(), self.snr))
        return True

    def tones(self, k, known=None):
        """Log-likelihood of every tone of every group at symbol k, from a non-coherent (Rician) detector
        whose noise includes the echo the room is still ringing with from the tones already sent.  Then
        roll that echo forward with the tones decided here (or `known`, when we already know them)."""
        p, rows = self.p, np.arange(self.p.G)
        t = self.t0 + k * p.Ns + int(round(self.tau))
        e = self.energies(t).reshape(p.G, p.M)
        A2 = self.s * self.s * self.a2
        var = self.noise[:, None] + self.s * self.s * self.B
        L = log_i0(2 * np.sqrt(A2 * e) / var) - A2 / var
        j = L.argmax(1) if known is None else np.asarray(known)
        lvl = np.mean((e[rows, j] - var[rows, j]) / self.a2[rows, j])
        self.s = 0.85 * self.s + 0.15 * float(np.sqrt(np.clip(lvl, 0.04, 25)))
        # Early/late gate: the two sound cards' clocks never quite agree, and over a 20 s frame that adds
        # up to more than the guard time.  Slide the window toward whichever side keeps more of the tones.
        d, cols = p.Ng // 3, rows * p.M + j
        early = np.sum(np.abs(self.at(t + p.Ng - d, p.Nw) @ p.basis[:, cols]) ** 2)
        late = np.sum(np.abs(self.at(t + p.Ng + d, p.Nw) @ p.basis[:, cols]) ** 2)
        self.tau = float(np.clip(self.tau + TRACK_GAIN * d * (late - early) / (late + early + 1e-20),
                                 -p.Ns / 3, p.Ns / 3))
        self.B *= self.q
        self.B[rows, j] += self.rho[:, 0] * self.a2[rows, j]
        return L

    def st_header(self, ev):
        p = self.p
        n = n_syms(HDR_LEN, p)
        if self.end < self.t0 + (n + 1) * p.Ns:
            return False
        saved = self.B.copy(), self.s, self.tau
        L = np.array([self.tones(k) for k in range(n)])
        raw, c = hard_bytes(bit_llrs(L, p.K)[:HDR_LEN * 8])
        try:
            hdr, fixed = rs_decode_soft(raw, c, HSYM)
        except RSError:
            ev.append(("false", "header unreadable"))
            return self.reset(self.sync + p.Lc // 2)
        mid, seq, total, flen, flags, ver = HDR.unpack(hdr)
        if ver != VERSION or not 5 <= flen <= p.chunk + 4 or seq >= total:
            ev.append(("false", "header nonsense"))
            return self.reset(self.sync + p.Lc // 2)
        # now that we know exactly which tones the header used, redo the echo bookkeeping with them
        self.B, self.s, self.tau = saved
        for k, v in enumerate(to_syms(rs_encode(hdr, HSYM), p)):
            self.tones(k, v)
        self.mid, self.seq, self.total, self.flen = mid, seq, total, flen
        self.rv, self.ctl = flags & ~CTL, flags & CTL
        self.sizes = block_sizes(flen)
        self.k_body = n
        self.n_body = n_syms(sum(self.sizes), p)
        self.L = []
        self.state = "body"
        ev.append(("header", mid, seq, total, flen, self.rv, self.snr))
        return True

    def st_body(self, ev):
        p = self.p
        have = min(self.n_body, (self.end - self.t0) // p.Ns - self.k_body - 1)
        while len(self.L) < have:
            self.L.append(self.tones(self.k_body + len(self.L)))
        if len(self.L) < self.n_body:
            top = np.sort(np.array(self.L[-48:]), 2) if self.L else np.zeros((0, 1, 2))
            ev.append(("progress", len(self.L) / self.n_body, (top[..., -1] - top[..., -2]).min(1)))
            return False
        llr = unscramble(bit_llrs(np.array(self.L), p.K), self.mid, self.seq, self.rv, self.sizes, p)
        ev.append(("chunk", self.mid, self.seq, self.total, self.flen, self.rv, self.ctl, llr, self.snr))
        return self.reset(self.t0 + (self.k_body + self.n_body) * p.Ns)

    def reset(self, frm):
        self.state = "search"
        self.search_from = frm
        return True


class Assembler:
    """Collects frames by message id.  Every copy of a frame that fails its checks is kept as soft bits and
    added to the next copy before decoding again (Chase combining): two copies that are each too damaged
    to decode usually decode together, because their errors fall in different places."""

    def __init__(self):
        self.msgs = {}

    def add(self, mid, seq, total, flen, rv, ctl, llr, snr):
        """Returns a dict: good, fixed, blocks, copies, dup, and payload once the message is complete."""
        m = self.msgs.setdefault(mid, {"total": total, "ctl": ctl, "parts": {}, "soft": {}, "guess": {},
                                       "fixed": 0, "snr": snr, "t": time.time(), "naks": 0, "rv": 0,
                                       "gave_up": False, "done": False, "acked": -1})
        m["t"] = time.time()
        m["snr"] = min(m["snr"], snr)
        m["rv"] = max(m["rv"], rv)
        res = {"good": True, "fixed": 0, "blocks": [], "copies": 0, "dup": False, "payload": None}
        if total != m["total"] or seq in m["parts"]:
            res["dup"] = True
            return res
        old = m["soft"].get(seq)
        acc, n = (old[1] + llr, old[2] + 1) if old and old[0] == flen else (llr, 1)
        chunk, good, fixed, blocks = decode_frame(acc, flen)
        res.update(good=good, fixed=fixed, blocks=blocks, copies=n)
        if not good:
            m["soft"][seq] = (flen, acc, n)
            m["guess"][seq] = chunk     # best guess, in case we end up salvaging
            return res
        m["fixed"] += fixed
        m["parts"][seq] = chunk
        m["soft"].pop(seq, None)
        m["guess"].pop(seq, None)
        if len(m["parts"]) < total:
            return res
        pieces = [unsqueeze(m["parts"][i]) for i in range(total)]
        for i, (_, ok) in enumerate(pieces):
            if not ok:                  # slipped past its CRC; ask for it again
                del m["parts"][i]
        if len(m["parts"]) == total:
            m["payload"] = res["payload"] = b"".join(d for d, _ in pieces)
            m["done"] = True
        return res

    def salvage(self, mid):
        """Best effort from whatever arrived: good frames whole, garbled ones decoded as far as their damage
        allows, each gap marked.  Returns (kind, name, data) as parse_payload does."""
        m = self.msgs[mid]
        out, head = [], True
        for i in range(m["total"]):
            if i in m["parts"]:
                out.append(unsqueeze(m["parts"][i])[0])
                continue
            data, _ = unsqueeze(m["guess"][i]) if i in m["guess"] else (b"", False)
            head = head and (i > 0 or len(data) > 2)
            what = "damaged" if i in m["guess"] else "missing"
            out += [data, f" ⟪frame {i + 1}/{m['total']} {what}⟫ ".encode()]
        blob = b"".join(out)
        if head and blob[:1] in (b"T", b"F"):
            return parse_payload(blob)
        return "file", f"message-{mid:04x}.bin", blob

    def missing(self, mid):
        m = self.msgs[mid]
        return [i for i in range(m["total"]) if i not in m["parts"]]


def parse_payload(payload):
    if payload[:1] == b"F" and len(payload) >= 2:
        n = payload[1]
        return "file", payload[2:2 + n].decode("utf-8", "replace"), payload[2 + n:]
    if payload[:1] in (b"A", b"N", b"G"):     # ack / resend request / gave up
        return "ctl", payload[:1].decode(), payload[1:]
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
    TX_STAGES = ["zip", "frame", "RS", "FSK", "on air", "ack"]
    RX_STAGES = ["listen", "sync", "train", "header", "body", "RS", "CRC", "join"]

    def __init__(self, scr, args):
        self.scr = scr
        self.args = args
        self.volume = args.volume
        self.set_profile(args.profile)
        self.q = queue.Queue()
        self.tx = None
        self.tx_info = None          # which message/frames the speaker is playing
        self.txq = deque()
        self.tx_pos = 0
        self.tx_len = 1
        self.outbox = {}             # our messages awaiting an ack, by id
        self.replies = deque()       # (not before, mid, chunks, seqs, rv, flags): replies held for the other side
        self.asm = Assembler()
        self.quiet_t = time.time()   # last time either side was on air
        self.batch = {}              # message id -> frame seqs the sender is playing this round
        self.wait = {}               # message id -> seconds of those frames still to come
        self.log = deque(maxlen=200)
        self.line = ""
        self.level = 0.0
        self.mic_abs = 0
        self.tail = np.zeros(4096, np.float32)
        self.rows = deque(maxlen=200)
        self.floor = None
        self.txs = None              # what the transmit panel shows
        self.ctl_note = ""
        self.rxs = {"stage": "listen", "corr": 0.0, "amp": None, "snr": None, "mid": None, "seq": 0,
                    "total": 0, "flen": 0, "prog": 0.0, "conf": [], "blocks": [], "bad": set(),
                    "note": "", "t": 0.0}
        self.colors()
        self.add("✦", f"airgap ready · {self.p.desc} · {self.p.bps:.0f} bit/s raw · "
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
        self.c256 = c = curses.COLORS >= 256
        heat = HEAT if c else HEAT8
        for i, h in enumerate(heat):
            curses.init_pair(10 + i, 0, h)
        self.nheat = len(heat)
        fg = {1: (51, 6),     # them / receive
              2: (213, 5),    # you / transmit
              3: (196, 1),    # error
              4: (245, 7),    # dim
              5: (226, 3),    # markers
              6: (46, 2),     # good
              7: (60, 4),     # panel border
              8: (214, 3)}    # resend / warning
        for k, (a, b) in fg.items():
            curses.init_pair(k, a if c else b, bg)
        curses.init_pair(30, 231 if c else 7, 54 if c else 5)    # title bar
        curses.init_pair(31, 16 if c else 0, 51 if c else 6)     # active receive stage
        curses.init_pair(32, 16 if c else 0, 213 if c else 5)    # active transmit stage

    def add(self, who, text, cp):
        # garbled frames decode to arbitrary bytes; curses chokes on control characters
        text = "".join(ch if ch.isprintable() else "·" for ch in text)
        self.log.append((time.strftime("%H:%M:%S"), who, text, cp))

    # -- audio ------------------------------------------------------------------
    def mic_cb(self, indata, frames, ti, status):
        self.q.put(indata[:, 0].copy())

    def spk_cb(self, out, frames, ti, status):
        out.fill(0)
        if self.tx is None and self.txq:
            w, info = self.txq.popleft()
            self.tx_pos, self.tx_len, self.tx_info = 0, len(w), info
            self.tx = w
        tx = self.tx
        if tx is not None:
            n = min(frames, len(tx) - self.tx_pos)
            out[:n, 0] = tx[self.tx_pos:self.tx_pos + n]
            self.tx_pos += n
            if self.tx_pos >= len(tx):
                self.tx = None
        if out.shape[1] > 1:
            out[:, 1] = out[:, 0]

    def send(self, payload, label=None):
        try:
            mid, chunks = new_message(payload, self.p)
        except ValueError as e:
            self.add("!", str(e), 3)
            return
        if label:   # a real message, not an ack: keep it so lost frames can be resent
            self.outbox[mid] = {"chunks": chunks, "label": label, "tries": 0}
            self.txs = {"label": label, "mid": mid, "raw": len(payload), "packed": sum(map(len, chunks)),
                        "method": "+".join(sorted({("none", "deflate", "lzma")[c[4 + c[1]] if c[0] == 3 else c[0]]
                                                    for c in chunks})),
                        "total": len(chunks), "sent": set(), "lost": set(), "t0": time.time(), "acked": None,
                        "resends": 0, "gave_up": None}
        else:
            self.ctl_note = {b"A": "↯ sending ack", b"N": "↯ asking for missing frames",
                             b"G": "↯ telling them we gave up"}.get(payload[:1], "")
            self.replies.append((time.time() + REPLY_GAP, mid, chunks, None, 0, CTL))
            return
        w = self.play(mid, chunks)
        if label:
            n = len(chunks)
            self.add("▸", f"{label}   ({n} frame{'s' if n != 1 else ''} · {len(w) / SR:.1f} s on air)", 2)

    def play(self, mid, chunks, seqs=None, rv=0, flags=0):
        seqs = list(range(len(chunks)) if seqs is None else seqs)
        w = modulate(mid, chunks, self.p, self.volume, seqs, rv, flags)
        bounds = np.cumsum([frame_samples(len(chunks[s]), self.p) for s in seqs])
        self.txq.append((w, {"mid": mid, "seqs": seqs, "bounds": bounds}))
        if not self.args.hear_self:
            queued = sum(len(x) for x, _ in self.txq) + (self.tx_len - self.tx_pos if self.tx is not None else 0)
            self.rx.ignore_until = max(self.rx.ignore_until, self.mic_abs + queued + SR // 2)
        return w

    def finish(self, mid, why=None):
        """Forget an outgoing message: delivered (why=None) or abandoned."""
        out = self.outbox.pop(mid)
        if why:
            self.add("✗", f"gave up ({why}): {out['label']}", 3)
        else:
            self.add("✓", f"delivered: {out['label']}", 6)
        if self.txs and self.txs["mid"] == mid:
            self.txs["gave_up" if why else "acked"] = why or time.time()

    def control(self, kind, data):
        mid = struct.unpack(">H", data[:2])[0]
        if mid not in self.outbox:
            return
        if kind == "A":
            self.finish(mid)
        elif kind == "G":
            self.finish(mid, "they kept a partial copy")
        else:
            seqs = [s for (s,) in struct.iter_unpack(">H", data[2:len(data) & ~1])
                    if s < len(self.outbox[mid]["chunks"])]
            self.resend(mid, seqs)

    def resend(self, mid, seqs=None):
        """Play another copy of the frames they asked for (all of them, if they never answered).  Each
        copy is scrambled differently, so it fails in different places and the two combine."""
        out = self.outbox[mid]
        if out["tries"] >= MAX_RESENDS:
            self.finish(mid, f"still not through after {MAX_RESENDS} resends" if seqs
                        else "no answer from the other side")
            return
        out["tries"] += 1
        seqs = seqs or list(range(len(out["chunks"])))
        if self.txs and self.txs["mid"] == mid:
            self.txs["resends"] += 1
            self.txs["lost"] = set(seqs)
            self.txs["sent"] -= set(seqs)
        self.add("↻", f"resend {out['tries']}/{MAX_RESENDS}: {len(seqs)} of {len(out['chunks'])} frames: "
                      f"{out['label']}", 8)
        self.replies.append((time.time() + REPLY_GAP, mid, out["chunks"], seqs, out["tries"], 0))

    def arq(self):
        """Ask for missing frames, and resend messages that were never acknowledged."""
        now = time.time()
        info = self.tx_info
        if info and self.tx is None:     # speaker finished a batch
            if self.txs and info["mid"] == self.txs["mid"]:
                self.txs["sent"].update(info["seqs"])
                self.txs["lost"] -= set(info["seqs"])
            self.tx_info = None
            self.ctl_note = ""
        if self.tx is not None or self.txq or self.rx.state != "search":
            self.quiet_t = now
        if self.replies and self.replies[0][0] <= now and self.rx.state == "search":
            self.play(*self.replies.popleft()[1:])
        if self.args.hear_self:
            return
        for mid, m in self.asm.msgs.items():
            # don't ask while the sender is still playing frames we failed to lock on to:
            # it isn't listening, and the request would be lost
            if m["gave_up"] or now - max(m["t"], self.quiet_t) < NAK_WAIT + self.wait.get(mid, 0):
                continue
            if m["done"]:
                if m["acked"] < m["rv"]:     # they resent a message we already have: our ack got lost
                    m["acked"] = m["rv"]
                    self.send(b"A" + struct.pack(">H", mid))
                continue
            miss = self.asm.missing(mid)
            if m["ctl"]:                 # a garbled ack or request: never resent, nothing to salvage
                m["gave_up"] = True
                continue
            if m["rv"] >= MAX_RESENDS or m["naks"] >= MAX_RESENDS:   # that was the sender's last copy
                m["gave_up"] = True
                self.add("✗", f"gave up after {MAX_RESENDS} resends: {len(miss)} of {m['total']} frames still "
                              f"bad — try --profile audible, less distance, or a less echoey spot", 3)
                self.salvage(mid)
                self.send(b"G" + struct.pack(">H", mid))
                return
            m["naks"] += 1
            m["t"] = now
            self.add("↻", f"missing {len(miss)} of {m['total']} frames — asking for resend "
                          f"{m['naks']}/{MAX_RESENDS} (keeping what we have to combine with it)", 8)
            self.batch[mid] = miss[:256]
            self.wait[mid] = self.frames_left(mid, None)
            self.send(b"N" + struct.pack(">H", mid) + b"".join(struct.pack(">H", s) for s in miss[:256]))
            return
        if self.outbox and now - self.quiet_t > ACK_WAIT:
            for mid in list(self.outbox):
                self.resend(mid)
            self.quiet_t = now

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
            self.add("!", "commands: /file PATH · /vol 0.4 · /profile ultra|audible|low|all · /quit", 3)
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
        if got:
            self.hear(np.concatenate(got))
        self.arq()

    def hear(self, x):
        self.mic_abs += len(x)
        self.level = 0.8 * self.level + 0.2 * float(np.sqrt(np.mean(x * x)))
        self.tail = np.concatenate([self.tail, x])[-len(self.tail):]
        for e in self.rx.feed(x):
            self.event(e)

    def event(self, e):
        kind, r = e[0], self.rxs
        r["t"] = time.time()
        if kind == "sync":
            r.update(stage="sync", note=f"chirp locked (corr {e[1]:.2f})", prog=0.0, conf=[], blocks=[])
        elif kind == "train":
            r.update(stage="train", amp=e[1], snr=e[2], note="learned the room's tone gains and echo")
        elif kind == "false":
            r.update(stage="listen", note=e[1])
        elif kind == "header":
            _, mid, seq, total, flen, rv, snr = e
            self.wait[mid] = self.frames_left(mid, seq, total)
            if mid != r["mid"]:
                r["bad"] = set()
            r.update(stage="body", mid=mid, seq=seq, total=total, flen=flen,
                     note=f"header ok · copy {rv + 1}" if rv else "header ok")
        elif kind == "progress":
            r["prog"], r["conf"] = e[1], e[2]
        elif kind == "chunk":
            _, mid, seq, total, flen, rv, ctl, llr, snr = e
            res = self.asm.add(mid, seq, total, flen, rv, ctl, llr, snr)
            m = self.asm.msgs[mid]
            self.wait[mid] = self.frames_left(mid, seq, total)
            if res["dup"]:                  # arq() re-acks once the resend is over
                return
            r.update(stage="join" if res["good"] else "CRC", blocks=res["blocks"], prog=1.0)
            n = res["copies"]
            if not res["good"]:
                r["bad"].add(seq)
                r["note"] = "checksum failed"
                if not ctl:
                    both = f" — even with {n} copies combined" if n > 1 else " — kept it to combine with a resend"
                    self.add("✗", f"frame {seq + 1}/{total} garbled{both}", 3)
                return
            r["bad"].discard(seq)
            r["note"] = f"frame {seq + 1}/{total} ok"
            if n > 1:
                self.add("✓", f"frame {seq + 1}/{total} recovered by combining {n} garbled copies", 6)
            if res["payload"] is not None:
                r["note"] = "message complete"
                m["acked"] = m["rv"]
                self.batch.pop(mid, None)   # any further copies would be whole resends
                self.deliver(mid, res["payload"])

    def frames_left(self, mid, seq, total=None):
        """Seconds of the sender's current round still to play after frame `seq`."""
        batch = self.batch.get(mid) or list(range(total or self.asm.msgs[mid]["total"]))
        after = batch[batch.index(seq) + 1:] if seq in batch else batch
        return len(after) * frame_samples(self.p.chunk, self.p) / SR

    def salvage(self, mid):
        m = self.asm.msgs[mid]
        typ, name, data = self.asm.salvage(mid)
        bad = f"{len(m['guess'])} garbled, {len(self.asm.missing(mid)) - len(m['guess'])} lost"
        if typ == "file":
            path = save_file(name + ".partial", data)
            self.add("⚠", f"partial file {name} ({len(data):,} bytes, {bad} of {m['total']} frames) → {path}", 8)
        else:
            self.add("⚠", f"partial message ({bad} of {m['total']} frames): {data[:400]}", 8)

    def deliver(self, mid, payload):
        m = self.asm.msgs[mid]
        note = f"   (RS fixed {m['fixed']} byte{'s' if m['fixed'] != 1 else ''} · SNR {m['snr']:.0f} dB)"
        if m["gave_up"]:
            note += " — recovered after all"
        typ, name, data = parse_payload(payload)
        if typ == "ctl":
            self.control(name, data)
            return
        if typ == "file":
            path = save_file(name, data)
            self.add("◂", f"file {name} ({len(data)} bytes) → {path}{note}", 1)
        else:
            shown = data if len(data) <= 400 else f"{data[:400]}… ({len(data):,} chars)"
            self.add("◂", shown + note, 1)
        if not self.args.hear_self:
            self.send(b"A" + struct.pack(">H", mid))

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

    def put(self, y, x, s, a=0):
        H, Wd = self.dims
        if 0 <= y < H and 0 <= x < Wd:
            try:
                self.scr.addstr(y, x, s[: Wd - x], a)
            except (curses.error, ValueError):
                pass

    @staticmethod
    def bar(frac, w):
        n = int(max(0.0, min(1.0, frac)) * w * 8)
        s = "█" * (n // 8) + ("▏▎▍▌▋▊▉"[n % 8 - 1] if n % 8 else "")
        return s.ljust(w)

    @staticmethod
    def spark(vals, w):
        v = np.asarray(vals, float)[-w:]
        if not len(v):
            return ""
        lo, hi = v.min(), v.max()
        v = (v - lo) / (hi - lo) if hi > lo else np.full(len(v), 0.5)
        return "".join("▁▂▃▄▅▆▇█"[int(x * 7.99)] for x in v)

    def box(self, y, x, h, w, title, cp):
        a = curses.color_pair(7)
        self.put(y, x, "╭" + "─" * (w - 2) + "╮", a)
        self.put(y, x + 2, f" {title} ", curses.color_pair(cp) | curses.A_BOLD)
        for i in range(1, h - 1):
            self.put(y + i, x, "│", a)
            self.put(y + i, x + w - 1, "│", a)
        self.put(y + h - 1, x, "╰" + "─" * (w - 2) + "╯", a)

    def chain(self, y, x, w, stages, states, active_cp):
        """states: per stage one of 'done', 'active', 'pending'."""
        names = stages if sum(len(s) + 4 for s in stages) - 3 <= w else [s[:4] for s in stages]
        spin = "◐◓◑◒"[int(time.time() * 6) % 4]
        for i, (name, st) in enumerate(zip(names, states)):
            if i:
                self.put(y, x, " › ", curses.color_pair(7))
                x += 3
            if st == "done":
                tok, a = f"✓{name}", curses.color_pair(6)
            elif st == "active":
                tok, a = f"{spin}{name}", curses.color_pair(active_cp) | curses.A_BOLD
            else:
                tok, a = f"·{name}", curses.color_pair(4)
            self.put(y, x, tok, a)
            x += len(tok)

    def label(self, y, x, name, cp=4):
        self.put(y, x, f"{name:<9}", curses.color_pair(cp))
        return x + 9

    def tx_panel(self, y, x, h, w):
        self.box(y, x, h, w, "▸ TRANSMIT", 2)
        t, ix, iw = self.txs, x + 2, w - 4
        if t is None:
            self.chain(y + 1, ix, iw, self.TX_STAGES, ["pending"] * 6, 32)
            self.put(y + 3, ix, "type a message, or /file PATH", curses.color_pair(4))
            if self.ctl_note:
                self.put(y + h - 2, ix, self.ctl_note, curses.color_pair(8) | curses.A_BOLD)
            return
        info = self.tx_info if self.tx is not None else None
        ours = info is not None and info["mid"] == t["mid"]
        cur = None
        if ours:
            k = min(int(np.searchsorted(info["bounds"], self.tx_pos, "right")), len(info["seqs"]) - 1)
            cur = info["seqs"][k]
            t["sent"].update(info["seqs"][:k])
        all_sent = len(t["sent"]) == t["total"]
        air = "active" if ours or self.txq else ("done" if all_sent else "pending")
        ack = "done" if t["acked"] else ("active" if all_sent and not t["gave_up"] else "pending")
        self.chain(y + 1, ix, iw, self.TX_STAGES, ["done"] * 4 + [air, ack], 32)
        ratio = t["raw"] / max(1, t["packed"])
        c = self.label(y + 2, ix, "payload")
        self.put(y + 2, c, f"{t['raw']:,} B ─{t['method']}→ {t['packed']:,} B  (×{ratio:.1f})",
                 curses.color_pair(2))
        c = self.label(y + 3, ix, "framing")
        self.put(y + 3, c, f"{t['total']} frame{'s' if t['total'] != 1 else ''} · RS(255,{BLOCK}) "
                 f"interleaved · CRC-32 · {self.p.G}×{self.p.M}-FSK", curses.color_pair(4))
        c = self.label(y + 4, ix, "on air")
        bw = max(8, iw - 9 - 30)
        if ours:
            fr = self.tx_pos / max(1, self.tx_len)
            eta = (self.tx_len - self.tx_pos) / SR
            self.put(y + 4, c, f"▕{self.bar(fr, bw)}▏ {fr * 100:3.0f}%  frame {cur + 1}  "
                     f"eta {int(eta) // 60}:{int(eta) % 60:02d}", curses.color_pair(2) | curses.A_BOLD)
        elif t["acked"]:
            d = t["acked"] - t["t0"]
            self.put(y + 4, c, f"✓ delivered in {int(d) // 60}:{int(d) % 60:02d}", curses.color_pair(6) | curses.A_BOLD)
        elif t["gave_up"]:
            self.put(y + 4, c, f"✗ gave up — {t['gave_up']}", curses.color_pair(3) | curses.A_BOLD)
        elif all_sent:
            out = self.outbox.get(t["mid"], {"tries": MAX_RESENDS})
            left = max(0.0, ACK_WAIT - (time.time() - max(t["t0"], self.quiet_t)))
            if out["tries"] < MAX_RESENDS:
                nxt = f"resend {out['tries'] + 1}/{MAX_RESENDS} in {left:2.0f}s"
            else:
                nxt = f"giving up in {left:2.0f}s"
            self.put(y + 4, c, f"waiting for ack… {nxt}  ▕{self.bar(left / ACK_WAIT, 12)}▏",
                     curses.color_pair(8) | curses.A_BOLD)
        else:
            self.put(y + 4, c, "queued", curses.color_pair(4))
        c = self.label(y + 5, ix, "frames")
        self.grid(y + 5, c, iw - 9, t["total"], lambda s: (
            ("▶", curses.color_pair(2) | curses.A_BOLD) if s == cur else
            ("↻", curses.color_pair(8)) if s in t["lost"] else
            ("■", curses.color_pair(6) if t["acked"] else curses.color_pair(2)) if s in t["sent"] else
            ("□", curses.color_pair(4))))
        extra = f"↻ resent {t['resends']}×   " if t["resends"] else ""
        self.put(y + h - 2, ix, (extra + self.ctl_note)[:iw], curses.color_pair(8) | curses.A_BOLD)
        self.put(y + h - 2, ix + len(extra + self.ctl_note) + 1, t["label"][:max(0, iw - len(extra + self.ctl_note) - 1)],
                 curses.color_pair(4))

    def grid(self, y, x, w, total, cell):
        if total <= 0:
            return
        if total <= w:
            for s in range(total):
                ch, a = cell(s)
                self.put(y, x + s, ch, a)
        else:   # too many frames: each cell shows its worst frame
            rank = {"▶": 0, "✗": 1, "↻": 2, "▣": 3, "□": 4, "■": 5}
            for i in range(w):
                lo, hi = i * total // w, max(i * total // w + 1, (i + 1) * total // w)
                ch, a = min((cell(s) for s in range(lo, hi)), key=lambda z: rank.get(z[0], 9))
                self.put(y, x + i, ch, a)

    def rx_panel(self, y, x, h, w):
        r, ix, iw = self.rxs, x + 2, w - 4
        self.box(y, x, h, w, "◂ RECEIVE", 1)
        k = self.RX_STAGES.index(r["stage"])
        if r["stage"] == "listen" or time.time() - r["t"] > 3 and self.rx.state == "search":
            states = ["active"] + ["pending"] * (len(self.RX_STAGES) - 1)
        else:
            states = ["done"] * k + ["active"] + ["pending"] * (len(self.RX_STAGES) - k - 1)
            if r["stage"] == "CRC":
                states[k] = "pending"
        self.chain(y + 1, ix, iw, self.RX_STAGES, states, 31)
        bw = max(8, min(24, iw - 9 - 16))
        c = self.label(y + 2, ix, "chirp")
        corr = self.rx.last_corr
        b = list(self.bar(corr, bw))
        b[min(bw - 1, int(Receiver.THRESH * bw))] = "┊"
        self.put(y + 2, c, f"▕{''.join(b)}▏ {corr:.2f} / {Receiver.THRESH}",
                 curses.color_pair(6 if corr > Receiver.THRESH else 1))
        c = self.label(y + 3, ix, "channel")
        if r["amp"] is not None:
            snr = f"SNR {r['snr']:.0f} dB  "
            self.put(y + 3, c, snr, curses.color_pair(1) | curses.A_BOLD)
            self.put(y + 3, c + len(snr), "gains " + self.spark(20 * np.log10(r["amp"] + 1e-9), iw - 9 - len(snr) - 6),
                     curses.color_pair(5))
        else:
            self.put(y + 3, c, "—", curses.color_pair(4))
        c = self.label(y + 4, ix, "frame")
        if r["mid"] is not None:
            self.put(y + 4, c, f"msg {r['mid']:04x} · frame {r['seq'] + 1}/{r['total']} · {r['flen']:,} B",
                     curses.color_pair(1))
        c = self.label(y + 5, ix, "body")
        active = self.rx.state == "body"
        if r["mid"] is not None:
            self.put(y + 5, c, f"▕{self.bar(r['prog'], bw)}▏ {r['prog'] * 100:3.0f}%",
                     curses.color_pair(1) | (curses.A_BOLD if active else 0))
            if len(r["conf"]):
                self.put(y + 5, c + bw + 8, "conf " + self.spark(r["conf"], iw - 9 - bw - 13), curses.color_pair(5))
        c = self.label(y + 6, ix, "RS")
        for i, f in enumerate(r["blocks"][: iw - 30]):
            ch, a = ("✗", 3) if f < 0 else ("✓", 6) if f == 0 else ("◆", 5)
            self.put(y + 6, c + i, ch, curses.color_pair(a))
        if r["blocks"]:
            fixed = sum(f for f in r["blocks"] if f > 0)
            self.put(y + 6, c + len(r["blocks"]) + 2, f"repaired {fixed} byte{'s' if fixed != 1 else ''}",
                     curses.color_pair(4))
        c = self.label(y + 7, ix, "message")
        m = self.asm.msgs.get(r["mid"]) if r["mid"] is not None else None
        if m:
            self.grid(y + 7, c, iw - 9 - 14, m["total"], lambda s: (
                ("▣", curses.color_pair(1) | curses.A_BOLD) if s == r["seq"] and active else
                ("■", curses.color_pair(6)) if s in m["parts"] else
                ("✗", curses.color_pair(3)) if s in r["bad"] else
                ("□", curses.color_pair(4))))
            self.put(y + 7, c + min(m["total"], iw - 9 - 14) + 1, f"{len(m['parts'])}/{m['total']}",
                     curses.color_pair(6 if m["done"] else 4))
        if r["note"]:
            self.put(y + h - 2, ix, r["note"][:iw], curses.color_pair(4))

    def draw(self):
        scr = self.scr
        H, W = scr.getmaxyx()
        Wd = W - 1
        self.dims = (H, Wd)
        scr.erase()
        put = self.put

        # title bar
        meter = int(min(1, self.level * 40) * 10)
        on = self.tx is not None
        dot = "◉" if not on or int(time.time() * 4) % 2 else "○"
        left = f" {dot} A I R G A P "
        mid = f" {self.p.desc} · {self.p.G}×{self.p.M}-FSK · {self.p.bps:.0f} bit/s raw "
        right = f" {'ON AIR' if on else 'listening'} · vol {self.volume:.2f} · mic {'▮' * meter}{'▯' * (10 - meter)} "
        title = curses.color_pair(30)
        put(0, 0, " " * Wd, title)
        put(0, 0, left, title | curses.A_BOLD)
        put(0, len(left), mid, title)
        put(0, max(len(left) + len(mid), Wd - len(right)), right, title | (curses.A_BOLD if on else 0))

        side = Wd >= 100
        ph = 10
        panels = ph if side else 2 * ph
        wf_h = max(3, min(12, H - 1 - 2 - panels - 6 - 1))

        # waterfall
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
        # axis with tone markers, coloured by group
        ay = 1 + wf_h
        axis = [" "] * Wd
        lo, hi = self.p.view
        step = 200 if hi - lo < 2500 else 500 if hi - lo < 6000 else 1000
        f = (int(lo) // step + 1) * step
        while f < hi:
            c = self.freq_col(f, Wd)
            lab = f"{f / 1000:.1f}k" if f >= 1000 else f"{f}"
            for i, ch in enumerate("┴" + lab):
                if 0 <= c + i < Wd:
                    axis[c + i] = ch
            f += step
        put(ay + 1, 0, "".join(axis), curses.color_pair(4))
        gcol = [5, 1, 2, 8]     # one colour per tone group
        for i, t in enumerate(self.p.freqs):
            put(ay, self.freq_col(t, Wd), "▾", curses.color_pair(gcol[i // self.p.M % 4]))

        # pipeline panels
        py = ay + 2
        if side:
            pw = Wd // 2
            self.tx_panel(py, 0, ph, pw)
            self.rx_panel(py, pw, ph, Wd - pw)
        else:
            self.tx_panel(py, 0, ph, Wd)
            self.rx_panel(py + ph, 0, ph, Wd)

        # log
        top = py + panels
        avail = H - top - 1
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
        prompt = " ❯ "
        put(H - 1, 0, (prompt + self.line)[-(Wd - 1):].ljust(Wd), curses.color_pair(2) | curses.A_BOLD)
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
    mid, chunks = new_message(payload, prof)
    w = modulate(mid, chunks, prof, args.volume)
    dur = len(w) / SR
    print(f"◉ {prof.desc} · {len(payload)} bytes → {sum(map(len, chunks))} compressed · "
          f"{len(chunks)} frames · {dur:.1f} s on air")
    sd.play(w, SR, device=args.output)
    t0 = time.time()
    while time.time() - t0 < dur:
        fr = (time.time() - t0) / dur
        print(f"\r  ▕{'█' * int(fr * 40)}{'░' * (40 - int(fr * 40))}▏ {fr * 100:3.0f}%", end="", flush=True)
        time.sleep(0.05)
    sd.wait()
    print(f"\r  ▕{'█' * 40}▏ 100%  sent.")


def decode_signal(x, prof, chunk=2400, asm=None):
    rx, asm = Receiver(prof), asm or Assembler()
    for i in range(0, len(x), chunk):
        for e in rx.feed(x[i:i + chunk]):
            if e[0] == "chunk":
                asm.add(*e[1:])
    for e in rx.feed(np.zeros(SR, np.float32)):
        if e[0] == "chunk":
            asm.add(*e[1:])
    return asm


def report(asm, prof):
    if not asm.msgs:
        print("  nothing decoded.")
    for mid, m in asm.msgs.items():
        note = f"RS fixed {m['fixed']}, SNR {m['snr']:.0f} dB"
        if "payload" not in m:
            print(f"  ✗ incomplete: frames {[s + 1 for s in asm.missing(mid)]} of {m['total']} bad   {note}")
            typ, name, data = asm.salvage(mid)
            if typ == "file":
                print(f"  ⚠ partial file {name} ({len(data):,} bytes) → {save_file(name + '.partial', data)}")
            elif typ == "text":
                print(f"  ⚠ partial text: “{data}”")
            continue
        typ, name, data = parse_payload(m["payload"])
        if typ == "file":
            print(f"  ✓ file {name} ({len(data)} bytes) → {save_file(name, data)}   {note}")
        elif typ == "text":
            print(f"  ✓ “{data}”   {note}")


def channel(w, rng, gain=0.08, noise=0.004, drift=40e-6, echo=True, burst=True, tilt=True, reverb=None):
    """A cruel simulated room: lead-in, echoes, tilt, noise, a clap, clock drift.  reverb=(RT60 seconds,
    direct-to-reverberant ratio in dB) adds a diffuse tail; 0 dB is roughly a living room at 1–2 m."""
    x = np.concatenate([np.zeros(int(rng.uniform(0.2, 1.0) * SR)), w, np.zeros(SR // 2)])
    if echo:
        h = np.zeros(int(0.009 * SR))
        h[0], h[int(0.003 * SR)], h[-1] = 1.0, 0.45, 0.25
        x = np.convolve(x, h)[:len(x)]
    if reverb:
        rt60, drr = reverb
        t = np.arange(int(rt60 * SR)) / SR
        h = rng.normal(0, 1, len(t)) * 10 ** (-3 * t / rt60)
        h[:int(0.003 * SR)] = 0
        h *= 10 ** (-drr / 20) / np.sqrt(np.sum(h * h))
        h[0] += 1
        n = 1 << int(np.ceil(np.log2(len(x) + len(h))))
        x = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(h, n), n)[:len(x)]
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
        e = int(rng.integers(0, NSYM + 1))               # erasures …
        k = int(rng.integers(0, (NSYM - e) // 2 + 1))    # … plus errors, within budget
        pos = rng.choice(len(enc), e + k, replace=False)
        for p in pos:
            enc[p] ^= int(rng.integers(1, 256))
        dec, fixed = rs_decode(bytes(enc), NSYM, sorted(int(p) for p in pos[:e]))
        assert dec == msg and fixed == e + k, (trial, e, k, fixed)
    print("200 random blocks with errors + erasures up to the limit — all repaired ✓")
    text = "The quick brown fox whispers across the air gap at 19 kHz. 🦊"
    for name in (args.profile_list or list(PROFILES)):
        prof = PROFILES[name]
        mid, chunks = new_message(make_payload(text=text), prof)
        w = modulate(mid, chunks, prof, 0.5)
        x = channel(w, rng)
        t0 = time.time()
        asm = decode_signal(x, prof)
        dt = time.time() - t0
        ok = asm.msgs.get(mid, {}).get("payload") == make_payload(text=text)
        print(f"{name:>8}: {prof.bps:4.0f} bit/s raw, {len(w) / SR:4.1f} s of audio, "
              f"decoded in {dt:.2f} s →", "✓" if ok else "✗")
        if not ok:
            report(asm, prof)
    # a bigger message over several frames, one of which is lost and resent
    prof = PROFILES["audible"]
    lines = [f"{i:04d} {' '.join(rng.choice(['air', 'gap', 'fox', 'tone', 'chirp', 'echo'], 6))}"
             for i in range(600)]
    payload = make_payload(text="\n".join(lines)) + bytes(rng.integers(0, 256, 1500).tolist())
    mid, chunks = new_message(payload, prof)
    w = modulate(mid, chunks, prof, 0.5)
    frame = len(w) // len(chunks)
    w[frame + SR: frame + 6 * SR] = 0                  # a 5 s dropout wipes out frame 2
    asm = decode_signal(channel(w, rng), prof)
    miss = asm.missing(mid) if mid in asm.msgs else list(range(len(chunks)))
    if miss:
        asm = decode_signal(channel(modulate(mid, chunks, prof, 0.5, miss, rv=1), rng), prof, asm=asm)
    ok = asm.msgs.get(mid, {}).get("payload") == payload
    print(f"   multi: {len(payload)} bytes → {sum(map(len, chunks))} on air in {len(chunks)} frames "
          f"({len(w) / SR:.1f} s); lost frames {[s + 1 for s in miss]} resent →", "✓" if ok else "✗")
    # an echoey room where no single copy gets through: copies are scrambled differently and combined
    prof = PROFILES["all"]
    text = "\n".join(lines + [f"{i:04d} {' '.join(rng.choice(['bit', 'byte', 'hum', 'buzz'], 5))}"
                              for i in range(600, 1200)])
    mid, chunks = new_message(make_payload(text=text), prof)
    asm, log = Assembler(), []
    for rv in range(MAX_RESENDS + 1):
        miss = asm.missing(mid) if mid in asm.msgs else list(range(len(chunks)))
        if not miss:
            break
        x = channel(modulate(mid, chunks, prof, 0.5, miss, rv=rv), rng, reverb=(0.4, -6), burst=False)
        rx = Receiver(prof)
        for i in range(0, len(x), 4800):
            for e in rx.feed(x[i:i + 4800]):
                if e[0] == "chunk":
                    r = asm.add(*e[1:])
                    log.append(f"{e[2] + 1}{'✓' if r['good'] else '✗'}")
        log.append("|")
    ok = asm.msgs.get(mid, {}).get("payload") == make_payload(text=text)
    print(f"    echo: reverb 0.4 s at -6 dB, frames per copy {' '.join(log[:-1])} → combined", "✓" if ok else "✗")
    # salvage: one frame never arrives; the rest still decompress, the gap is marked
    mid, chunks = new_message(make_payload(text=text), prof)
    x = channel(modulate(mid, chunks, prof, 0.5, [s for s in range(len(chunks)) if s != 1]), rng)
    asm = decode_signal(x, prof)
    typ, _, got = asm.salvage(mid)
    parts = got.split(f" ⟪frame 2/{len(chunks)} missing⟫ ")
    ok = typ == "text" and len(parts) == 2 and text.startswith(parts[0]) and text.endswith(parts[1])
    print(f" salvage: frame 2 of {len(chunks)} lost, kept {len(got):,} of {len(text):,} chars →", "✓" if ok else "✗")


def main():
    ap = argparse.ArgumentParser(description="Acoustic air-gap messenger (multi-tone FSK + Reed–Solomon)")
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
        mid, chunks = new_message(payload, PROFILES[args.profile])
        w = modulate(mid, chunks, PROFILES[args.profile], args.volume)
        write_wav(args.out, w)
        print(f"wrote {args.out} ({len(w) / SR:.1f} s, {PROFILES[args.profile].desc})")
    elif args.mode == "decode":
        if not args.text:
            ap.error("decode needs a WAV path")
        report(decode_signal(read_wav(args.text[0]), PROFILES[args.profile]), PROFILES[args.profile])
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
