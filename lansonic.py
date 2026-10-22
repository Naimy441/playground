#!/usr/bin/env python3
"""
lansonic.py — listen to your local network.

Sweeps your subnet, then pings every device it finds and turns the replies
into music.  The network supplies the randomness; a little conductor keeps it
musical:

  * everything is quantised to a 16th-note grid, with a chord progression,
    a bass line and a soft kick/hat groove underneath
  * every device is an instrument with its own register and stereo place
  * a device's latency relative to its own normal picks the pitch: quicker
    than usual climbs, slower than usual sinks; melodies move by step, land
    on chord tones on strong beats, and slow replies arrive late (syncopation)
  * jitter adds vibrato, a dropped packet is a muted glitch and a rim click
  * the gateway drives the bass; the pad follows overall health — lush when
    everything answers, plain when some packets go missing, sour in trouble
  * optional MIDI out: one channel per instrument, GM programs, drums on 10,
    and MIDI clock so a DAW can follow along

    python3 lansonic.py                 # scan your /24 and play
    python3 lansonic.py --fake          # a pretend network (no pinging)
    python3 lansonic.py --hosts 1.1.1.1 8.8.8.8 example.com
    python3 lansonic.py --midi          # also play a virtual MIDI port "LAN Sonifier"
    python3 lansonic.py --midi --midi-port IAC --midi-clock --midi-only
    python3 lansonic.py --list-midi

Keys
  space   mute / unmute            + -   tempo
  s       scale & progression      k     key (round the circle of fifths)
  m       mode: sequencer / orbit  d     pad on/off      b   beat on/off
  v V     volume down / up         r     rescan
  ↑ ↓     select device            enter mute device    o  solo device
  q       quit
"""
import argparse
import curses
import heapq
import ipaddress
import locale
import math
import platform
import random
import re
import socket
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import sounddevice as sd
except Exception:  # still runs as a visualiser / MIDI source
    sd = None

SR = 44100
BLOCK = 1024
KEYS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
SPARK = "▁▂▃▄▅▆▇█"

# scale intervals, and a progression given as the scale degree each chord is built on.
# Chords stack every other scale degree, so they are always made of scale notes.
SCALES = {
    "major pentatonic": ([0, 2, 4, 7, 9], [0, 4, 3, 1]),
    "dorian":           ([0, 2, 3, 5, 7, 9, 10], [0, 3, 6, 4]),
    "aeolian":          ([0, 2, 3, 5, 7, 8, 10], [0, 5, 6, 4]),
    "minor pentatonic": ([0, 3, 5, 7, 10], [0, 4, 3, 2]),
    "lydian":           ([0, 2, 4, 6, 7, 9, 11], [0, 1, 5, 4]),
    "egyptian":         ([0, 2, 5, 7, 10], [0, 3, 1, 4]),
    "hirajoshi":        ([0, 2, 3, 7, 8], [0, 3, 4, 2]),
    "in-sen":           ([0, 1, 5, 7, 10], [0, 2, 4, 3]),
}

# instrument roles: timbre, register centre (MIDI note, key C), GM program, base velocity, gate (beats)
ROLES = [
    ("epiano",  62, 4,   0.42, 0.75),
    ("kalimba", 74, 108, 0.36, 0.40),
    ("vibes",   67, 11,  0.34, 1.00),
    ("harp",    57, 46,  0.40, 0.70),
    ("celesta", 79, 8,   0.26, 0.50),
    ("marimba", 65, 12,  0.40, 0.35),
]
ROLE_PAN = [-0.15, 0.45, -0.55, 0.25, 0.65, -0.35]
BASS_CH, PAD_CH, DRUM_CH = 0, 1, 9
GM_BASS, GM_PAD = 33, 89
GM_SQUARE, GM_CHIP_BASS = 80, 38
# chip style: pulse width per role (the register comes from ROLES) and a base velocity
CHIP_ROLES = [("sq50", 0.30), ("sq12", 0.36), ("sq25", 0.32), ("sq50", 0.24), ("sq12", 0.30), ("sq25", 0.28)]
ORBITS = [3, 4, 5, 6, 7, 5, 9, 4, 11, 6]      # free mode: each device's period in 16ths


def note_name(m):
    return f"{KEYS[m % 12]}{m // 12 - 1}"


def chord_name(root_pc, pcs):
    iv = {(p - root_pc) % 12 for p in pcs}
    if 4 in iv:
        q = ""
    elif 3 in iv:
        q = "m"
    elif 5 in iv:
        q = "sus4"
    elif 2 in iv:
        q = "sus2"
    else:
        q = "5"
    if 10 in iv:
        q += "7"
    elif 11 in iv:
        q += "maj7"
    elif 9 in iv:
        q += "6"
    if 2 in iv and not q.startswith("sus"):
        q += "add9"
    if 6 in iv and 7 not in iv:
        q += "♭5"
    return KEYS[root_pc] + q


# ============================================================================ synth
def nes(env):
    """Quantise an envelope to 16 steps, like a 4-bit volume register."""
    return np.ceil(env * 15) / 15


def nes_noise(frames, hold):
    vals = np.sign(np.random.randn(frames // hold + 2))
    return np.repeat(vals, hold)[:frames]


DUTY = {"sq12": 0.125, "sq25": 0.25, "sq50": 0.5}
CHIP_KINDS = set(DUTY) | {"tri", "ckick", "csnare", "chat", "cglitch", "cblip"}


class Voice:
    DUR = {"epiano": 2.4, "kalimba": 1.4, "vibes": 3.0, "harp": 2.4, "celesta": 2.0, "marimba": 1.2,
           "bass": 1.0, "pad": 4.0, "kick": 0.5, "hat": 0.12, "rim": 0.08, "glitch": 0.4, "blip": 0.6,
           "sq12": 0.4, "sq25": 0.4, "sq50": 0.4, "tri": 0.3, "ckick": 0.18, "csnare": 0.2,
           "chat": 0.05, "cglitch": 0.32, "cblip": 0.035}
    REL = {"pad": 1.6, "bass": 0.12, "sq12": 0.02, "sq25": 0.02, "sq50": 0.02, "tri": 0.01, "cblip": 0.005}
    SEND = {"kick": 0.04, "bass": 0.08, "hat": 0.12, "rim": 0.25, "pad": 0.55}

    def __init__(self, kind, f, vel, pan, vib=0.0, dur=None):
        self.kind, self.f, self.vel, self.vib = kind, f, vel, vib
        a = (pan + 1) * math.pi / 4
        self.gl, self.gr = math.cos(a), math.sin(a)
        self.gate = dur or self.DUR[kind]
        self.rel = self.REL.get(kind, 0.08)
        self.n = int((self.gate + self.rel) * SR)
        self.send = 0.1 if kind in CHIP_KINDS else self.SEND.get(kind, 0.38)
        self.pos = 0
        self.seed = np.random.rand(6) * 6.283

    @property
    def done(self):
        return self.pos >= self.n

    def render(self, frames):
        t = (self.pos + np.arange(frames)) / SR
        self.pos += frames
        f, k = self.f, self.kind
        ph = 2 * math.pi * f * t
        if self.vib:  # vibrato that fades in, like a player leaning on the note
            ph -= (f * self.vib / 5.5) * np.cos(2 * math.pi * 5.5 * t) * np.minimum(1, t / 0.35)
        att = 1 - np.exp(-t * 260)
        sin = np.sin
        if k == "epiano":
            idx = 1.3 * np.exp(-t * 2.4) + 0.2
            y = sin(ph + idx * sin(ph)) * np.exp(-t * 1.2) + 0.08 * sin(14 * ph) * np.exp(-t * 16)
            y *= 0.62
        elif k == "kalimba":
            y = (sin(ph) + 0.3 * sin(6.12 * ph) * np.exp(-t * 18) +
                 0.12 * sin(2 * ph) * np.exp(-t * 6)) * np.exp(-t * 2.8)
        elif k == "vibes":
            y = (sin(ph) + 0.2 * sin(4 * ph) * np.exp(-t * 3)) * np.exp(-t * 0.9)
            y *= 0.82 + 0.18 * sin(2 * math.pi * 5.2 * t)
        elif k == "harp":
            y = sum(sin(h * ph + self.seed[h]) / (h * h) * np.exp(-t * (1.1 + 0.9 * h)) for h in range(1, 6))
            y *= 0.85
        elif k == "celesta":
            y = (sin(ph) + 0.26 * sin(4 * ph) * np.exp(-t * 5) +
                 0.07 * sin(7.2 * ph) * np.exp(-t * 9)) * np.exp(-t * 1.9)
        elif k == "marimba":
            y = sin(ph) * np.exp(-t * 3.6) + 0.22 * sin(3.98 * ph) * np.exp(-t * 17)
        elif k == "bass":
            y = sin(ph) + 0.35 * sin(2 * ph) * np.exp(-t * 2.5) + 0.1 * sin(3 * ph) * np.exp(-t * 5)
            y = np.tanh(1.4 * y) * (0.55 + 0.45 * np.exp(-t * 1.5)) * 0.75
        elif k == "pad":
            env = np.minimum(1, t / min(1.0, self.gate * 0.5))
            y = (sin(ph * 0.997 + self.seed[0]) + sin(ph * 1.003 + self.seed[1]) +
                 0.3 * sin(2.002 * ph + self.seed[2]) + 0.1 * sin(3 * ph + self.seed[3])) * env * 0.3
            att = 1
        elif k == "kick":
            kp = 2 * math.pi * (46 * t + 120 * (1 - np.exp(-t * 30)) / 30)
            y = sin(kp) * np.exp(-t * 7) + 0.15 * np.random.randn(frames) * np.exp(-t * 300)
        elif k == "hat":
            n = np.random.randn(frames + 1)
            y = (n[1:] - n[:-1]) * 0.5 * np.exp(-t * 55)
        elif k == "rim":
            y = (0.6 * sin(2 * math.pi * 1650 * t) + 0.5 * np.random.randn(frames)) * np.exp(-t * 85)
        elif k == "glitch":  # a note that sags away: the packet that never came back
            gp = 2 * math.pi * f * (0.25 / math.log(2)) * (1 - 2 ** (-t / 0.25))
            y = (sin(gp) + 0.3 * sin(2 * gp)) * np.exp(-t * 9) * 0.6
        # ---- 8-bit voices: naive pulse/triangle/noise, 4-bit stepped envelopes
        elif k in DUTY:
            d = DUTY[k]
            fr = (ph / (2 * math.pi)) % 1.0
            y = (np.where(fr < d, 1.0, -1.0) - (2 * d - 1)) * nes(0.55 + 0.45 * np.exp(-t * 7)) * 0.55
            att = 1
        elif k == "tri":
            fr = (ph / (2 * math.pi)) % 1.0
            y = (np.round((2 * np.abs(2 * fr - 1)) * 7.5) / 7.5 - 1) * 0.38
            att = 1
        elif k == "ckick":
            kp = (40 * t + 140 * (1 - np.exp(-t * 40)) / 40) % 1.0
            y = (2 * np.abs(2 * kp - 1) - 1) * nes(np.exp(-t * 16)) * 0.75
            y[: int(0.006 * SR)] += nes_noise(frames, 2)[: int(0.006 * SR)] * 0.3
        elif k == "csnare":
            y = nes_noise(frames, 3) * nes(np.exp(-t * 17)) * 0.32
            y += np.sign(sin(2 * math.pi * 190 * t)) * nes(np.exp(-t * 45)) * 0.15
        elif k == "chat":
            y = nes_noise(frames, 1) * nes(np.exp(-t * 75)) * 0.12
        elif k == "cglitch":  # the "ouch" sweep: two octaves down in a fifth of a second
            gp = (f * (0.2 / math.log(4)) * (1 - 4 ** (-t / 0.2))) % 1.0
            y = np.where(gp < 0.5, 1.0, -1.0) * nes(np.exp(-t * 7)) * 0.26
        elif k == "cblip":
            fr = (ph / (2 * math.pi)) % 1.0
            y = np.where(fr < 0.5, 1.0, -1.0) * 0.18
        else:  # blip
            y = sin(ph) * np.exp(-t * 9) + 0.15 * sin(2 * ph) * np.exp(-t * 20)
        gate = np.clip((self.gate + self.rel - t) / self.rel, 0, 1)
        return y * att * gate * self.vel


class Reverb:
    """Uniformly-partitioned FFT convolution with a synthetic stereo hall."""

    def __init__(self, seconds=2.6, B=BLOCK):
        rng = np.random.default_rng(11)
        L = int(seconds * SR)
        t = np.arange(L) / SR
        f = np.fft.rfftfreq(L, 1 / SR)
        self.B, self.P = B, int(math.ceil(L / B))
        hs = []
        for _ in range(2):
            ir = rng.standard_normal(L) * np.exp(-6.9 * t / seconds)
            ir[: int(0.018 * SR)] = 0                                   # pre-delay
            X = np.fft.rfft(ir) / (1 + (f / 4500) ** 2) / (1 + (120 / (f + 1)) ** 2)  # warm, no mud
            ir = np.fft.irfft(X, L)
            ir /= np.sqrt(np.sum(ir ** 2))
            parts = np.zeros(self.P * B)
            parts[:L] = ir
            hs.append(np.fft.rfft(parts.reshape(self.P, B), 2 * B, axis=1))
        self.H = np.stack(hs)                                          # (2, P, B+1)
        self.fdl = np.zeros((self.P, B + 1), complex)
        self.prev = np.zeros(B)

    def process(self, x):
        X = np.fft.rfft(np.concatenate([self.prev, x]))
        self.prev = x.copy()
        self.fdl = np.roll(self.fdl, 1, axis=0)
        self.fdl[0] = X
        Y = np.einsum("pk,cpk->ck", self.fdl, self.H)
        return np.fft.irfft(Y, 2 * self.B, axis=1)[:, self.B:]


class Synth:
    def __init__(self, enabled=True):
        self.voices = []
        self.lock = threading.Lock()
        self.volume = 0.7
        self.muted = False
        self.level = 0.0
        self.chip = False   # 8-bit mode: dry room, short echo, a little bit-crush
        self.reverb = Reverb()
        self.set_delay(0.4)
        self.stream = None
        self.error = None
        if enabled and sd is not None:
            try:
                self.stream = sd.OutputStream(samplerate=SR, channels=2, dtype="float32",
                                              blocksize=BLOCK, callback=self.cb)
                self.stream.start()
            except Exception as e:  # no audio device etc.
                self.error = str(e)
        elif enabled:
            self.error = "sounddevice not installed (pip install sounddevice) — running silent"

    def set_delay(self, seconds):
        self.D = max(2 * BLOCK, int(seconds * SR))
        self.bl, self.br = np.zeros(self.D, np.float32), np.zeros(self.D, np.float32)
        self.di = 0

    def note(self, kind, midi, vel=0.5, pan=0.0, vib=0.0, dur=None):
        if self.stream is None:
            return
        f = 440.0 * 2 ** ((midi - 69) / 12)
        with self.lock:
            if len(self.voices) < 64:
                self.voices.append(Voice(kind, f, vel, pan, vib, dur))

    def render(self, frames):
        dry = np.zeros((frames, 2))
        send = np.zeros(frames)
        with self.lock:
            voices = list(self.voices)
        for v in voices:
            y = v.render(frames)
            dry[:, 0] += y * v.gl
            dry[:, 1] += y * v.gr
            send += y * v.send
        with self.lock:
            self.voices = [v for v in self.voices if not v.done]
        wet = self.reverb.process(send).T if frames == BLOCK else np.zeros((frames, 2))
        # dotted-eighth ping-pong echo, fed from the reverb send
        idx = (self.di + np.arange(frames)) % self.D
        rl, rr = self.bl[idx].copy(), self.br[idx].copy()
        self.bl[idx] = send * 0.5 + rr * 0.32
        self.br[idx] = rl * 0.9
        self.di = (self.di + frames) % self.D
        if self.chip:
            mix = dry + wet * 0.08 + np.stack([rl, rr], 1) * 0.16
        else:
            mix = dry + wet * 0.42 + np.stack([rl, rr], 1) * 0.22
        g = 0.0 if self.muted else self.volume
        out = np.tanh(mix * g * 0.85)
        if self.chip:
            out = np.round(out * 96) / 96
        return out.astype(np.float32)

    def cb(self, out, frames, ti, status):
        out[:] = self.render(frames)
        self.level = float(np.sqrt(np.mean(out ** 2)))

    def close(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()


# ============================================================================ MIDI
class Midi:
    """Note-on/off with scheduled releases, GM setup and optional MIDI clock."""

    def __init__(self, port=None, clock=False):
        import mido
        self.mido = mido
        if port:
            names = mido.get_output_names()
            match = [n for n in names if port.lower() in n.lower()]
            if not match:
                raise RuntimeError(f"no MIDI output matching {port!r} (have: {', '.join(names) or 'none'})")
            self.out, self.name = mido.open_output(match[0]), match[0]
        else:
            self.out, self.name = mido.open_output("LAN Sonifier", virtual=True), "LAN Sonifier (virtual)"
        self.clock = clock
        self.slock = threading.Lock()
        self.cv = threading.Condition()
        self.heap, self.token, self.seq = [], {}, 0
        self.alive = True
        self.sent = 0
        threading.Thread(target=self._releases, daemon=True).start()

    def send(self, kind, **kw):
        with self.slock:
            self.out.send(self.mido.Message(kind, **kw))

    def setup(self, chip=False):
        if chip:   # square leads, synth bass, square arp
            progs = {BASS_CH: GM_CHIP_BASS, PAD_CH: GM_SQUARE}
            for r in range(len(ROLES)):
                progs[2 + r] = GM_SQUARE
        else:
            progs = {BASS_CH: GM_BASS, PAD_CH: GM_PAD}
            for r, role in enumerate(ROLES):
                progs[2 + r] = role[2]
        for ch, prog in progs.items():
            self.send("program_change", channel=ch, program=prog)
            self.send("control_change", channel=ch, control=7, value=100)
            self.send("control_change", channel=ch, control=91, value=15 if chip else 70)  # reverb send
            pan = ROLE_PAN[ch - 2] if ch >= 2 else 0.0
            self.send("control_change", channel=ch, control=10, value=int(64 + pan * 63))

    def start(self):
        if self.clock:
            self.send("start")

    def note(self, ch, note, vel, dur):
        note = max(0, min(127, int(note)))
        v = max(1, min(127, int(vel * 127)))
        key = (ch, note)
        with self.cv:
            if key in self.token:          # retrigger: close the old one first
                self.send("note_off", channel=ch, note=note)
            self.seq += 1
            self.token[key] = self.seq
            heapq.heappush(self.heap, (time.time() + dur, self.seq, ch, note))
            self.cv.notify()
        self.send("note_on", channel=ch, note=note, velocity=v)
        self.sent += 1

    def cc(self, ch, control, value):
        self.send("control_change", channel=ch, control=control, value=max(0, min(127, int(value))))

    def tick(self):
        if self.clock:
            self.send("clock")

    def _releases(self):
        with self.cv:
            while self.alive:
                if not self.heap:
                    self.cv.wait(0.5)
                    continue
                when, tok, ch, note = self.heap[0]
                wait = when - time.time()
                if wait > 0:
                    self.cv.wait(wait)
                    continue
                heapq.heappop(self.heap)
                if self.token.get((ch, note)) == tok:
                    del self.token[(ch, note)]
                    self.send("note_off", channel=ch, note=note)

    def close(self):
        self.alive = False
        try:
            if self.clock:
                self.send("stop")
            for ch in range(16):
                self.send("control_change", channel=ch, control=123, value=0)  # all notes off
            self.out.close()
        except Exception:
            pass


# ============================================================================ network
IS_MAC = platform.system() == "Darwin"


def ping(ip, timeout=1.0):
    """One ICMP echo via the system ping. Returns latency in ms or None."""
    if IS_MAC:
        cmd = ["ping", "-c", "1", "-W", str(int(timeout * 1000)), "-n", ip]
    else:
        cmd = ["ping", "-c", "1", "-n", "-W", str(max(1, int(math.ceil(timeout)))), ip]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 1.5)
    except (subprocess.TimeoutExpired, OSError):
        return None
    m = re.search(r"time[=<]\s*([\d.]+)\s*ms", r.stdout)
    return float(m.group(1)) if m else None


def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def gateway():
    try:
        if IS_MAC:
            out = subprocess.run(["route", "-n", "get", "default"], capture_output=True, text=True, timeout=3).stdout
            m = re.search(r"gateway:\s*(\S+)", out)
        else:
            out = subprocess.run(["ip", "route"], capture_output=True, text=True, timeout=3).stdout
            m = re.search(r"default via (\S+)", out)
        return m.group(1) if m else None
    except Exception:
        return None


def arp_table():
    try:
        out = subprocess.run(["arp", "-an"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return {}
    return {m.group(1): m.group(2)
            for m in re.finditer(r"\((\d+\.\d+\.\d+\.\d+)\) at ([0-9a-fA-F:]+)", out)}


def rdns(ip):
    try:
        return socket.gethostbyaddr(ip)[0].split(".")[0]
    except Exception:
        return ""


class FakeNet:
    """A pretend home network with personalities."""
    HOSTS = [  # ip, name, base ms, jitter ms, loss
        ("192.168.1.1", "router", 1.8, 0.8, 0.0),
        ("192.168.1.10", "this-laptop", 0.06, 0.02, 0.0),
        ("192.168.1.23", "living-room-tv", 7, 6, 0.02),
        ("192.168.1.31", "sleepy-iphone", 45, 140, 0.12),
        ("192.168.1.40", "printer", 4, 3, 0.05),
        ("192.168.1.57", "smart-bulb", 22, 35, 0.08),
        ("192.168.1.66", "nas", 1.2, 0.4, 0.0),
        ("192.168.1.88", "raspberrypi", 3, 2, 0.01),
        ("192.168.1.102", "game-console", 9, 4, 0.01),
    ]

    def __init__(self):
        self.p = {h[0]: h for h in self.HOSTS}

    def ping(self, ip, timeout=1.0):
        _, _, base, jit, loss = self.p[ip]
        lat = base + abs(random.gauss(0, jit))
        if random.random() < 0.03:
            lat *= random.uniform(4, 15)
        if random.random() < loss or lat > timeout * 1000:
            time.sleep(timeout)
            return None
        time.sleep(lat / 1000)
        return round(lat, 3)


# ============================================================================ app
class Host:
    def __init__(self, ip, name="", gw=False, me=False):
        self.ip, self.name, self.gw, self.me = ip, name, gw, me
        self.hist = deque(maxlen=48)
        self.sent = self.lost = 0
        self.last = None
        self.note = None
        self.flash = 0.0
        self.muted = False
        self.busy = False
        self.role = 0
        self.pan = 0.0
        self.color = 1
        self.mu = None        # running mean of log2 latency: this device's "normal"

    @property
    def voice(self):
        return "bass" if self.gw else ROLES[self.role][0]

    @property
    def loss(self):
        return self.lost / self.sent if self.sent else 0.0

    def jitter(self):
        v = [x for x in list(self.hist)[-8:] if x is not None]
        if len(v) < 3:
            return 0.0
        return float(np.std(v) / (np.mean(v) + 1e-3))

    def deviation(self, lat):
        """How unusual this reply is, in octaves of latency (+ = slower than normal)."""
        ll = math.log2(lat + 0.05)
        if self.mu is None:
            self.mu = ll
        dev = ll - self.mu
        self.mu = 0.85 * self.mu + 0.15 * ll
        return dev


class App:
    def __init__(self, scr, args):
        self.scr = scr
        self.args = args
        self.fake = FakeNet() if args.fake else None
        self.pinger = self.fake.ping if self.fake else ping
        self.synth = Synth(not (args.silent or args.midi_only))
        self.events = deque(maxlen=6)
        self.midi = None
        self.midi_err = None
        if args.midi or args.midi_only or args.midi_port:
            try:
                self.midi = Midi(args.midi_port, args.midi_clock)
            except Exception as e:
                self.midi_err = f"MIDI unavailable: {e}"
        self.chip = args.chip
        self.synth.chip = self.chip
        self.bpm = args.bpm or (124 if self.chip else 96)
        self.scale_i = list(SCALES).index("aeolian") if self.chip else 0
        self.clk = 0
        self.sched = []               # (clock, seq, fn): notes due on a future MIDI-clock tick
        self.sched_seq = 0
        self.arp = []
        self.dialog = deque(maxlen=4)  # Undertale-style typewriter messages (chip style)
        self.typing = None             # (text, started, pitch)
        self.typed = 0
        self.down = {}                 # host ip -> consecutive losses
        self.key = KEYS.index(args.key) if args.key in KEYS else 0
        self.mode = "seq"
        self.drone = True
        self.beat = True
        self.muted = False
        self.hosts = []
        self.sel = 0
        self.solo = None
        self.scanning = False
        self.scan_prog = (0, 1)
        self.roll = deque(maxlen=400)
        self.rr = 0
        self.tick = 0
        self.pending = deque()
        self.gw_state = None          # (latency, deviation) of the last gateway reply
        self.chord_deg = 0
        self.chord_pcs = set()
        self.chord_label = ""
        self.pool = ThreadPoolExecutor(max_workers=48)
        self.running = True
        self.lock = threading.Lock()
        self.init_colors()
        self.update_delay()
        self.set_chord(0)

    # ------------------------------------------------------------------ colours
    def init_colors(self):
        curses.start_color()
        try:
            curses.use_default_colors()
            bg = -1
        except curses.error:
            bg = 0
        c256 = curses.COLORS >= 256
        cols = [51, 213, 220, 120, 75, 209, 159, 183, 229, 117] if c256 else [6, 5, 3, 2, 4, 1, 6, 5, 3, 2]
        for i, c in enumerate(cols):
            curses.init_pair(1 + i, c, bg)
        self.ncol = len(cols)
        curses.init_pair(20, 196 if c256 else 1, bg)   # loss
        curses.init_pair(21, 240 if c256 else 7, bg)   # dim
        self.c256 = c256
        self.bg = bg
        self.theme()
        curses.init_pair(23, 46 if c256 else 2, bg)    # healthy
        curses.init_pair(24, 214 if c256 else 3, bg)   # meh
        curses.init_pair(25, 236 if c256 else 7, bg)   # bar lines

    # ------------------------------------------------------------------ harmony
    @property
    def scale(self):
        return list(SCALES.values())[self.scale_i][0]

    @property
    def progression(self):
        return list(SCALES.values())[self.scale_i][1]

    @property
    def key_shift(self):
        return self.key if self.key <= 6 else self.key - 12

    def degree_pc(self, d):
        sc = self.scale
        return (self.key + sc[d % len(sc)]) % 12

    def chord_degrees(self, extra=False):
        r = self.chord_deg
        return [r, r + 2, r + 4] + ([r + 6] if extra else [])

    def set_chord(self, bar):
        prog = self.progression
        self.chord_deg = prog[bar % len(prog)]
        self.chord_pcs = {self.degree_pc(d) for d in self.chord_degrees()}
        self.chord_root = self.degree_pc(self.chord_deg)
        self.chord_label = chord_name(self.chord_root, self.chord_pcs)

    def nearest(self, pc, around):
        """MIDI note with pitch class pc closest to `around`."""
        base = around - ((around - pc) % 12)
        return base if around - base <= 6 else base + 12

    def update_delay(self):
        self.synth.set_delay(60 / self.bpm * 0.75)

    def audible(self, h):
        if self.solo is not None:
            return h is self.solo
        return not h.muted

    # ------------------------------------------------------------------ replies
    def on_reply(self, h, lat):
        h.busy = False
        h.sent += 1
        h.hist.append(lat)
        h.last = lat
        h.flash = time.time()
        name = h.name or h.ip
        if lat is None:
            h.lost += 1
            self.down[h.ip] = self.down.get(h.ip, 0) + 1
            if self.down[h.ip] == 1:
                self.say("The gateway stumbles. The bass holds its breath." if h.gw
                         else f"{name} dropped a packet.")
        else:
            if self.down.get(h.ip, 0) >= 2:
                self.say(f"{name} is back! It's filled with determination.")
            self.down[h.ip] = 0
        if h.gw:
            self.gw_state = None if lat is None else (lat, h.deviation(lat))
            return
        with self.lock:
            self.pending.append((h, lat, self.tick))

    def probe(self, h):
        try:
            lat = self.pinger(h.ip, self.args.timeout)
        except Exception:
            lat = None
        if self.running:
            self.on_reply(h, lat)

    def say(self, text, pitch=None):
        """Queue a line for the dialogue box (and the plain event line)."""
        self.events.append((time.time(), text))
        if not self.dialog or self.dialog[-1][0] != text:
            self.dialog.append((text, pitch or random.choice([69, 71, 74, 76])))

    def later(self, clocks, fn):
        """Run fn `clocks` MIDI-clock ticks from now (24 per beat)."""
        with self.lock:
            self.sched_seq += 1
            heapq.heappush(self.sched, (self.clk + clocks, self.sched_seq, fn))

    def send_ping(self, h):
        if not h.busy:
            h.busy = True
            self.pool.submit(self.probe, h)

    def health(self):
        hs = [h for h in self.hosts if h.sent]
        if not hs:
            return 1.0, 0.0
        recent = [x for h in hs for x in list(h.hist)[-10:]]
        loss = sum(1 for x in recent if x is None) / max(1, len(recent))
        lats = [x for x in recent if x is not None]
        return 1 - loss, (float(np.median(lats)) if lats else 0.0)

    # ------------------------------------------------------------------ playing
    @property
    def beat_len(self):
        return 60 / self.bpm

    def accent(self, pos):
        if pos == 0:
            a = 1.0
        elif pos % 8 == 0:
            a = 0.88
        elif pos % 4 == 0:
            a = 0.78
        elif pos % 2 == 0:
            a = 0.66
        else:
            a = 0.56
        return a * random.uniform(0.9, 1.05)

    def pick_note(self, h, lat, pos):
        _, centre, *_ = ROLES[h.role]
        centre += self.key_shift
        strong = pos % 4 == 0
        pcs = self.chord_pcs if strong else {self.degree_pc(d) for d in range(len(self.scale))}
        cands = [m for m in range(centre - 9, centre + 10) if m % 12 in pcs]
        dev = h.deviation(lat)
        x = min(1.0, max(0.0, 0.5 - dev * 0.4))   # quicker than usual → higher
        target = cands[int(round(x * (len(cands) - 1)))]
        if h.note is not None and abs(h.note - centre) < 14:
            # move by step: no leaps wider than a fourth
            want = max(h.note - 5, min(h.note + 5, target))
            pick = min(cands, key=lambda m: (abs(m - want), abs(m - target)))
            if pick == h.note and random.random() < 0.6:
                i = cands.index(pick)
                step = 1 if target > pick or (target == pick and random.random() < 0.5) else -1
                pick = cands[max(0, min(len(cands) - 1, i + step))]
            return pick
        return target

    def play_reply(self, h, lat, pos):
        if not self.audible(h) or self.muted:
            return
        name, _, _, base_vel, gate = ROLES[h.role]
        if self.chip:
            name, base_vel = CHIP_ROLES[h.role]
        ch = 2 + h.role
        if lat is None:
            m = (h.note or ROLES[h.role][1] + self.key_shift) - 12
            if self.chip:
                self.synth.note("cglitch", m + 12, 0.9, h.pan)
            else:
                self.synth.note("glitch", m, 0.32, h.pan)
                self.synth.note("rim", 60, 0.28, h.pan)
            if self.midi:
                self.midi.note(DRUM_CH, 37, 0.35, 0.1)
            self.roll[-1][-1] = 20
            h.note = None
            return
        m = self.pick_note(h, lat, pos)
        h.note = m
        vel = base_vel * self.accent(pos)
        jit = h.jitter()
        vib = min(0.018, jit * 0.012)
        if self.chip:
            g = gate * self.beat_len
            self.synth.note(name, m, vel, h.pan, vib + 0.004, dur=g)
            # the classic chiptune echo channel: a quieter thin copy a dotted 8th later
            self.later(18, lambda: self.synth.note("sq12", m, vel * 0.32, -h.pan, 0.0, dur=g * 0.8))
        else:
            self.synth.note(name, m, vel, h.pan, vib)
        if self.midi:
            self.midi.cc(ch, 1, min(127, jit * 160))
            self.midi.note(ch, m, min(1.0, vel * 2.0), gate * self.beat_len)
        self.roll[-1][m] = h.color

    def new_bar(self, bar):
        self.set_chord(bar)
        ok, _ = self.health()
        if bar % 8 == 4:
            self.flavour(ok)
        if self.chip:
            # chords become a quick square-wave arpeggio up high instead of a pad
            degs = self.chord_degrees(extra=ok > 0.96)
            up = sorted({self.nearest(self.degree_pc(d), 74 + self.key_shift + 3 * i)
                         for i, d in enumerate(degs)})
            if ok < 0.85:
                up.append(up[0] + 1)
            self.arp = up + up[-2:0:-1] if len(up) > 2 else up
            return
        if self.drone and not self.muted:
            degs = self.chord_degrees(extra=ok > 0.96)
            notes = sorted({self.nearest(self.degree_pc(d), 55 + self.key_shift + 3 * i)
                            for i, d in enumerate(degs)})
            if ok < 0.85:                        # trouble: a semitone rub against the root
                notes.append(notes[0] + 13)
            dur = self.beat_len * 4 * 1.02
            for n in notes:
                self.synth.note("pad", n, 0.075, random.uniform(-0.5, 0.5), dur=dur)
                if self.midi:
                    self.midi.note(PAD_CH, n, 0.4, dur * 0.98)

    def flavour(self, ok):
        n = len(self.hosts)
        if ok > 0.97:
            lines = ["The network hums quietly. You feel at peace.",
                     f"All {n} devices answer. It's a beautiful day outside.",
                     "Packets drift by like fireflies.",
                     "The gateway keeps the beat. Somewhere, a skeleton taps its foot."]
        elif ok > 0.85:
            lines = ["Somewhere, a phone is taking a nap.",
                     "A few packets wander off. The band plays on.",
                     "The Wi-Fi smells faintly of microwave."]
        else:
            lines = ["The air is heavy with lost packets.",
                     "You feel like you're going to have a bad time.",
                     "The router is trying its best."]
        self.say(random.choice(lines))

    def bass_step(self, pos):
        """The gateway is the bass player: steady when it is, busy when it's quick, absent when it drops."""
        if self.muted:
            return
        gw = next((h for h in self.hosts if h.gw), None)
        if gw is not None and not self.audible(gw):
            return
        st = self.gw_state if gw is not None else (1.0, 0.0)
        if st is None:
            if pos == 0:
                self.roll[-1][-2] = 20
            return
        _, dev = st
        root = self.nearest(self.chord_root, 40 + self.key_shift)
        fifth = self.nearest(self.degree_pc(self.chord_deg + 4), root + 7)
        third = self.nearest(self.degree_pc(self.chord_deg + 2), root + 4)
        if self.chip:
            # triangle bass: octave bounce when the gateway is quick, root-fifth walk normally
            o = root + 12
            if dev < -0.15:
                pattern = {p: ((root, o)[(p // 2) % 2], 0.8, 0.4) for p in range(0, 16, 2)}
                pattern[14] = (fifth, 0.8, 0.4)
            elif dev > 0.6:
                pattern = {0: (root, 0.8, 3.6)}
            else:
                pattern = {0: (root, 0.8, 0.9), 3: (root, 0.7, 0.4), 6: (o, 0.7, 0.4),
                           8: (fifth, 0.8, 0.9), 11: (fifth, 0.7, 0.4), 14: (third, 0.7, 0.4)}
        else:
            pattern = {0: (root, 0.62, 2.5), 10: (fifth, 0.42, 1.4)}
            if dev < -0.15:
                pattern[6] = (root + 12, 0.36, 0.9)
                pattern[14] = (third, 0.32, 0.5)
            elif dev > 0.6:
                pattern = {0: (root, 0.55, 3.8)}
        if pos in pattern:
            n, v, beats = pattern[pos]
            v *= random.uniform(0.93, 1.03)
            self.synth.note("tri" if self.chip else "bass", n, v, 0.0, dur=beats * self.beat_len)
            if self.midi:
                self.midi.note(BASS_CH, n, min(1.0, v * 1.7), beats * self.beat_len * 0.95)
            self.roll[-1][-2] = gw.color if gw else 1

    def drums(self, pos):
        if not self.beat or self.muted:
            return
        ok, _ = self.health()
        hits = []
        if self.chip:
            bar = self.tick // 16
            if pos in (0, 8) or (pos == 10 and ok > 0.96):
                hits.append(("ckick", 36, 0.85 if pos == 0 else 0.7))
            if pos in (4, 12):
                hits.append(("csnare", 38, 0.8))
            elif bar % 4 == 3 and pos in (13, 14, 15) and ok > 0.9:   # a little fill
                hits.append(("csnare", 38, 0.45 + 0.12 * (pos - 13)))
            if pos % 2 == 0:
                hits.append(("chat", 42, 0.9 if pos % 4 == 2 else 0.5))
            for kind, gm, v in hits:
                self.synth.note(kind, 60, v, 0.1 if kind == "chat" else 0.0)
                if self.midi:
                    self.midi.note(DRUM_CH, gm, min(1.0, v * 0.9), 0.08)
            return
        if pos == 0:
            hits.append(("kick", 36, 0.75))
        elif pos == 8 and ok > 0.85:
            hits.append(("kick", 36, 0.5))
        elif pos == 7 and ok > 0.97 and random.random() < 0.3:
            hits.append(("kick", 36, 0.3))
        if pos % 4 == 2:
            hits.append(("hat", 42, 0.16 * random.uniform(0.8, 1.1)))
        elif pos % 2 == 1 and random.random() < 0.25:
            hits.append(("hat", 42, 0.07))
        for kind, gm, v in hits:
            self.synth.note(kind, 60, v, 0.12 if kind == "hat" else 0.0)
            if self.midi:
                self.midi.note(DRUM_CH, gm, min(1.0, v * 1.3), 0.08)

    def step16(self):
        tick = self.tick
        pos = tick % 16
        self.roll.append({})
        if pos == 0:
            self.new_bar(tick // 16)
        self.drums(pos)
        self.bass_step(pos)
        if self.chip and self.drone and self.arp and not self.muted:
            n = self.arp[tick % len(self.arp)]
            v = 0.13 if pos % 4 == 0 else 0.09
            self.synth.note("sq12", n, v, 0.3 * (1 if tick % 2 else -1), dur=self.beat_len / 4 * 0.7)
            if self.midi:
                self.midi.note(PAD_CH, n, v * 2.5, self.beat_len / 4 * 0.7)
        hosts = list(self.hosts)
        players = [h for h in hosts if not h.gw]
        gw = next((h for h in hosts if h.gw), None)
        # pings go out one 16th early, so a quick reply lands on the beat and a slow one lands late
        if gw is not None and pos == 15:
            self.send_ping(gw)
        if players:
            if self.mode == "seq":
                if pos % 2 == 1 and pos != 15:
                    for _ in range(len(players)):
                        h = players[self.rr % len(players)]
                        self.rr += 1
                        if not h.busy:
                            self.send_ping(h)
                            break
            else:
                for i, h in enumerate(players):
                    p = ORBITS[i % len(ORBITS)]
                    if (tick + 1) % p == i % p:
                        self.send_ping(h)
        # play what came back, at most two notes per 16th
        with self.lock:
            due = []
            while self.pending and len(due) < 2:
                item = self.pending.popleft()
                if tick - item[2] <= 8:
                    due.append(item)
        for h, lat, _ in due:
            self.play_reply(h, lat, pos)
        self.tick += 1

    # ------------------------------------------------------------------ discovery
    def discover(self):
        self.scanning = True
        self.hosts = []
        me = local_ip()
        if self.fake:
            gw, me = "192.168.1.1", "192.168.1.10"
            self.net = "192.168.1.0/24 (pretend)"
            for i, (ip, name, *_) in enumerate(FakeNet.HOSTS):
                time.sleep(0.18)
                self.add_host(ip, name, ip == gw, ip == me)
                self.scan_prog = (i + 1, len(FakeNet.HOSTS))
        elif self.args.hosts:
            self.net = "custom hosts"
            targets = self.args.hosts
            for i, t in enumerate(targets):
                try:
                    ip = socket.gethostbyname(t)
                except OSError:
                    continue
                self.add_host(ip, t if t != ip else rdns(ip), False, False)
                self.scan_prog = (i + 1, len(targets))
        else:
            gw = gateway()
            net = ipaddress.ip_network(self.args.subnet or f"{me}/24", strict=False)
            self.net = str(net)
            ips = [str(i) for i in net.hosts()][:1024]
            self.scan_prog = (0, len(ips))
            done = [0]

            def sweep(ip):
                lat = ping(ip, 0.7)
                done[0] += 1
                self.scan_prog = (done[0], len(ips))
                if lat is not None or ip in (gw, me):
                    self.add_host(ip, "", ip == gw, ip == me)
            with ThreadPoolExecutor(max_workers=96) as ex:
                list(ex.map(sweep, ips))
            if gw and not any(h.ip == gw for h in self.hosts) and ping(gw, 1.0) is not None:
                self.add_host(gw, "", True, False)
            arp = arp_table()
            for h in self.hosts:
                h.mac = arp.get(h.ip, "")
            with ThreadPoolExecutor(max_workers=32) as ex:
                for h, n in zip(self.hosts, ex.map(rdns, [h.ip for h in self.hosts])):
                    h.name = n
        with self.lock:
            self.hosts.sort(key=lambda h: tuple(int(p) for p in h.ip.split(".")))
            self.assign_voices()
        self.scanning = False

    def add_host(self, ip, name, gw, me):
        with self.lock:
            if any(h.ip == ip for h in self.hosts):
                return
            self.hosts.append(Host(ip, name, gw, me))
            self.assign_voices()
            n = len(self.hosts)
        # each discovery chimes, climbing the scale
        sc = self.scale
        m = 62 + self.key_shift + 12 * (n // len(sc)) + sc[n % len(sc)]
        self.synth.note("sq25" if self.chip else "celesta", min(m, 96), 0.3, random.uniform(-0.6, 0.6),
                        dur=0.12 if self.chip else None)
        self.events.append((time.time(), f"found {ip}" + (f" ({name})" if name else "")))

    def assign_voices(self):
        others = 0
        for i, h in enumerate(self.hosts):
            h.color = 1 + i % self.ncol
            if h.gw:
                h.pan = 0.0
                continue
            h.role = others % len(ROLES)
            h.pan = max(-0.9, min(0.9, ROLE_PAN[h.role] + 0.12 * ((others // len(ROLES)) % 3 - 1)))
            others += 1

    # ------------------------------------------------------------------ clock
    def conductor(self):
        self.discover()
        if self.midi:
            self.midi.setup(self.chip)
            self.midi.start()
        self.say(f"{len(self.hosts)} devices found. The band takes the stage.")
        nxt = time.perf_counter()
        clk = 0
        while self.running:
            if self.scanning:
                time.sleep(0.05)
                nxt = time.perf_counter()
                continue
            now = time.perf_counter()
            if now < nxt:
                time.sleep(min(0.002, nxt - now))
                continue
            nxt += 60 / self.bpm / 24                 # MIDI clock resolution: 24 per quarter
            if nxt < now - 0.1:
                nxt = now
            if self.midi:
                self.midi.tick()
            self.clk = clk
            if clk % 6 == 0:
                self.step16()
            while True:
                with self.lock:
                    if not self.sched or self.sched[0][0] > clk:
                        break
                    fn = heapq.heappop(self.sched)[2]
                if not self.muted:
                    fn()
            clk += 1

    # ------------------------------------------------------------------ ui
    def theme(self):
        if self.chip:   # black, white and that yellow
            curses.init_pair(22, 226 if self.c256 else 3, 16 if self.c256 else 0)
        else:
            curses.init_pair(22, 0, 51 if self.c256 else 6)

    def draw_dialog(self, put, y0, W, now):
        """An Undertale-ish text box: lines type themselves out, one blip per letter."""
        w = min(W - 4, 86)
        x0 = 2
        white = curses.A_BOLD
        put(y0, x0, "┌" + "─" * (w - 2) + "┐", white)
        put(y0 + 3, x0, "└" + "─" * (w - 2) + "┘", white)
        for r in (1, 2):
            put(y0 + r, x0, "│" + " " * (w - 2) + "│", white)
        if self.typing is None or (self.typed >= len(self.typing[0]) and now - self.typing[1] >
                                   len(self.typing[0]) / 30 + 2.2 and self.dialog):
            if self.dialog:
                text, pitch = self.dialog.popleft()
                self.typing, self.typed = (text, now, pitch), 0
        if self.typing is None:
            return
        text, started, pitch = self.typing
        n = min(len(text), int((now - started) * 30))
        if n > self.typed:
            ch = text[n - 1]
            if self.chip and ch.isalnum() and not self.muted:
                self.synth.note("cblip", pitch + random.choice((0, 0, 2)), 0.5, 0.0)
            self.typed = n
        shown = "* " + text[:n]
        lines = [shown[:w - 6], shown[w - 6:2 * (w - 6)]]
        for r, line in enumerate(lines):
            put(y0 + 1 + r, x0 + 2, line, white)

    def draw(self):
        scr = self.scr
        H, W = scr.getmaxyx()
        scr.erase()
        now = time.time()

        def put(y, x, s, a=0):
            if 0 <= y < H and 0 <= x < W - 1:
                try:
                    scr.addstr(y, x, s[: W - 1 - x], a)
                except curses.error:
                    pass

        title = " ♥ LAN SONIFIER " if self.chip else " ♪ LAN SONIFIER "
        put(0, 0, title.ljust(W - 1), curses.color_pair(22) | curses.A_BOLD)
        ok, med = self.health()
        hp = 23 if ok > 0.96 else 24 if ok > 0.85 else 20
        scale = list(SCALES)[self.scale_i]
        put(0, len(title) + 1, f"{getattr(self, 'net', '…')}  ·  {scale} in {KEYS[self.key]}  ·  "
            f"{self.bpm} bpm  ·  {'sequencer' if self.mode == 'seq' else 'orbits'}  ·  "
            f"vol {int(self.synth.volume * 100)}%{'  MUTED' if self.muted else ''}",
            curses.color_pair(22))
        bar_w = 20
        fill = int(ok * bar_w)
        put(1, 1, "health ", curses.A_DIM)
        put(1, 8, "█" * fill + "░" * (bar_w - fill), curses.color_pair(hp))
        pos = self.tick % 16
        beats = "".join("●" if (pos // 4) == i else "○" for i in range(4))
        info = (f" {ok * 100:5.1f}% replies  ·  median {med:6.2f} ms  ·  "
                f"bar {self.tick // 16 + 1:>3} {beats}  chord ")
        x = 9 + bar_w
        put(1, x, info, curses.A_DIM)
        x += len(info)
        put(1, x, f"{self.chord_label:<9}", curses.color_pair(hp) | curses.A_BOLD)
        flags = (f" pad {'on' if self.drone else 'off'} · beat {'on' if self.beat else 'off'} · "
                 f"out {'▮' * int(min(1, self.synth.level * 5) * 8):<8}")
        put(1, x + 9, flags, curses.A_DIM)
        if self.midi:
            put(2, 1, f"MIDI → {self.midi.name}{' + clock' if self.midi.clock else ''} · "
                f"{self.midi.sent} notes", curses.color_pair(4))
        elif self.midi_err:
            put(2, 1, self.midi_err, curses.color_pair(20))
        if self.synth.error:
            put(2, W - len(self.synth.error) - 2, self.synth.error, curses.color_pair(20))

        if self.scanning:
            d, n = self.scan_prog
            w = max(10, W - 40)
            f = int(w * d / max(1, n))
            put(4, 2, f"sweeping {getattr(self, 'net', '')}  ", curses.A_BOLD)
            put(5, 2, "▕" + "█" * f + "·" * (w - f) + f"▏ {d}/{n}", curses.color_pair(1))

        roll_h = max(6, min(14, (H - 9) // 3))
        list_top = 7 if self.scanning else 4
        hist_w = min(48, max(8, W - 92))
        put(list_top, 2, f"{'':2}{'device':<22}{'address':<17}{'voice':<9}{'latency':>9}  "
            f"{'history':<{hist_w}}  {'loss':>5}  note", curses.A_UNDERLINE)
        box_h = 4 if self.chip else 0
        max_rows = H - roll_h - list_top - 4 - box_h
        hosts = list(self.hosts)
        if self.sel >= len(hosts):
            self.sel = max(0, len(hosts) - 1)
        start = max(0, self.sel - max_rows + 1)
        for row, h in enumerate(hosts[start:start + max_rows]):
            i = start + row
            y = list_top + 1 + row
            lit = now - h.flash < 0.18
            cp = curses.color_pair(20 if (h.last is None and h.sent) else h.color)
            dot = "●" if lit else ("○" if self.audible(h) else "·")
            name = h.name or ("this machine" if h.me else "gateway" if h.gw else "—")
            name = ("⌂ " if h.gw else "★ " if h.me else "") + name
            attr = curses.A_BOLD if lit else 0
            if i == self.sel:
                attr |= curses.A_REVERSE
            if self.chip and i == self.sel:
                dot, cp = "♥", curses.color_pair(20)
            put(y, 2, dot, cp | curses.A_BOLD)
            lat = "  —  " if h.last is None else f"{h.last:7.2f}ms"
            if h.sent and h.last is None:
                lat = "  lost "
            line = f" {name[:21]:<21}{h.ip:<17}{h.voice:<9}{lat:>9}  "
            put(y, 3, line, attr | (0 if self.audible(h) else curses.color_pair(21)))
            x = 3 + len(line)
            for j, v in enumerate(list(h.hist)[-hist_w:]):
                if v is None:
                    put(y, x + j, "×", curses.color_pair(20))
                else:
                    lv = 1 - min(1, max(0, (math.log10(max(v, 0.05)) + 1.3) / 4.2))
                    put(y, x + j, SPARK[min(7, int(lv * 8))], cp)
            x += hist_w + 2
            put(y, x, f"{h.loss * 100:4.0f}%", curses.color_pair(20) if h.loss > 0.05 else curses.A_DIM)
            if h.note is not None and not h.gw:
                put(y, x + 7, note_name(h.note), cp | (curses.A_BOLD if lit else curses.A_DIM))

        # piano roll: time flows left, one column per 16th, bar lines every 16
        top = H - roll_h - 2 - box_h
        put(top - 1, 1, "─" * (W - 3), curses.color_pair(21))
        put(top - 1, 3, f" piano roll · {self.chord_label} ", curses.color_pair(21))
        cols = list(self.roll)[-(W - 4):]
        offset = W - 3 - len(cols)
        lo, hi = 48, 90
        first_tick = self.tick - len(cols)
        for c, col in enumerate(cols):
            if (first_tick + c) % 16 == 0:
                for y in range(top, top + roll_h):
                    put(y, offset + c, "┊", curses.color_pair(25))
            for m, cp in col.items():
                if m == -1:
                    y, ch = top + roll_h - 1, "×"
                elif m == -2:
                    y, ch = top + roll_h - 1, "▬"
                else:
                    y = top + int((hi - min(hi, max(lo, m))) * (roll_h - 2) / (hi - lo))
                    ch = "●"
                put(y, offset + c, ch, curses.color_pair(cp) | curses.A_BOLD)
        if self.chip:
            self.draw_dialog(put, H - 2 - box_h, W, now)
        else:
            ev = [m for t, m in list(self.events) if now - t < 6]
            if ev:
                put(H - 2, 2, "  ·  ".join(ev[-3:]), curses.color_pair(21))
        put(H - 1, 0, " space mute · +/- tempo · s scale · k key · m mode · c chip · d pad · b beat · v/V vol · "
            "↑↓ select · enter mute · o solo · r rescan · q quit ".ljust(W - 1)[: W - 1], curses.A_REVERSE)
        scr.refresh()

    def handle(self, k):
        if k in (ord('q'), 27):
            return False
        if k == ord(' '):
            self.muted = not self.muted
        elif k in (ord('+'), ord('=')):
            self.bpm = min(220, self.bpm + 6)
            self.update_delay()
        elif k == ord('-'):
            self.bpm = max(50, self.bpm - 6)
            self.update_delay()
        elif k == ord('s'):
            self.scale_i = (self.scale_i + 1) % len(SCALES)
            self.events.append((time.time(), "scale: " + list(SCALES)[self.scale_i]))
        elif k == ord('k'):
            self.key = (self.key + 7) % 12
            self.events.append((time.time(), "key: " + KEYS[self.key]))
        elif k == ord('m'):
            self.mode = "free" if self.mode == "seq" else "seq"
        elif k == ord('d'):
            self.drone = not self.drone
        elif k == ord('c'):
            self.chip = not self.chip
            self.synth.chip = self.chip
            self.theme()
            if self.midi:
                self.midi.setup(self.chip)
            self.say("You switched to 8-bit. It's nostalgic." if self.chip else "Back to the lush hall sound.")
        elif k == ord('b'):
            self.beat = not self.beat
        elif k == ord('v'):
            self.synth.volume = max(0.0, self.synth.volume - 0.1)
        elif k == ord('V'):
            self.synth.volume = min(1.5, self.synth.volume + 0.1)
        elif k == curses.KEY_UP:
            self.sel = max(0, self.sel - 1)
        elif k == curses.KEY_DOWN:
            self.sel = min(len(self.hosts) - 1, self.sel + 1)
        elif k in (10, 13, curses.KEY_ENTER) and self.hosts:
            h = self.hosts[self.sel]
            h.muted = not h.muted
        elif k == ord('o') and self.hosts:
            h = self.hosts[self.sel]
            self.solo = None if self.solo is h else h
        elif k == ord('r') and not self.scanning:
            threading.Thread(target=self.discover, daemon=True).start()
        return True

    def run(self):
        threading.Thread(target=self.conductor, daemon=True).start()
        try:
            while True:
                k = self.scr.getch()
                while k != -1:
                    if not self.handle(k):
                        return
                    k = self.scr.getch()
                self.draw()
                time.sleep(1 / 24)
        finally:
            self.running = False
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.synth.close()
            if self.midi:
                self.midi.close()


def main(scr, args):
    curses.curs_set(0)
    scr.nodelay(True)
    scr.keypad(True)
    App(scr, args).run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Listen to the rhythm and health of your network")
    ap.add_argument("--hosts", nargs="+", help="ping these hosts instead of scanning")
    ap.add_argument("--subnet", help="subnet to sweep, e.g. 192.168.0.0/24 (default: your /24)")
    ap.add_argument("--fake", action="store_true", help="simulate a home network instead of pinging")
    ap.add_argument("--bpm", type=int, default=None, help="tempo (default 96, or 124 with --chip)")
    ap.add_argument("--chip", action="store_true", help="8-bit style: pulse leads, triangle bass, noise drums")
    ap.add_argument("--key", default="D", choices=KEYS)
    ap.add_argument("--timeout", type=float, default=0.8, help="ping timeout (s)")
    ap.add_argument("--midi", action="store_true", help="send MIDI to a virtual port called 'LAN Sonifier'")
    ap.add_argument("--midi-port", help="send MIDI to an existing output instead (substring match, e.g. IAC)")
    ap.add_argument("--midi-clock", action="store_true", help="also send MIDI clock + start/stop")
    ap.add_argument("--midi-only", action="store_true", help="MIDI out only, no built-in synth")
    ap.add_argument("--list-midi", action="store_true", help="list MIDI outputs and exit")
    ap.add_argument("--silent", action="store_true", help="no built-in audio")
    args = ap.parse_args()
    if args.list_midi:
        try:
            import mido
            names = mido.get_output_names()
            print("\n".join(names) if names else "no MIDI outputs (use --midi for a virtual port)")
        except ImportError:
            print("needs: pip install mido python-rtmidi")
        raise SystemExit
    locale.setlocale(locale.LC_ALL, "")
    curses.wrapper(main, args)
