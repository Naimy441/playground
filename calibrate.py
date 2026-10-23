#!/usr/bin/env python3
"""Loopback frequency-response sweep for the air-gap modem.

Plays a short ladder of tones from low to high out of the speaker while
recording the microphone, then measures how much of each tone actually
survives the speaker -> air -> mic round-trip.  Use it to find the highest
carrier your hardware can still push through before picking an `airgap`
profile.

    python3 calibrate.py                 # full sweep, recommend a carrier
    python3 calibrate.py --lo 15000 --hi 23000 --step 200   # zoom the top
    python3 calibrate.py --volume 1.0    # louder (default 0.7)

Run it on the SAME machine airgap will transmit from, in a quiet room,
with the speaker at the volume you'll actually use.
"""

import argparse
import sys

import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None

SR = 48000
DUR = 0.12                       # seconds per tone (middle is measured)
GAP = 0.03                       # silence between tones
PAD = 0.15                       # leading/trailing silence

# airgap's `ultra`-style band: M tones spaced df apart, plus chirp headroom.
M, DF = 16, 120
BANDWIDTH = (M - 1) * DF + int(1.6 * DF)   # span the decoder needs to be clean


def build_sweep(freqs):
    """One signal: PAD silence, each tone (raised-cosine) + GAP, PAD silence."""
    n_tone = int(DUR * SR)
    n_gap = int(GAP * SR)
    n_pad = int(PAD * SR)
    env = np.ones(n_tone)
    taper = int(0.01 * SR)
    rc = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, taper))
    env[:taper], env[-taper:] = rc, rc[::-1]
    t = np.arange(n_tone) / SR
    parts = [np.zeros(n_pad)]
    starts = []
    for f in freqs:
        starts.append(sum(len(p) for p in parts))
        parts.append((np.sin(2 * np.pi * f * t) * env))
        parts.append(np.zeros(n_gap))
    parts.append(np.zeros(n_pad))
    sig = np.concatenate(parts).astype(np.float32)
    return sig, np.array(starts), n_tone


def goertzel_mag(seg, f):
    """Amplitude of frequency f within seg (normalized to a unit sine = ~1)."""
    n = np.arange(len(seg))
    return 2.0 * np.abs(seg @ np.exp(-2j * np.pi * f * n / SR)) / len(seg)


def estimate_offset(rec, played_start, n_tone):
    """Find playback->record latency from the first tone's energy onset."""
    env = np.abs(rec)
    k = int(0.005 * SR)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    noise = np.median(env[: int(PAD * 0.7 * SR)]) + 1e-6
    thresh = max(noise * 8, env.max() * 0.05)
    search = env[played_start:]
    hits = np.nonzero(search > thresh)[0]
    if not len(hits):
        return 0
    return int(hits[0])


def main():
    ap = argparse.ArgumentParser(description="Air-gap loopback frequency sweep")
    ap.add_argument("--lo", type=int, default=500, help="lowest tone (Hz)")
    ap.add_argument("--hi", type=int, default=23000, help="highest tone (Hz)")
    ap.add_argument("--step", type=int, default=500, help="tone spacing (Hz)")
    ap.add_argument("--volume", type=float, default=0.7, help="playback 0..1")
    ap.add_argument("--input", type=int, default=None, help="input device index")
    ap.add_argument("--output", type=int, default=None, help="output device index")
    ap.add_argument("--floor", type=float, default=-20.0,
                    help="dB below peak still considered usable")
    args = ap.parse_args()

    if sd is None:
        sys.exit("sounddevice not available — pip install sounddevice")
    if args.hi > SR // 2:
        sys.exit(f"--hi must be <= {SR // 2} Hz (Nyquist)")

    freqs = np.arange(args.lo, args.hi + 1, args.step)
    sig, starts, n_tone = build_sweep(freqs)
    dur = len(sig) / SR
    print(f"sweeping {len(freqs)} tones {args.lo}-{args.hi} Hz "
          f"({dur:.1f} s, vol {args.volume:g}) ...")

    sd.default.samplerate = SR
    if args.input is not None:
        sd.default.device = (args.input, args.output)
    elif args.output is not None:
        sd.default.device = (None, args.output)

    rec = sd.playrec((sig * args.volume)[:, None], channels=1, dtype="float32")
    sd.wait()
    rec = rec[:, 0]

    off = estimate_offset(rec, int(PAD * SR), n_tone)
    guard = int(0.25 * n_tone)                     # ignore edges (align slack)
    noise_seg = rec[: int(PAD * 0.7 * SR)]
    noise = goertzel_mag(noise_seg, (args.lo + args.hi) / 2) if len(noise_seg) else 1e-6

    mags = []
    for s, f in zip(starts, freqs):
        a = s + off + guard
        b = a + (n_tone - 2 * guard)
        seg = rec[a:b]
        if len(seg) < n_tone // 2:
            mags.append(0.0)
            continue
        mags.append(goertzel_mag(seg, f))
    mags = np.array(mags)

    peak = mags.max() + 1e-12
    db = 20 * np.log10(mags / peak + 1e-12)
    snr = 20 * np.log10(mags / (noise + 1e-12) + 1e-12)

    print(f"\n  latency offset: {off / SR * 1000:.0f} ms   "
          f"peak at {freqs[int(np.argmax(mags))]} Hz\n")
    print("   freq      level         SNR")
    for f, d, s in zip(freqs, db, snr):
        bar = "#" * int(np.clip((d + 60) / 60 * 40, 0, 40))
        mark = "  <- usable top" if d >= args.floor else ""
        print(f"  {f:6d}  {d:6.1f} dB  {bar:<40} {s:5.1f}{mark}")

    # Highest contiguous ultra-style band whose tones all clear the floor.
    good = db >= args.floor
    best_f0 = None
    for i in range(len(freqs) - 1, -1, -1):
        f0 = freqs[i]
        lo_f, hi_f = f0, f0 + BANDWIDTH
        in_band = (freqs >= lo_f - DF) & (freqs <= hi_f + DF)
        if in_band.any() and good[in_band].all():
            best_f0 = f0
            break

    print()
    if best_f0 is None:
        print("No band clears the floor — raise --volume or lower --floor.")
    else:
        top = best_f0 + (M - 1) * DF
        print(f"Recommended highest usable carrier: f0 = {best_f0} Hz "
              f"(16 tones {best_f0}-{top} Hz).")
        print("To use it, edit the ultra profile in airgap.py:")
        print(f'    "ultra": Profile("ultra", {best_f0}, {DF}, 0.040, 0.010, 4, 2, 3, '
              f'"near-ultrasonic {best_f0/1000:.1f}-{top/1000:.1f} kHz"),')


if __name__ == "__main__":
    main()
