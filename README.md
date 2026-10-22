# playground

Four small toys, each one self-contained Python file. Needs Python 3.9+ and `numpy`. The two audio toys also need `sounddevice` (`pip install sounddevice`), and MIDI needs `pip install mido python-rtmidi`, and `sand.py` uses Pillow, if you have it, for real fonts.

| script | what it is |
|---|---|
| `fluid.py` | Navier–Stokes smoke tank in curses (Jos Stam's Stable Fluids) |
| `sand.py` | falling-sand sculptures: letterforms and fractals that form and then crumble |
| `lansonic.py` | pings your subnet and turns the replies into pentatonic music |
| `airgap.py` | sends text and files between laptops over (near-)ultrasonic FSK |

Use a 256-colour terminal (iTerm2, Kitty, WezTerm, Ghostty and modern Terminal.app all work). Bigger windows look better. Press `h` in the curses apps for help.

---

## fluid.py — terminal smoke

```bash
python3 fluid.py
```

It starts in a demo mode: two wandering jets, plus a random explosion or vortex every few seconds. Touch any action key and you're driving.

- **Physics:** semi-Lagrangian advection, warm-started Jacobi pressure projection, vorticity confinement for curly turbulence, buoyancy so the smoke rises, three dye channels that mix colour. Physics runs with a 2:1 cell aspect ratio, so vortices stay round even though terminal cells are tall.
- **Controls:** arrows or WASD move and aim the nozzle. `space` puffs, `f` toggles a continuous jet, `e` plants a permanent emitter, `x` explodes, `v`/`V` spins, `1-7`/`0` pick the dye colour (0 is rainbow). You can also drag with the mouse.
- **Views:** `r` cycles through smoke, velocity (arrows coloured by direction), vorticity and pressure. `m` cycles glyph sets (ASCII, blocks, braille, dots).
- **Tuning:** `g` sets vorticity strength, `b` toggles buoyancy, `[ ]` change how fast the smoke fades, `- =` change viscosity.

## sand.py — sand & calligraphy

```bash
python3 sand.py                        # built-in playlist: words alternating with fractals
python3 sand.py "hello" "goodbye"      # your words
python3 sand.py --only-text --font zapfino "Ink"
```

The grid uses ▀▄ half-blocks, so each cell holds two grains.

1. **Assembly:** grains arrive in one of three ways. *pour* fills the shape from the bottom up, *write* sweeps in left to right like a brush, and *swirl* spirals in around the centre.
2. **Hold:** the shape sits for a moment while a glint sweeps across it.
3. **Dissolve:** one of *crumble*, *wind* (blown off like dust) or *explode*. Falling grains land in a cellular-automaton sand pile. A hole then opens in the floor and the pile drains out like an hourglass.

Letterforms come from your system fonts: Zapfino, Snell Roundhand, Savoye, Apple Chancery, Brush Script, Didot and others. A built-in pixel font is the fallback. The fractals are: Barnsley fern, Sierpiński triangle and carpet, Mandelbrot, Heighway dragon, Lévy C, Vicsek, Koch snowflake and a rose.

Keys: `space` dissolves now, `t` types new words, `a`/`d` set the assembly and dissolve styles, `f` picks a font, `c` picks a palette, `+ -` change speed. Click to knock grains loose.

## lansonic.py — Audio LAN Sonifier

```bash
python3 lansonic.py                    # sweep your /24, then play
python3 lansonic.py --fake             # a pretend home network, no pinging
python3 lansonic.py --hosts 1.1.1.1 8.8.8.8 github.com
python3 lansonic.py --midi             # also play a virtual MIDI port "LAN Sonifier"
python3 lansonic.py --chip --fake      # 8-bit, Undertale-ish
python3 lansonic.py --midi-port IAC --midi-clock --midi-only   # drive a DAW
python3 lansonic.py --list-midi
```

The network supplies the randomness and a small conductor keeps it musical. Listen to `lansonic_demo.m4a` for 45 seconds of the fake network.

- **Grid:** everything is quantised to 16th notes. Pings go out one 16th early, so a quick reply lands on the beat and a slow one lands late. Latency turns into syncopation.
- **Harmony:** each scale has a chord progression built from its own notes, so nothing clashes. The options are major pentatonic, Dorian, minor pentatonic, Lydian, Egyptian, hirajoshi and in-sen; `s` cycles them and `k` walks the circle of fifths.
- **Melody:** each device is an instrument with its own register: e-piano, kalimba, vibes, harp, celesta or marimba. A reply quicker than *that device's* usual climbs, and a slower one sinks. Melodies move by step (no leap wider than a fourth) and land on chord tones on strong beats. Jitter adds vibrato.
- **Rhythm section:** the gateway is the bass player. It gets busy when it's quick, simple when it's sluggish, and drops out when it drops packets. A soft kick and hat groove sits underneath; `b` toggles it.
- **Pad:** the pad follows overall health. It adds a 7th or 9th when everything answers, plays a plain triad when some packets go missing, and adds a semitone rub when things are in trouble. Lost packets sound as a sagging glitch plus a rim click.
- **Sound:** FM and additive timbres go through a convolution hall reverb and a dotted-eighth echo.
- **MIDI:** bass is on channel 1, pad on 2, the six instrument roles on 3–8 and drums on 10. Each channel gets a GM program, plus pan and reverb CCs. Mod wheel follows jitter. Optional MIDI clock with start/stop lets a DAW follow the tempo. In GarageBand or Logic, pick the "LAN Sonifier" input (or the IAC bus with `--midi-port IAC`).

**Chip style** (`--chip`, or press `c` at any time) is an Undertale-flavoured 8-bit arrangement:

- **Voices:** NES-style 12.5%, 25% and 50% pulse leads with stepped 4-bit envelopes. Each lead has a quieter echo a dotted 8th later.
- **Rhythm section:** a 4-bit triangle bass that bounces octaves when the gateway is quick, and noise-channel kick, snare and hats with little fills.
- **Harmony:** the chord becomes a quick square arpeggio. The default is a minor (aeolian) key at 124 bpm.
- **Lost packets:** the "ouch" pitch-sweep.
- **Look:** a black, white and yellow theme with a ♥ cursor, and a dialogue box where lines like `* The router is trying its best.` type themselves out with a blip per letter.
- **MIDI:** switches to square-lead and synth-bass GM programs.

Hear it in `lansonic_chip_demo.m4a`.

Keys: `space` mutes, `+ -` change tempo, `c` toggles chip style, `m` switches between sequencer and orbit polyrhythms, `d` toggles the pad, `b` toggles the beat, `v V` set volume, `↑↓`/`enter`/`o` select, mute and solo devices, `r` rescans.

## airgap.py — ultrasonic air-gap messenger

```bash
python3 airgap.py                      # chat UI with a live spectrogram waterfall
python3 airgap.py --profile audible    # 1.2–3.6 kHz: audible, but much more robust
python3 airgap.py --profile low --volume 0.3
python3 airgap.py send "meet at noon"  # one-shot
python3 airgap.py send --file key.txt
python3 airgap.py selftest             # simulated noisy room, no audio needed
python3 airgap.py wav "hi" -o hi.wav && python3 airgap.py decode hi.wav
```

Run it on two laptops with the same `--profile` and type. In chat, `/file PATH` sends a file, and files land in `./received/`. `/vol 0.4` sets transmit volume and `/profile` switches band. With `--hear-self` you can test on one machine, because it decodes its own transmissions through the air.

**How a frame is built:**
`[chirp 120 ms] [16-tone training sweep] [header: length+version, RS(11,3)] [body: RS(255,235) blocks] `
with a CRC-32 over the payload.

- **Sync:** a matched filter on the chirp, normalised against in-band energy.
- **Training sweep:** fine-tunes symbol timing and measures each tone's gain through the speaker, room and mic chain.
- **Demodulation:** each symbol is a 16-FSK nibble. Decisions use the high-SNR maximum-likelihood metric for the learned gains, with tracking for slow fading.
- **Error correction:** Reed–Solomon is implemented from scratch (Berlekamp–Massey, Chien search, Forney) and fixes up to 10 bad bytes per block.

| profile | band | raw rate |
|---|---|---|
| `ultra` | 18.0–19.8 kHz | 100 baud |
| `audible` | 1.2–3.6 kHz | 114 baud |
| `low` | 300–1200 Hz | 57 baud |

Small laptop speakers roll off near 20 kHz. If `ultra` struggles, move the machines closer, raise `--volume`, or switch to `audible`. `selftest` runs a simulated room with echoes, treble roll-off, hum, a hand-clap burst and clock drift between the two sound cards.
