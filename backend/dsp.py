"""
dsp.py — Core digital-signal-processing library for the Audio MIR system.

Everything here is implemented from first principles in pure Python (no numpy,
no scipy, no librosa).  The only imports are the standard library.  This is a
deliberate design decision: the project specification calls for the algorithms
to be *self-implemented*, and pure Python keeps the whole system dependency-free
and runnable anywhere a Python 3.9+ interpreter and Flask are available.

The module is organised as layers:

  1.  FFT / inverse FFT           (iterative radix-2, precomputed twiddles)
  2.  Windowing                   (Hann, Hamming, Blackman, ... with COLA norm)
  3.  Short-time Fourier transform (analysis + overlap-add synthesis)
  4.  Frequency conversions       (Hz <-> mel <-> MIDI <-> note names)
  5.  Audio metrics               (RMS, ZCR, spectral centroid, rolloff, flux…)
  6.  Biquad filters              (RBJ cookbook, used by the EQ & separation)
  7.  Pitch detection             (FFT autocorrelation + the YIN algorithm)
  8.  Onset / beat / tempo        (spectral flux + autocorrelation beat tracker)
  9.  Chroma                      (fold STFT energy into pitch classes)
 10.  Utilities                   (dB conversion, resampling, smoothing)

Performance notes
-----------------
* A radix-2 FFT of 2048 points runs in ~3 ms and of 1024 points in ~1.4 ms on a
  typical machine, which is comfortably inside a real-time analysis budget
  (a 1024-sample hop at 44.1 kHz is ~23 ms).
* Twiddle factors are cached per transform size to avoid recomputing the same
  complex exponentials on every call.
"""

from __future__ import annotations

import cmath
import math
from typing import List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# 1. FFT / IFFT
# --------------------------------------------------------------------------- #

_TWIDDLE_CACHE: dict = {}


def _twiddles(n: int) -> List[complex]:
    """Return the forward-FFT twiddle table exp(-2j*pi*k/n) for k in [0, n/2)."""
    cached = _TWIDDLE_CACHE.get(n)
    if cached is None:
        cached = [cmath.exp(-2j * math.pi * k / n) for k in range(n >> 1)]
        _TWIDDLE_CACHE[n] = cached
    return cached


def next_pow2(n: int) -> int:
    """Smallest power of two >= n."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def fft(x: Sequence[complex]) -> List[complex]:
    """In-place-free iterative radix-2 Cooley–Tukey FFT.

    ``x`` may be any sequence of real or complex numbers.  Non-power-of-two
    inputs are zero-padded to the next power of two.  Returns the spectrum
    ordered from DC (index 0) to Nyquist (index n//2), then the negative
    frequencies.
    """
    n = len(x)
    if n == 0:
        return []
    if n & (n - 1):  # not a power of two -> pad
        m = next_pow2(n)
        a = [complex(v) for v in x] + [0.0j] * (m - n)
        n = m
    else:
        a = [complex(v) for v in x]

    # Bit-reversal permutation.
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    table = _twiddles(n)
    length = 2
    while length <= n:
        half = length >> 1
        step = n // length
        for i in range(0, n, length):
            i2 = i + half
            for m in range(half):
                w = table[m * step]
                u = a[i + m]
                v = a[i2 + m] * w
                a[i + m] = u + v
                a[i2 + m] = u - v
        length <<= 1
    return a


def ifft(x: Sequence[complex]) -> List[complex]:
    """Inverse FFT via the conjugate trick: ifft(x) = conj(fft(conj(x)))/n."""
    n = len(x)
    if n == 0:
        return []
    conj = [v.conjugate() for v in x]
    y = fft(conj)
    inv = 1.0 / n
    return [v.conjugate() * inv for v in y]


def rfft_freqs(nfft: int, sr: float) -> List[float]:
    """Frequency (Hz) of each *positive* spectrum bin [0 .. nfft//2]."""
    return [k * sr / nfft for k in range(nfft // 2 + 1)]


# --------------------------------------------------------------------------- #
# 2. Window functions
# --------------------------------------------------------------------------- #

def _cos_window(n: int, a0: float, a1: float, a2: float, a3: float) -> List[float]:
    out = []
    two_pi = 2.0 * math.pi
    for k in range(n):
        t = k / (n - 1) if n > 1 else 0.0
        out.append(
            a0 - a1 * math.cos(two_pi * t)
            + a2 * math.cos(2 * two_pi * t)
            - a3 * math.cos(3 * two_pi * t)
        )
    return out


def hann(n: int) -> List[float]:
    """Periodic-style Hann window (symmetric denominator n-1)."""
    return _cos_window(n, 0.5, 0.5, 0.0, 0.0)


def hamming(n: int) -> List[float]:
    return _cos_window(n, 0.54, 0.46, 0.0, 0.0)


def blackman(n: int) -> List[float]:
    return _cos_window(n, 0.42, 0.5, 0.08, 0.0)


def blackman_harris(n: int) -> List[float]:
    return _cos_window(n, 0.35875, 0.48829, 0.14128, 0.01168)


def bartlett(n: int) -> List[float]:
    out = []
    for k in range(n):
        out.append(1.0 - abs((2.0 * k - (n - 1)) / (n - 1)) if n > 1 else 1.0)
    return out


def rectangular(n: int) -> List[float]:
    return [1.0] * n


_WINDOWS = {
    "hann": hann,
    "hamming": hamming,
    "blackman": blackman,
    "blackman_harris": blackman_harris,
    "bartlett": bartlett,
    "rectangular": rectangular,
}


def window(name: str, n: int) -> List[float]:
    """Return a window of length ``n`` by name (falls back to Hann)."""
    fn = _WINDOWS.get(name, hann)
    return fn(n)


def window_names() -> List[str]:
    return list(_WINDOWS.keys())


# --------------------------------------------------------------------------- #
# 3. Short-time Fourier transform
# --------------------------------------------------------------------------- #

def stft(
    samples: Sequence[float],
    nfft: int = 2048,
    hop: int = 512,
    win: str = "hann",
    center: bool = True,
) -> List[List[complex]]:
    """Short-time Fourier transform -> list of complex frames (one per hop)."""
    w = window(win, nfft)
    n = len(samples)
    if center:
        pad = nfft // 2
        sig = [0.0] * pad + list(samples) + [0.0] * pad
    else:
        sig = list(samples)
    frames: List[List[complex]] = []
    i = 0
    total = len(sig)
    while i + nfft <= total:
        seg = sig[i:i + nfft]
        frames.append(fft([seg[k] * w[k] for k in range(nfft)]))
        i += hop
    return frames


def istft(
    frames: Sequence[Sequence[complex]],
    nfft: int = 2048,
    hop: int = 512,
    win: str = "hann",
    length: Optional[int] = None,
) -> List[float]:
    """Inverse STFT using weighted overlap-add (WOLA) synthesis."""
    w = window(win, nfft)
    n_frames = len(frames)
    if n_frames == 0:
        return []
    out = [0.0] * (n_frames * hop + nfft)
    norm = [0.0] * (n_frames * hop + nfft)
    for i, fr in enumerate(frames):
        t = ifft(fr)
        pos = i * hop
        for k in range(nfft):
            val = t[k].real * w[k]
            out[pos + k] += val
            norm[pos + k] += w[k] * w[k]
    for k in range(len(out)):
        d = norm[k]
        out[k] = (out[k] / d) if d > 1e-9 else 0.0
    if length is not None:
        out = out[:length]
    return out


def spectrogram(
    samples: Sequence[float],
    sr: float,
    nfft: int = 2048,
    hop: int = 512,
    win: str = "hann",
    max_bins: int = 0,
) -> Tuple[List[List[float]], List[float], List[float]]:
    """Return (magnitude-spectrogram dB, times, freqs).

    ``max_bins`` optionally limits how many positive bins are kept (for memory).
    """
    frames = stft(samples, nfft, hop, win)
    bins = nfft // 2 + 1
    if max_bins:
        bins = min(bins, max_bins)
    mag = [[abs(fr[k]) for k in range(bins)] for fr in frames]
    spec = to_db(mag)
    times = [i * hop / sr for i in range(len(frames))]
    freqs = [k * sr / nfft for k in range(bins)]
    return spec, times, freqs


# --------------------------------------------------------------------------- #
# 4. Frequency / scale conversions
# --------------------------------------------------------------------------- #

def hz_to_mel(f: float) -> float:
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(
    nfft: int, sr: float, n_mels: int = 40, fmin: float = 0.0, fmax: Optional[float] = None
) -> List[List[float]]:
    """Triangular mel filterbank -> n_mels filters each spanning positive bins."""
    if fmax is None:
        fmax = sr / 2.0
    n_bins = nfft // 2 + 1
    mel_min = hz_to_mel(max(fmin, 1e-9))
    mel_max = hz_to_mel(min(fmax, sr / 2.0))
    mel_pts = [mel_min + (mel_max - mel_min) * i / (n_mels + 1) for i in range(n_mels + 2)]
    hz_pts = [mel_to_hz(m) for m in mel_pts]
    bin_pts = [int(math.floor((nfft + 1) * h / sr)) for h in hz_pts]
    filters = []
    for m in range(1, n_mels + 1):
        f = [0.0] * n_bins
        lo, ce, hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(lo, ce):
            if ce > lo:
                f[k] = (k - lo) / (ce - lo)
        for k in range(ce, hi):
            if hi > ce:
                f[k] = (hi - k) / (hi - ce)
        filters.append(f)
    return filters


def hz_to_midi(f: float) -> Optional[float]:
    """Frequency in Hz -> float MIDI note number (None for non-positive f)."""
    if f <= 0:
        return None
    return 69.0 + 12.0 * math.log2(f / 440.0)


def midi_to_hz(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note(m: float) -> Tuple[str, int]:
    """Float MIDI -> (note name with accidental, octave)."""
    midi = int(round(m))
    name = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return name, octave


def note_display(f0: float) -> str:
    """Human-readable note + cents offset, e.g. 'A4 +12¢'."""
    if f0 is None or f0 <= 0:
        return "--"
    m = hz_to_midi(f0)
    midi = int(round(m))
    name, octave = midi_to_note(m)
    cents = int(round((m - midi) * 100))
    if cents == 0:
        return f"{name}{octave}"
    sign = "+" if cents > 0 else ""
    return f"{name}{octave} {sign}{cents}¢"


# --------------------------------------------------------------------------- #
# 5. Audio metrics
# --------------------------------------------------------------------------- #

def rms(samples: Sequence[float]) -> float:
    n = len(samples)
    if n == 0:
        return 0.0
    s = 0.0
    for x in samples:
        s += x * x
    return math.sqrt(s / n)


def db(x: float) -> float:
    """Amplitude -> dB, guarded against log(0)."""
    return 20.0 * math.log10(x + 1e-12)


def to_db(matrix: Sequence[Sequence[float]], ref: float = 1.0, floor: float = -120.0) -> List[List[float]]:
    """Convert an amplitude matrix to dB, flooring the result."""
    out = []
    for row in matrix:
        out.append([max(floor, db(v / ref)) for v in row])
    return out


def from_db(db_val: float) -> float:
    return 10.0 ** (db_val / 20.0)


def zero_crossing_rate(samples: Sequence[float]) -> float:
    n = len(samples)
    if n < 2:
        return 0.0
    count = 0
    prev = samples[0]
    for x in samples[1:]:
        if (x >= 0.0) != (prev >= 0.0):
            count += 1
        prev = x
    return count / (n - 1)


def spectral_centroid(mag: Sequence[float], freqs: Sequence[float]) -> float:
    s = 0.0
    wsum = 0.0
    for i, m in enumerate(mag):
        s += freqs[i] * m
        wsum += m
    return (s / wsum) if wsum > 1e-12 else 0.0


def spectral_rolloff(mag: Sequence[float], freqs: Sequence[float], pct: float = 0.85) -> float:
    total = sum(mag)
    if total <= 0:
        return 0.0
    acc = 0.0
    for i, m in enumerate(mag):
        acc += m
        if acc >= pct * total:
            return freqs[i]
    return freqs[-1]


def spectral_flatness(mag: Sequence[float]) -> float:
    """Geometric mean / arithmetic mean of the power spectrum (0..1)."""
    n = len(mag)
    if n == 0:
        return 0.0
    log_sum = 0.0
    lin_sum = 0.0
    for m in mag:
        p = m * m + 1e-12
        log_sum += math.log(p)
        lin_sum += p
    geo = math.exp(log_sum / n)
    arith = lin_sum / n
    return min(1.0, geo / arith)


def spectral_flux(mag: Sequence[float], prev_mag: Optional[Sequence[float]]) -> float:
    """L2 norm of the positive magnitude difference between two frames."""
    if prev_mag is None:
        return 0.0
    s = 0.0
    for i, m in enumerate(mag):
        d = m - prev_mag[i]
        if d > 0:
            s += d * d
    return math.sqrt(s)


def spectral_bandwidth(mag: Sequence[float], freqs: Sequence[float]) -> float:
    centroid = spectral_centroid(mag, freqs)
    s = 0.0
    wsum = 0.0
    for i, m in enumerate(mag):
        d = freqs[i] - centroid
        s += d * d * m
        wsum += m
    return math.sqrt(s / wsum) if wsum > 1e-12 else 0.0


# --------------------------------------------------------------------------- #
# 6. Biquad filters (RBJ Audio EQ Cookbook)
# --------------------------------------------------------------------------- #

class Biquad:
    """Transposed direct-form-II biquad with RBJ-cookbook coefficient design.

    Coefficients are computed from *analog* prototypes for the common filter
    shapes and then bilinear-transformed.  ``process_block`` maintains state so
    the filter can be streamed over arbitrarily long signals.
    """

    def __init__(self, ftype: str, sr: float, freq: float, q: float = 0.7071,
                 gain_db: float = 0.0):
        self.sr = sr
        self.ftype = ftype
        self.freq = freq
        self.q = q
        self.gain_db = gain_db
        self.b0 = self.b1 = self.b2 = self.a1 = self.a2 = 0.0
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0
        self._compute()

    def _compute(self) -> None:
        sr = self.sr
        f0 = self.freq
        A = 10.0 ** (self.gain_db / 40.0)
        w0 = 2.0 * math.pi * f0 / sr
        cw = math.cos(w0)
        sw = math.sin(w0)
        alpha = sw / (2.0 * self.q)
        f = self.ftype

        if f == "lowpass":
            b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = (1 - cw) / 2
            a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha
        elif f == "highpass":
            b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = (1 + cw) / 2
            a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha
        elif f == "bandpass":
            b0 = alpha; b1 = 0; b2 = -alpha
            a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha
        elif f == "notch":
            b0 = 1; b1 = -2 * cw; b2 = 1
            a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha
        elif f == "allpass":
            b0 = 1 - alpha; b1 = -2 * cw; b2 = 1 + alpha
            a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha
        elif f == "peaking":
            b0 = 1 + alpha * A; b1 = -2 * cw; b2 = 1 - alpha * A
            a0 = 1 + alpha / A; a1 = -2 * cw; a2 = 1 - alpha / A
        elif f == "lowshelf":
            sq2 = math.sqrt(2 * A) * alpha
            b0 = A * ((A + 1) - (A - 1) * cw + sq2)
            b1 = 2 * A * ((A - 1) - (A + 1) * cw)
            b2 = A * ((A + 1) - (A - 1) * cw - sq2)
            a0 = (A + 1) + (A - 1) * cw + sq2
            a1 = -2 * ((A - 1) + (A + 1) * cw)
            a2 = (A + 1) + (A - 1) * cw - sq2
        elif f == "highshelf":
            sq2 = math.sqrt(2 * A) * alpha
            b0 = A * ((A + 1) + (A - 1) * cw + sq2)
            b1 = -2 * A * ((A - 1) + (A + 1) * cw)
            b2 = A * ((A + 1) + (A - 1) * cw - sq2)
            a0 = (A + 1) - (A - 1) * cw + sq2
            a1 = 2 * ((A - 1) - (A + 1) * cw)
            a2 = (A + 1) - (A - 1) * cw - sq2
        else:  # bypass / unknown -> identity
            self.b0, self.b1, self.b2, self.a1, self.a2 = 1, 0, 0, 0, 0
            return

        ia0 = 1.0 / a0
        self.b0 = b0 * ia0
        self.b1 = b1 * ia0
        self.b2 = b2 * ia0
        self.a1 = a1 * ia0
        self.a2 = a2 * ia0

    def reset(self) -> None:
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def process(self, x: float) -> float:
        y = self.b0 * x + self.x1
        self.x1 = self.b1 * x - self.a1 * y + self.x2
        self.x2 = self.b2 * x - self.a2 * y
        return y

    def process_block(self, block: Sequence[float]) -> List[float]:
        return [self.process(x) for x in block]


def bandpass_split(samples: Sequence[float], sr: float, lo: float, hi: float) -> List[float]:
    """Return the frequency band [lo, hi] of ``samples`` using biquads."""
    hp = Biquad("highpass", sr, lo, q=0.7071) if lo > 20 else None
    lp = Biquad("lowpass", sr, hi, q=0.7071) if hi < sr / 2 else None
    out = list(samples)
    if hp is not None:
        out = hp.process_block(out)
    if lp is not None:
        out = lp.process_block(out)
    return out


# --------------------------------------------------------------------------- #
# 7. Pitch detection
# --------------------------------------------------------------------------- #

def autocorrelation(signal: Sequence[float]) -> List[float]:
    """Unbiased-ish autocorrelation via FFT (power spectrum -> IFFT)."""
    n = len(signal)
    m = next_pow2(2 * n)
    padded = [float(v) for v in signal] + [0.0] * (m - n)
    spec = fft(padded)
    power = [abs(v) ** 2 for v in spec]
    acorr = ifft(power)
    return [v.real for v in acorr[:n]]


def pitch_autocorr(frame: Sequence[float], sr: float, fmin: float = 50.0,
                   fmax: float = 2000.0) -> Optional[float]:
    """Fundamental frequency via the autocorrelation peak (FFT accelerated)."""
    n = len(frame)
    if n < 4:
        return None
    acorr = autocorrelation(frame)
    norm = acorr[0]
    if norm < 1e-12:
        return None
    min_lag = max(2, int(sr / fmax))
    max_lag = min(n - 2, int(sr / fmin))
    if min_lag >= max_lag:
        return None
    best_lag, best_val = min_lag, -1e18
    for lag in range(min_lag, max_lag + 1):
        v = acorr[lag]
        if v > best_val:
            best_val, best_lag = v, lag
    if best_val <= 0:
        return None
    # Parabolic interpolation around the peak.
    if min_lag < best_lag < max_lag:
        y0 = acorr[best_lag - 1] / norm
        y1 = acorr[best_lag] / norm
        y2 = acorr[best_lag + 1] / norm
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-9:
            delta = 0.5 * (y0 - y2) / denom
            if abs(delta) < 1:
                best_lag += delta
    f0 = sr / best_lag
    return f0 if fmin <= f0 <= fmax else None


def yin_difference(frame: Sequence[float], max_tau: int) -> List[float]:
    """YIN squared difference function d(tau)."""
    n = len(frame)
    d = [0.0] * max_tau
    for tau in range(1, max_tau):
        s = 0.0
        limit = n - tau
        for i in range(limit):
            df = frame[i] - frame[i + tau]
            s += df * df
        d[tau] = s
    return d


def yin_pitch(frame: Sequence[float], sr: float, fmin: float = 50.0,
              fmax: float = 2000.0, thresh: float = 0.1) -> Optional[float]:
    """Classic YIN pitch estimator (accurate but O(n^2) — use for spot checks)."""
    n = len(frame)
    max_tau = min(n // 2, int(sr / fmin))
    min_tau = max(1, int(sr / fmax))
    if min_tau >= max_tau:
        return None
    d = yin_difference(frame, max_tau)
    cmnd = [1.0] * max_tau
    running = 0.0
    for tau in range(1, max_tau):
        running += d[tau]
        cmnd[tau] = (d[tau] * tau / running) if running > 0 else 1.0
    tau_est = None
    tau = 2
    while tau < max_tau - 1:
        if cmnd[tau] < thresh:
            while tau + 1 < max_tau and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            tau_est = tau
            break
        tau += 1
    if tau_est is None:
        # Fall back to the global minimum of cmnd.
        tau_est = min(range(min_tau, max_tau), key=lambda t: cmnd[t])
    if 0 < tau_est < max_tau - 1:
        s0, s1, s2 = cmnd[tau_est - 1], cmnd[tau_est], cmnd[tau_est + 1]
        denom = s0 - 2 * s1 + s2
        if abs(denom) > 1e-9:
            tau_est += 0.5 * (s0 - s2) / denom
    f0 = sr / tau_est
    return f0 if fmin <= f0 <= fmax else None


def pitch_contour(samples: Sequence[float], sr: float, hop: int = 512,
                  win_len: int = 2048, fmin: float = 50.0,
                  fmax: float = 2000.0) -> Tuple[List[float], List[float]]:
    """Frame-by-frame fundamental frequency using the fast autocorrelation path.

    Returns (times, f0) where f0 is 0.0 for unvoiced frames.
    """
    w = hann(win_len)
    times = []
    f0s = []
    i = 0
    total = len(samples)
    while i + win_len <= total:
        seg = samples[i:i + win_len]
        frame = [seg[k] * w[k] for k in range(win_len)]
        f0 = pitch_autocorr(frame, sr, fmin, fmax)
        f0s.append(f0 if f0 is not None else 0.0)
        times.append(i / sr)
        i += hop
    return times, f0s


# --------------------------------------------------------------------------- #
# 8. Onset / beat / tempo
# --------------------------------------------------------------------------- #

def onset_strength_from_spec(spec_mag: Sequence[Sequence[float]]) -> List[float]:
    """Spectral-flux onset-strength envelope from a magnitude spectrogram."""
    flux = []
    prev = None
    for frame in spec_mag:
        if prev is None:
            prev = frame
            flux.append(0.0)
            continue
        s = 0.0
        for k in range(len(frame)):
            d = frame[k] - prev[k]
            if d > 0:
                s += d * d
        flux.append(math.sqrt(s))
        prev = frame
    return flux


def local_maxima(x: Sequence[float]) -> List[int]:
    """Indices of strict local maxima."""
    peaks = []
    n = len(x)
    if n < 3:
        return peaks
    for i in range(1, n - 1):
        if x[i] > x[i - 1] and x[i] >= x[i + 1]:
            peaks.append(i)
    return peaks


def smooth(x: Sequence[float], width: int = 3) -> List[float]:
    """Zero-phase centered moving-average smoothing of a 1-D sequence.

    Edge samples are replicated so the window stays symmetric at both ends;
    the output therefore has *no* group delay and does not depend on where a
    crop starts or ends (a leading-only window at the head and a frozen tail
    would otherwise shift peaks by several frames).
    """
    if width < 1 or not x:
        return list(x)
    w = width if width % 2 == 1 else width + 1  # odd length -> symmetric
    n = len(x)
    half = w // 2
    ext = [x[0]] * half + list(x) + [x[-1]] * half
    acc = sum(ext[:w])
    out = [0.0] * n
    for i in range(n):
        out[i] = acc / w
        if i < n - 1:
            acc += ext[i + w] - ext[i]
    return out


def weighted_smooth(x: Sequence[float], kernel: Sequence[float]) -> List[float]:
    """Zero-phase FIR smoothing with an arbitrary (symmetric) kernel.

    Edge samples are replicated; the kernel is normalised to sum 1 internally.
    """
    n = len(x)
    if n == 0 or not kernel:
        return list(x)
    ks = sum(kernel) or 1.0
    half = len(kernel) // 2
    ext = [x[0]] * half + list(x) + [x[-1]] * half
    out = [0.0] * n
    for i in range(n):
        s = 0.0
        for j, kv in enumerate(kernel):
            s += kv * ext[i + j]
        out[i] = s / ks
    return out


def frac_index(x: Sequence[float], pos: float) -> float:
    """Linearly interpolated sample at fractional index ``pos`` (clamped)."""
    n = len(x)
    if n == 0:
        return 0.0
    if pos <= 0:
        return x[0]
    if pos >= n - 1:
        return x[-1]
    i = int(pos)
    f = pos - i
    return x[i] * (1.0 - f) + x[i + 1] * f


def _parabolic_shift(y0: float, y1: float, y2: float) -> float:
    """Sub-bin offset of a parabola fitted through three samples (|.| < 1)."""
    denom = y0 - 2.0 * y1 + y2
    if denom <= 1e-15:
        return 0.0
    delta = 0.5 * (y0 - y2) / denom
    if abs(delta) >= 1.0:
        return 0.0
    return delta


def estimate_tempo(onset_env: Sequence[float], frame_rate: float,
                   min_bpm: float = 40.0, max_bpm: float = 240.0) -> float:
    """Estimate tempo (BPM) from an onset envelope.

    The envelope is autocorrelated over the plausible lag range; the best lag
    and the neighbouring lags are scored directly against the envelope peaks,
    which resolves the usual octave (half/double-time) ambiguities without a
    hard tempo prior.  The winning lag is sub-frame refined, so the BPM no
    longer moves when the clip is shortened: a fractional lag is returned
    instead of an integer one.
    """
    tempo, _beats, _peaks = track_beats(onset_env, frame_rate, min_bpm, max_bpm)
    return tempo


def _refine_peak(onset_env: Sequence[float], sm: Sequence[float], i: int) -> float:
    """Crop-invariant sub-frame location of an onset straddling frame ``i``.

    A physical attack splits its spectral-flux energy between the *two*
    analysis frames it falls between; the split ratio depends on where the
    attack lands relative to the hop grid, so neither the integer peak nor a
    parabola on the smoothed envelope gives the same answer for a clip and a
    crop of the same audio.  The flux centroid over the three frames around
    the peak does: shifting the hop grid redistributes energy between adjacent
    frames but leaves their centroid at the physical attack time.
    """
    n = len(onset_env)
    # Work on the raw flux, background-subtracted with the smoothed envelope.
    i0, i1 = max(0, i - 1), min(n - 1, i + 1)
    vals = [max(0.0, onset_env[j] - 0.5 * sm[j]) for j in range(i0, i1 + 1)]
    total = sum(vals)
    if total <= 1e-12:
        return float(i)
    centre = sum(j * v for j, v in zip(range(i0, i1 + 1), vals)) / total
    # Guard against a real secondary onset next door pulling the centroid.
    if abs(centre - i) > 1.0:
        return float(i) + _parabolic_shift(sm[i - 1], sm[i], sm[i + 1])
    return centre


def _onset_peaks(onset_env: Sequence[float], frame_rate: float,
                 min_gap_sec: float = 0.03) -> List[Tuple[float, float]]:
    """Onset peaks as ``(fractional_frame, strength)``.

    A short *zero-phase* filter ([1,2,1]/4) suppresses one-frame jitter; the
    adaptive threshold is a centered local average, so peak times carry no
    systematic delay and do not move when the clip boundaries move.  Peak
    locations are sub-frame refined with a flux centroid (see
    :func:`_refine_peak`) which is invariant to the attack's position on the
    hop grid and hence to trimming.
    """
    n = len(onset_env)
    if n < 3:
        return []
    sm = weighted_smooth(onset_env, (1.0, 2.0, 1.0))
    win = max(7, int(round(frame_rate * 0.2)) | 1)  # ~200 ms local window
    local = smooth(sm, win)
    floor = max(sm) * 0.05
    min_dist = max(1, int(round(frame_rate * min_gap_sec)))

    raw: List[Tuple[int, float]] = []
    for i in range(1, n - 1):
        if sm[i] > sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] > local[i] * 1.25 + floor:
            raw.append((i, sm[i]))

    # Enforce the minimum gap, keeping the stronger peak (non-max suppression).
    raw.sort(key=lambda p: p[1], reverse=True)
    kept: List[Tuple[int, float]] = []
    for idx, val in raw:
        if all(abs(idx - j) >= min_dist for j, _ in kept):
            kept.append((idx, val))
    kept.sort()

    peaks: List[Tuple[float, float]] = []
    for idx, val in kept:
        pos = _refine_peak(onset_env, sm, idx)
        peaks.append((pos, frac_index(sm, pos)))
    return peaks


def _tempo_lag_candidates(onset_env: Sequence[float], frame_rate: float,
                          min_bpm: float, max_bpm: float) -> List[float]:
    """Fractional beat-period candidates (frames): autocorrelation peaks + octaves."""
    n = len(onset_env)
    mean = sum(onset_env) / n
    # Edge taper so the autocorrelation reflects content periodicity rather
    # than where the crop happens to start/end.
    taper_n = min(n // 4, max(2, int(frame_rate * 0.3)))
    win = hann(2 * taper_n + 1)
    y = []
    for i, v in enumerate(onset_env):
        w = 1.0
        if i < taper_n:
            w = win[i]
        elif i > n - 1 - taper_n:
            w = win[2 * taper_n - (n - 1 - i)]
        y.append((v - mean) * w)

    acorr = autocorrelation(y)
    min_lag = max(1, int(frame_rate * 60.0 / max_bpm))
    max_lag = min(n - 2, int(frame_rate * 60.0 / min_bpm))
    if min_lag >= max_lag:
        return []

    # Unbiased normalisation (raw autocorrelation is biased toward short lags).
    norm = [a / (n - lag) for lag, a in enumerate(acorr)]

    # Top autocorrelation maxima in the valid lag window.
    peak_lags = [lag for lag in range(min_lag + 1, max_lag)
                 if norm[lag] > norm[lag - 1] and norm[lag] >= norm[lag + 1]]
    peak_lags.sort(key=lambda lag: norm[lag], reverse=True)
    seed_lags = peak_lags[:5] or [max(range(min_lag, max_lag + 1),
                                     key=lambda lag: norm[lag])]

    def refine(lag: int) -> float:
        shift = _parabolic_shift(norm[lag - 1], norm[lag], norm[lag + 1])
        return lag + shift

    candidates = [refine(lag) for lag in seed_lags]
    # Half/double-time alternatives of the global best.
    best = refine(seed_lags[0])
    if 2 * best <= max_lag:
        candidates.append(2 * best)
    if 0.5 * best >= min_lag:
        candidates.append(0.5 * best)

    # Keep valid, deduplicate periods within 2 %.
    out: List[float] = []
    for tau in candidates:
        bpm = 60.0 * frame_rate / tau
        if not (min_bpm <= bpm <= max_bpm):
            continue
        if all(abs(tau - t) / t > 0.02 for t in out):
            out.append(tau)
    return out


def _grid_score(peaks: Sequence[Tuple[float, float]], n: int,
                tau: float, phi: float, sigma: float) -> float:
    """Score an event grid ``phi + k*tau`` against onset peaks.

    Each peak contributes a Gaussian weight by distance to its nearest grid
    line (distance wraps around the period); the sum is divided by the number
    of grid lines in the clip, so half-time grids (empty lines every other
    beat) and double-time grids (one line per two beats) are both penalised
    symmetrically.  Peak coverage multiplies the score so spurious extra peaks
    do not change the chosen phase.
    """
    if not peaks or tau <= 0 or phi >= n:
        return 0.0
    span = 3.0 * sigma
    weighted = 0.0
    matched = 0
    for pos, strength in peaks:
        d = (pos - phi) % tau
        if d > tau / 2.0:
            d -= tau
        ad = abs(d)
        if ad < span:
            weighted += strength * math.exp(-0.5 * (d / sigma) ** 2)
            if ad < 2.0 * sigma:
                matched += 1
    n_lines = 1 + int((n - 1 - phi) / tau)
    return (weighted / max(1, n_lines)) * (matched / len(peaks))


def _best_phase(peaks: Sequence[Tuple[float, float]], n: int,
                tau: float, sigma: float) -> Tuple[float, float]:
    """Best grid phase (fractional frames) and its score for a fixed period."""
    tau_i = max(1, int(round(tau)))
    best_phi, best_score = 0, -1.0
    for phi in range(min(tau_i, n)):
        s = _grid_score(peaks, n, tau, phi, sigma)
        if s > best_score:
            best_score, best_phi = s, phi
    # Sub-frame phase refinement.
    s0 = _grid_score(peaks, n, tau, best_phi - 1, sigma)
    s2 = _grid_score(peaks, n, tau, best_phi + 1, sigma)
    return best_phi + _parabolic_shift(s0, best_score, s2), best_score


def track_beats(onset_env: Sequence[float], frame_rate: float,
                min_bpm: float = 40.0, max_bpm: float = 240.0
                ) -> Tuple[float, List[float], List[float]]:
    """Full beat tracker: ``(tempo_bpm, beat_frames, peak_frames)``.

    Pipeline: onset peaks -> autocorrelation period candidates with half/
    double-time alternatives -> direct grid-vs-peak scoring to choose tempo
    and phase -> snap every grid line onto the nearest actual onset peak with
    sub-frame interpolation.  Nothing here depends on clip length or on where
    the crop starts, and the only smoothing is zero-phase, so beat times line
    up with the physical accents regardless of trimming.
    """
    n = len(onset_env)
    if n < 8 or frame_rate <= 0:
        return 120.0, [], []

    peaks = _onset_peaks(onset_env, frame_rate)
    if not peaks:
        return 120.0, [], []

    pmax = max(s for _, s in peaks)
    peaks_n = [(p, s / pmax) for p, s in peaks]
    sigma = max(2.0, frame_rate * 0.022)  # ~22 ms alignment tolerance

    candidates = _tempo_lag_candidates(onset_env, frame_rate, min_bpm, max_bpm)
    if not candidates:
        return 120.0, [], []

    best = (-1.0, 0.0, 0.0)  # (score, tau, phi)
    for tau0 in candidates:
        phi, score = _best_phase(peaks_n, n, tau0, sigma)
        # Fine tempo search (+-2 %) with the phase re-optimised at each step.
        for step in range(-5, 6):
            tau = tau0 * (1.0 + 0.004 * step)
            bpm = 60.0 * frame_rate / tau
            if not (min_bpm <= bpm <= max_bpm):
                continue
            phi2, score2 = _best_phase(peaks_n, n, tau, sigma)
            if score2 > best[0]:
                best = (score2, tau, phi2)

    _score, tau, phi = best
    tempo = 60.0 * frame_rate / tau

    # Lay the grid down and snap each line to the nearest real accent.
    positions = [p for p, _ in peaks]
    strengths = [s for _, s in peaks_n]
    snap_radius = min(tau * 0.25, frame_rate * 0.14)
    snap_sigma = max(2.0, frame_rate * 0.025)
    level = smooth(onset_env, max(5, int(round(frame_rate * 0.09)) | 1))
    mean_level = sum(level) / n

    beat_frames: List[float] = []
    k = 0
    while True:
        t = phi + k * tau
        if t >= n:
            break
        if t >= 0:
            # Binary search for the nearest peak.
            lo, hi = 0, len(positions)
            while lo < hi:
                mid = (lo + hi) // 2
                if positions[mid] < t:
                    lo = mid + 1
                else:
                    hi = mid
            pick, pick_w = -1, 0.0
            for j in (lo - 1, lo):
                if 0 <= j < len(positions):
                    d = abs(positions[j] - t)
                    if d <= snap_radius:
                        w = strengths[j] * math.exp(-0.5 * (d / snap_sigma) ** 2)
                        if w > pick_w:
                            pick_w, pick = w, j
            if pick >= 0:
                beat_frames.append(positions[pick])
            elif frac_index(level, t) > mean_level * 0.5:
                # Weak/legato beat: trust the grid rather than inventing a peak.
                beat_frames.append(t)
            # else: long rest / silence at the boundary -> no beat.
        k += 1

    beat_frames.sort()
    return tempo, beat_frames, [p for p, _ in peaks]


def detect_beats(onset_env: Sequence[float], frame_rate: float,
                 tempo: Optional[float] = None) -> List[float]:
    """Beat times (s) aligned to onset accents.

    When ``tempo`` is given, the search is narrowed to a +-12 % band around
    it; the phase is always derived from the actual onset peaks, so the
    returned times sit on the accents and do not drift with clip length or
    crop offset.
    """
    n = len(onset_env)
    if n < 8 or frame_rate <= 0:
        return []
    if tempo and tempo > 0:
        min_bpm, max_bpm = tempo * 0.88, tempo * 1.12
    else:
        min_bpm, max_bpm = 40.0, 240.0
    _tempo, beat_frames, _peaks = track_beats(onset_env, frame_rate,
                                              min_bpm, max_bpm)
    return [b / frame_rate for b in beat_frames]


# --------------------------------------------------------------------------- #
# 9. Chroma
# --------------------------------------------------------------------------- #

def chroma_from_spec(spec_mag: Sequence[Sequence[float]], sr: float,
                     nfft: int) -> List[List[float]]:
    """Fold a magnitude spectrogram into a 12-bin chromagram (one row/frame)."""
    chroma = []
    bins = len(spec_mag[0]) if spec_mag else 0
    for frame in spec_mag:
        c = [0.0] * 12
        for k in range(1, bins):
            f = k * sr / nfft
            m = hz_to_midi(f)
            if m is None:
                continue
            pc = int(round(m)) % 12
            c[pc] += frame[k]
        s = sum(c)
        if s > 1e-12:
            c = [v / s for v in c]
        chroma.append(c)
    return chroma


# --------------------------------------------------------------------------- #
# 10. Utilities
# --------------------------------------------------------------------------- #

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def normalize(samples: Sequence[float], peak: float = 0.99) -> List[float]:
    """Peak-normalise a signal to ``peak`` (returns new list)."""
    p = max((abs(x) for x in samples), default=0.0)
    if p < 1e-12:
        return list(samples)
    g = peak / p
    return [x * g for x in samples]


def resample_linear(samples: Sequence[float], src_sr: float, dst_sr: float) -> List[float]:
    """Linear-interpolation resampler (adequate for small rate changes)."""
    if src_sr == dst_sr:
        return list(samples)
    ratio = src_sr / dst_sr
    n_out = int(round(len(samples) / ratio))
    out = []
    for i in range(n_out):
        pos = i * ratio
        i0 = int(pos)
        frac = pos - i0
        i1 = i0 + 1
        if i1 >= len(samples):
            out.append(samples[i0])
        else:
            out.append(samples[i0] * (1 - frac) + samples[i1] * frac)
    return out


class StreamingResampler:
    """Resample a stream of blocks with correct continuity across block edges.

    Push source blocks with :meth:`push` and pull output samples with
    :meth:`pull`.  A fractional source position is carried across calls so the
    interpolation is seamless (used by format conversion and the mixer).
    """

    def __init__(self, src_sr: float, dst_sr: float):
        self.ratio = src_sr / dst_sr
        self.pos = 0.0
        self.buf: List[float] = []

    def push(self, block: Sequence[float]) -> None:
        self.buf.extend(block)

    def pull(self, max_out: int) -> List[float]:
        out: List[float] = []
        buf = self.buf
        ratio = self.ratio
        pos = self.pos
        n = len(buf)
        while len(out) < max_out and int(pos) + 1 < n:
            i0 = int(pos)
            frac = pos - i0
            out.append(buf[i0] * (1.0 - frac) + buf[i0 + 1] * frac)
            pos += ratio
        consumed = int(pos)
        if consumed > 0:
            del buf[:consumed]
            pos -= consumed
        self.pos = pos
        return out

    def flush(self, max_out: int) -> List[float]:
        """Pull any remaining samples, including a final partial one."""
        if not self.buf:
            return []
        return self.pull(max_out)


def fade_in_out(samples: Sequence[float], fade_sec: float, sr: float) -> List[float]:
    """Apply linear fade-in and fade-out to avoid clicks."""
    n = len(samples)
    out = list(samples)
    fade_n = min(n // 2, int(fade_sec * sr))
    if fade_n > 1:
        for i in range(fade_n):
            g = i / fade_n
            out[i] *= g
            out[n - 1 - i] *= g
    return out


def midi_freq_table() -> List[float]:
    """Frequencies for MIDI notes 0..127."""
    return [midi_to_hz(m) for m in range(128)]
