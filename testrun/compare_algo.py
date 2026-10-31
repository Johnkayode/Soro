import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import yaml
from scipy.io import wavfile

from soro.suppressor import build_suppressor, suppression_algorithms, SuppressionAlgorithm

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_IN_DIR = _SCRIPT_DIR / "samples"
_DEFAULT_OUT_DIR = _SCRIPT_DIR / "bench_out"
_EPS = 1e-8


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare suppression algorithms using the same input WAV."
    )
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--samplerate", type=int, default=None)
    p.add_argument("--frame_ms", type=int, default=None)
    p.add_argument("--calib_sec", type=float, default=2.0)
    p.add_argument("--eval_sec", type=float, default=12.0)
    p.add_argument("--device_in", type=str, default=None)
    p.add_argument("--input_wav", type=str, required=True, help="WAV input; first channel is used if multichannel")
    p.add_argument("--out_dir", type=str, default=str(_DEFAULT_OUT_DIR))
    return p.parse_args()


def load_config(path: str):
    cfg = {
        "samplerate": 16000,
        "frame_ms": 20,
        "highpass_hz": 80,
        "noise_beta": 1.5,
        "noise_floor": 0.02,
        "ema_alpha": 0.96,
        "speech_alpha": 0.995,
        "noise_gate_db": 6.0,
        "gain_smooth": 0.8,
        "vad_snr_db": 3.0,
        "vad_speech_ratio": 0.25,
        "vad_hangover_frames": 6,
        "dd_alpha": 0.98,
        "omlsa_q": 0.3,
        "omlsa_min_gain_db": -20.0,
        "min_track_frames": 50,
        "device_in": "default",
    }
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    except FileNotFoundError:
        pass
    return cfg


def rms_db(x: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + _EPS))


def frame_rms(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    out = []
    for i in range(0, max(1, len(x) - frame_len + 1), hop):
        fr = x[i : i + frame_len]
        if len(fr) < frame_len:
            fr = np.pad(fr, (0, frame_len - len(fr)))
        out.append(np.sqrt(np.mean(fr * fr)))
    return np.asarray(out, dtype=np.float32)


def save_wav(path: Path, sr: int, x: np.ndarray):
    y = np.clip(x, -1.0, 1.0)
    wavfile.write(path, sr, (y * 32767.0).astype(np.int16))


def run_algo(name: SuppressionAlgorithm, signal: np.ndarray, cfg: dict, calib_samples: int):
    ns = build_suppressor(
        algo=name,
        sr=int(cfg["samplerate"]),
        frame_ms=int(cfg["frame_ms"]),
        noise_floor=cfg["noise_floor"],
        ema_alpha=cfg["ema_alpha"],
        speech_alpha=cfg["speech_alpha"],
        noise_gate_db=cfg["noise_gate_db"],
        gain_smooth=cfg["gain_smooth"],
        highpass_hz=cfg["highpass_hz"],
        vad_snr_db=cfg["vad_snr_db"],
        vad_speech_ratio=cfg["vad_speech_ratio"],
        vad_hangover_frames=cfg["vad_hangover_frames"],
        noise_beta=cfg["noise_beta"],
        dd_alpha=cfg["dd_alpha"],
        omlsa_q=cfg["omlsa_q"],
        omlsa_min_gain_db=cfg["omlsa_min_gain_db"],
        min_track_frames=cfg["min_track_frames"],
    )
    hop = ns.hop

    cal = signal[:calib_samples]
    eva = signal[calib_samples:]

    for i in range(0, len(cal), hop):
        c = cal[i : i + hop]
        if len(c) < hop:
            c = np.pad(c, (0, hop - len(c)))
        ns.calibrate_noise(c.astype(np.float32))

    y = np.zeros_like(eva, dtype=np.float32)
    w = 0
    for i in range(0, len(eva), hop):
        c = eva[i : i + hop]
        n = len(c)
        if n < hop:
            c = np.pad(c, (0, hop - n))
        o = ns.process(c.astype(np.float32))
        y[w : w + n] = o[:n]
        w += n
    return y, hop


def summarize_metrics(x: np.ndarray, y: np.ndarray, frame_len: int, hop: int):
    in_db = rms_db(x)
    out_db = rms_db(y)
    total_cut_db = in_db - out_db

    in_fr = frame_rms(x, frame_len, hop)
    out_fr = frame_rms(y, frame_len, hop)

    low_n = max(1, int(0.2 * len(in_fr)))
    high_n = max(1, int(0.2 * len(in_fr)))
    idx_low = np.argsort(in_fr)[:low_n]
    idx_high = np.argsort(in_fr)[-high_n:]

    noise_cut_db = rms_db(in_fr[idx_low]) - rms_db(out_fr[idx_low])
    speech_attn_db = rms_db(in_fr[idx_high]) - rms_db(out_fr[idx_high])

    return {
        "in_dbfs": in_db,
        "out_dbfs": out_db,
        "total_cut_db": total_cut_db,
        "noise_cut_est_db": noise_cut_db,
        "speech_attn_est_db": speech_attn_db,
    }


def read_input_wav(path: Path, target_sr: int):
    sr, data = wavfile.read(path)
    x = data.astype(np.float32)
    if data.ndim > 1:
        x = x[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        scale = np.iinfo(data.dtype).max
        x /= float(scale)
    if sr != target_sr:
        raise ValueError(f"input_wav sample rate is {sr}, expected {target_sr}")
    return x


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.samplerate is not None:
        cfg["samplerate"] = args.samplerate
    if args.frame_ms is not None:
        cfg["frame_ms"] = args.frame_ms
    if args.device_in is not None:
        cfg["device_in"] = args.device_in

    sr = int(cfg["samplerate"])
    frame_len = int(sr * int(cfg["frame_ms"]) / 1000)
    hop = frame_len // 2
    calib_samples = int(args.calib_sec * sr)
    eval_samples = int(args.eval_sec * sr)
    total_samples = calib_samples + eval_samples

    input_wav_path = Path(args.input_wav)
    if not input_wav_path.is_absolute():
        input_wav_path = _DEFAULT_IN_DIR / input_wav_path
    x = read_input_wav(input_wav_path, sr)
    if len(x) < total_samples:
        raise ValueError(
            f"input_wav is too short ({len(x)/sr:.2f}s). Need at least {(total_samples/sr):.2f}s."
        )
    x = x[:total_samples]
    print(f"Loaded WAV: {input_wav_path}")
    filename = input_wav_path.stem

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    cal = x[:calib_samples]
    eva = x[calib_samples:]
    save_wav(run_dir / f"{filename}_input_eval.wav", sr, eva)

    rows = []
    for algo in suppression_algorithms.keys():
        y, algo_hop = run_algo(algo, x, cfg, calib_samples)
        metrics = summarize_metrics(eva, y, frame_len=frame_len, hop=algo_hop)
        metrics["algorithm"] = algo
        rows.append(metrics)
        algo_name = getattr(algo, "value", str(algo))
        save_wav(run_dir / f"{filename}_{algo_name}.wav", sr, y)
        print(
            f"{algo:20s} total_cut={metrics['total_cut_db']:+5.2f} dB  "
            f"noise_cut~={metrics['noise_cut_est_db']:+5.2f} dB  "
            f"speech_attn~={metrics['speech_attn_est_db']:+5.2f} dB"
        )

    rows.sort(key=lambda r: (r["noise_cut_est_db"] - abs(r["speech_attn_est_db"])), reverse=True)

    csv_path = run_dir / f"{filename}_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "in_dbfs",
                "out_dbfs",
                "total_cut_db",
                "noise_cut_est_db",
                "speech_attn_est_db",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nSaved comparison CSV: {csv_path}")
    print(f"Saved WAVs in: {run_dir}")


if __name__ == "__main__":
    main()
