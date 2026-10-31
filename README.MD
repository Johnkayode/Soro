## Sòrò

Sòrò is a noise-suppression microphone for third-party applications. 

## Algorithms
Four spectral noise-suppression algorithms are current implemented.
They share a short-time Fourier transform (STFT) processing pipeline. The pipeline tracks an estimate of the noise power spectrum, computes a frequency-dependent gain, smooths that gain over time, and reconstructs the output with overlap-add. The algorithms differ in how they estimate the clean-speech gain:

### Spectral Subtraction

Subtracts a scaled estimate of the noise magnitude from each input magnitude-frequency bin. The `noise_beta` setting controls the amount subtracted; a gain floor prevents bins from being completely removed. It is simple and fast, but an inaccurate noise estimate can cause speech distortion or musical-noise artifacts.

### Decision-Directed Wiener (DD Wiener)

Uses a Wiener filter whose gain depends on the estimated speech-to-noise ratio. Its decision-directed estimate combines the current frame’s SNR with the previous frame’s estimated clean-speech power, helping keep gains stable between frames. `dd_alpha` controls how strongly it relies on the previous estimate.

### Log-MMSE

Uses a log-spectral-amplitude minimum mean-square-error estimator. Like DD Wiener, it uses a decision-directed prior SNR, but applies the Log-MMSE gain function to estimate speech magnitude. It is designed to reduce residual noise while avoiding some of the harshness of direct spectral subtraction.

### OMLSA (Optimally Modified Log-Spectral Amplitude)

Uses an optimally modified log-spectral-amplitude gain and estimates the probability that speech is present in each frequency bin. It combines speech-present and speech-absent gains and uses a minima-controlled recursive noise estimate to reduce speech leakage into the noise model. `omlsa_q`, `omlsa_min_gain_db`, and `min_track_frames` control aspects of this behavior.

## Setup

The project pins Python 3.11.6 in `.python-version` and uses `uv` to manage the virtual environment and dependencies. Install `uv` if needed (on macOS with Homebrew):

```sh
brew install uv
```

From the project root, create or update the environment:

```sh
uv sync
```


## Benchmark Tests

Check that the benchmark command is available:

```sh
uv run python testrun/compare_algo.py --help
```

The benchmark requires a WAV file and applies every algorithm to the same clip. Sample WAVs are stored in `testrun/samples/`. For example, `car_48000hz.wav` is sampled at 48 kHz:

```sh
uv run python testrun/compare_algo.py \
  --input_wav car_48000hz.wav \
  --samplerate 48000
```

Relative `--input_wav` paths are resolved from `testrun/samples/`; absolute paths are also accepted. Multichannel files use their first channel. The WAV sample rate must match `--samplerate` or the value in `config.yaml`; the benchmark does not resample audio. Run commands from the project root so the default `config.yaml` is found.

By default, the first 2 seconds are used to estimate background noise and are excluded from the evaluation output. Ideally this segment contains noise without speech. If the recording has no noise-only lead-in, use `--calib_sec 0`; the algorithms will estimate noise while processing, though initial output quality may be lower. The input must contain at least `calib_sec + eval_sec` seconds (14 seconds by default).

Results go into a timestamped directory under `testrun/bench_out/`. For the example input, the files include:

```text
testrun/bench_out/<timestamp>/car_48000hz_input_eval.wav
testrun/bench_out/<timestamp>/car_48000hz_spectral_subtraction.wav
testrun/bench_out/<timestamp>/car_48000hz_dd_weiner.wav
testrun/bench_out/<timestamp>/car_48000hz_log_mmse.wav
testrun/bench_out/<timestamp>/car_48000hz_omlsa.wav
testrun/bench_out/<timestamp>/car_48000hz_comparison.csv
```

Use `--out_dir /path/to/results` to choose a different parent directory; each run still gets its own timestamped subdirectory.

| Option | Default | Description |
| --- | --- | --- |
| `--input_wav` | Required | WAV filename in `testrun/samples/`, or an absolute path |
| `--config` | `config.yaml` | Configuration file, resolved from the current working directory |
| `--samplerate` | From config (`16000`) | Expected input WAV sample rate |
| `--frame_ms` | From config (`20`) | Processing frame length in milliseconds |
| `--calib_sec` | `2.0` | Noise-only calibration duration at the start of the WAV |
| `--eval_sec` | `12.0` | Duration to process after calibration |
| `--out_dir` | `testrun/bench_out` | Parent directory for timestamped results |
