// Same peak selection and pairing as fingerprint.py, using browser audio decoding.
const SR = 12000;
const FRAME = 1024;
const HOP = 240;
const BINS = FRAME / 2 + 1;
const MAX_SECONDS = 15;
const RECORD_SECONDS = 5;
const PLOT_LAST_BIN = Math.floor(3500 * FRAME / SR);
const FORMAT = "dsp501-peak-pairs-v2";
const HANN = Float64Array.from({ length: FRAME }, (_, i) =>
  (1 - Math.cos(2 * Math.PI * i / FRAME)) / 2
);
const COS = Float64Array.from({ length: FRAME / 2 }, (_, i) => Math.cos(2 * Math.PI * i / FRAME));
const SIN = Float64Array.from({ length: FRAME / 2 }, (_, i) => -Math.sin(2 * Math.PI * i / FRAME));
const REVERSED = Uint16Array.from({ length: FRAME }, (_, i) =>
  Number.parseInt(i.toString(2).padStart(10, "0").split("").reverse().join(""), 2)
);

function magnitudes(wave) {
  const frames = Math.floor((wave.length - FRAME) / HOP) + 1;
  const values = new Float64Array(frames * BINS);
  const real = new Float64Array(FRAME);
  const imag = new Float64Array(FRAME);
  let maximum = 0;
  let strongestFrame = 0;
  for (let frame = 0; frame < frames; frame++) {
    imag.fill(0);
    for (let i = 0; i < FRAME; i++) {
      real[REVERSED[i]] = wave[frame * HOP + i] * HANN[i];
    }
    for (let size = 2; size <= FRAME; size *= 2) {
      const half = size / 2;
      const step = FRAME / size;
      for (let start = 0; start < FRAME; start += size) {
        for (let j = 0; j < half; j++) {
          const a = start + j;
          const b = a + half;
          const cos = COS[j * step];
          const sin = SIN[j * step];
          const r = real[b] * cos - imag[b] * sin;
          const i = real[b] * sin + imag[b] * cos;
          real[b] = real[a] - r;
          imag[b] = imag[a] - i;
          real[a] += r;
          imag[a] += i;
        }
      }
    }
    for (let bin = 0; bin < BINS; bin++) {
      const magnitude = Math.hypot(real[bin], imag[bin]);
      values[frame * BINS + bin] = magnitude;
      if (magnitude > maximum) {
        maximum = magnitude;
        strongestFrame = frame;
      }
    }
  }
  return { frames, values, maximum, strongestFrame };
}

function spectralPeaks({ values, frames, maximum }) {
  const bandMax = new Float64Array(values.length);
  for (let frame = 0; frame < frames; frame++) {
    for (let bin = 0; bin < BINS; bin++) {
      let highest = 0;
      for (let f = Math.max(0, bin - 5); f <= Math.min(BINS - 1, bin + 5); f++) {
        highest = Math.max(highest, values[frame * BINS + f]);
      }
      bandMax[frame * BINS + bin] = highest;
    }
  }
  const peaks = [];
  const first = Math.ceil(120 * FRAME / SR);
  const last = Math.floor(3500 * FRAME / SR);
  for (let frame = 0; frame < frames; frame++) {
    for (let bin = first; bin <= last; bin++) {
      const value = values[frame * BINS + bin];
      if (value <= 0.018 * maximum) continue;
      let isPeak = true;
      for (let t = Math.max(0, frame - 6); t <= Math.min(frames - 1, frame + 6); t++) {
        if (value < bandMax[t * BINS + bin]) {
          isPeak = false;
          break;
        }
      }
      if (isPeak) peaks.push([frame, bin]);
    }
  }
  return peaks;
}

async function digest(a, b, delay) {
  const bytes = new DataView(new ArrayBuffer(6));
  bytes.setUint16(0, a, true);
  bytes.setUint16(2, b, true);
  bytes.setUint16(4, delay, true);
  const hash = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(hash, 0, 8), x => x.toString(16).padStart(2, "0")).join("");
}

export async function fingerprintQuery(wave, onTrace) {
  if (wave.length < SR || wave.length > SR * MAX_SECONDS) {
    throw new Error("Choose a clip at least one second long.");
  }
  const spectrum = magnitudes(wave);
  const peaks = spectralPeaks(spectrum);
  const hashes = new Map();
  const pairs = [];
  let firstPair;
  for (let i = 0; i < peaks.length; i++) {
    const [frame, bin] = peaks[i];
    let paired = 0;
    for (let j = i + 1; j < peaks.length; j++) {
      const [target, other] = peaks[j];
      const delay = target - frame;
      if (delay > 45) break;
      if (delay < 5) continue;
      const a = Math.floor(bin / 2);
      const b = Math.floor(other / 2);
      // Python rounds half-integers to the nearest even integer.
      const dt = delay % 4 === 1 ? Math.floor(delay / 2) : Math.round(delay / 2);
      const key = (a * 512 + b) * 32 + dt;
      if (!hashes.has(key)) hashes.set(key, digest(a, b, dt));
      firstPair ??= [peaks[i], peaks[j]];
      pairs.push([hashes.get(key), frame]);
      if (pairs.length > 16000) throw new Error("Try a shorter clip.");
      if (++paired === 8) break;
    }
  }
  const query = {
    format: FORMAT,
    samples: wave.length,
    pairs: await Promise.all(pairs.map(async ([hash, frame]) => [await hash, frame])),
  };
  onTrace?.({ spectrum, peaks, firstPair, pairCount: pairs.length });
  return query;
}

async function decodeLocally(blob) {
  const context = new OfflineAudioContext(1, 1, SR);
  const audio = await context.decodeAudioData(await blob.arrayBuffer());
  const wave = new Float32Array(Math.min(audio.length, SR * MAX_SECONDS));
  for (let channel = 0; channel < audio.numberOfChannels; channel++) {
    const samples = audio.getChannelData(channel);
    for (let i = 0; i < wave.length; i++) wave[i] += samples[i] / audio.numberOfChannels;
  }
  return wave;
}

function plotLine(context, count, y, color) {
  const { width, height } = context.canvas;
  context.beginPath();
  context.lineWidth = 2;
  context.strokeStyle = color;
  for (let i = 0; i < count; i++) {
    const x = i * (width - 1) / (count - 1);
    const point = (height - 1) * y(i);
    if (i === 0) context.moveTo(x, point);
    else context.lineTo(x, point);
  }
  context.stroke();
}

function drawSpectrogram(context, spectrum) {
  const { width, height } = context.canvas;
  const image = context.createImageData(width, height);
  const reference = spectrum.maximum || 1;
  for (let y = 0; y < height; y++) {
    const bin = Math.round((1 - y / (height - 1)) * PLOT_LAST_BIN);
    for (let x = 0; x < width; x++) {
      const frame = Math.round(x * (spectrum.frames - 1) / (width - 1));
      const level = Math.max(0.001, spectrum.values[frame * BINS + bin] / reference);
      const t = 1 + Math.log10(level) / 3;
      const pixel = (y * width + x) * 4;
      image.data[pixel] = 12 + 235 * t ** 2;
      image.data[pixel + 1] = 24 + 199 * t;
      image.data[pixel + 2] = 42 + 114 * t * (2 - t);
      image.data[pixel + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
}

function drawDspSteps(root, wave, { spectrum, peaks, firstPair, pairCount }) {
  const plots = Object.fromEntries(Array.from(root.querySelectorAll("[data-plot]"), canvas =>
    [canvas.dataset.plot, canvas.getContext("2d")]
  ));
  if (Object.values(plots).some(context => !context)) return;
  const teal = "#087e74";
  const gray = "#64748b";
  for (const context of Object.values(plots)) {
    context.fillStyle = "#f5f9fa";
    context.fillRect(0, 0, context.canvas.width, context.canvas.height);
  }

  // Plot the full amplitude range in each pixel, including short transients.
  const overview = plots.wave;
  const { width, height } = overview.canvas;
  overview.beginPath();
  overview.strokeStyle = teal;
  overview.lineWidth = 1.5;
  for (let x = 0; x < width; x++) {
    let minimum = 1;
    let maximum = -1;
    const end = Math.floor((x + 1) * wave.length / width);
    for (let i = Math.floor(x * wave.length / width); i < end; i++) {
      minimum = Math.min(minimum, wave[i]);
      maximum = Math.max(maximum, wave[i]);
    }
    overview.moveTo(x, height * (0.5 - 0.46 * maximum));
    overview.lineTo(x, height * (0.5 - 0.46 * minimum) + 1);
  }
  overview.stroke();

  const start = spectrum.strongestFrame * HOP;
  const frame = wave.subarray(start, start + FRAME);
  const amplitude = Math.max(0.001, ...frame.map(Math.abs));
  plotLine(plots.window, FRAME, i => 0.5 - 0.46 * frame[i] / amplitude, gray);
  plotLine(plots.window, FRAME, i => 0.5 - 0.46 * frame[i] * HANN[i] / amplitude, teal);

  const fftStart = spectrum.strongestFrame * BINS;
  const reference = spectrum.maximum || 1;
  plotLine(plots.fft, PLOT_LAST_BIN + 1, bin =>
    -Math.log10(Math.max(0.001, spectrum.values[fftStart + bin] / reference)) / 3, teal
  );

  drawSpectrogram(plots.stft, spectrum);
  const map = plots.peaks;
  map.drawImage(plots.stft.canvas, 0, 0);
  const x = frame => frame * (map.canvas.width - 1) / (spectrum.frames - 1);
  const y = bin => (1 - bin / PLOT_LAST_BIN) * (map.canvas.height - 1);
  map.fillStyle = "#acf6e7";
  for (const [time, bin] of peaks) {
    map.fillRect(x(time) - 1, y(bin) - 1, 3, 3);
  }
  if (firstPair) {
    const [[a, f1], [b, f2]] = firstPair;
    map.strokeStyle = "#fff";
    map.lineWidth = 2;
    map.beginPath();
    map.moveTo(x(a), y(f1));
    map.lineTo(x(b), y(f2));
    map.stroke();
  }

  root.querySelector('[data-axis="wave"]').textContent =
    `0–${(wave.length / SR).toFixed(2)} s · Amplitude from −1 to 1`;
  const lastFrame = (spectrum.frames - 1) * HOP / SR;
  root.querySelector('[data-axis="stft"]').textContent =
    `0–${lastFrame.toFixed(2)} s (frame starts) · 0–3.5 kHz · −60 to 0 dB`;
  const peakSummary = `${peaks.length.toLocaleString()} peaks · ${pairCount.toLocaleString()} pairs`;
  root.querySelector('[data-axis="peaks"]').textContent =
    firstPair ? `${peakSummary} · White line shows one pair` : peakSummary;

  root.querySelector(".dsp-empty").hidden = true;
  root.querySelectorAll("[data-plot], [data-axis]").forEach(node => { node.hidden = false; });
}

export default function ({ parentElement: root, setTriggerValue }) {
  const upload = root.querySelector(".audio-file");
  const drop = root.querySelector(".drop-zone");
  const record = root.querySelector(".record-button");
  const card = root.querySelector(".record-card");
  const action = root.querySelector(".record-action");
  const countdown = root.querySelector(".record-time");
  const status = root.querySelector("[role=status]");
  let recorder;
  let stream;
  let timer;
  let tick;
  let disposed = false;

  function busy(value) {
    upload.disabled = value;
    drop.disabled = value;
    record.disabled = value;
    if (value) drop.removeAttribute("data-dragover");
  }

  function stopTracks() {
    clearTimeout(timer);
    clearInterval(tick);
    stream?.getTracks().forEach(track => track.stop());
    stream = undefined;
    card.removeAttribute("data-recording");
    countdown.textContent = "0:05";
    action.textContent = "Start recording";
  }

  async function analyze(blob, label) {
    busy(true);
    status.textContent = `Analyzing ${label}…`;
    try {
      const wave = await decodeLocally(blob);
      let trace;
      const query = await fingerprintQuery(wave, value => { trace = value; });
      if (!disposed) {
        drawDspSteps(root, wave, trace);
        status.textContent = "";
        // The waveform, plots, and filenames stay in the browser.
        setTriggerValue("query", query);
      }
    } catch (error) {
      status.textContent = error instanceof DOMException
        ? "This browser could not decode the audio. Try an MP3 or WAV file."
        : error.message;
    } finally {
      busy(false);
    }
  }

  async function useFile(file) {
    if (!file || drop.disabled) return;
    if (file.size > 20 * 1024 * 1024) {
      status.textContent = "Choose a file up to 20 MB.";
      return;
    }
    await analyze(file, file.name);
  }

  drop.onclick = () => upload.click();
  drop.ondragover = event => {
    if (!Array.from(event.dataTransfer.types).includes("Files")) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = drop.disabled ? "none" : "copy";
    if (!drop.disabled) drop.setAttribute("data-dragover", "");
  };
  drop.ondragleave = () => drop.removeAttribute("data-dragover");
  drop.ondrop = event => {
    event.preventDefault();
    drop.removeAttribute("data-dragover");
    return useFile(event.dataTransfer.files[0]);
  };
  upload.onchange = () => {
    const file = upload.files[0];
    upload.value = "";
    return useFile(file);
  };

  record.onclick = async () => {
    busy(true);
    status.textContent = "Waiting for microphone access…";
    try {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        throw new Error("Microphone recording is unavailable in this browser.");
      }
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      if (disposed) {
        stopTracks();
        return;
      }
      recorder = new MediaRecorder(stream);
      const chunks = [];
      recorder.ondataavailable = ({ data }) => {
        if (data.size) chunks.push(data);
      };
      recorder.onerror = () => {
        recorder.onstop = null;
        stopTracks();
        status.textContent = "Recording failed. Try again or choose an audio file.";
        busy(false);
      };
      recorder.onstop = async () => {
        stopTracks();
        if (!disposed) await analyze(new Blob(chunks, { type: recorder.mimeType }), "recording");
      };
      recorder.start();
      status.textContent = "Recording for five seconds…";
      card.setAttribute("data-recording", "");
      action.textContent = "Recording…";
      const end = performance.now() + RECORD_SECONDS * 1000;
      tick = setInterval(() => {
        const seconds = Math.max(0, Math.ceil((end - performance.now()) / 1000));
        countdown.textContent = `0:0${seconds}`;
      }, 100);
      timer = setTimeout(() => {
        if (recorder.state === "recording") recorder.stop();
      }, RECORD_SECONDS * 1000);
    } catch (error) {
      stopTracks();
      status.textContent = error.name === "NotAllowedError"
        ? "Allow microphone access or choose an audio file."
        : error.message;
      busy(false);
    }
  };

  return () => {
    disposed = true;
    if (recorder?.state === "recording") recorder.stop();
    stopTracks();
  };
}
