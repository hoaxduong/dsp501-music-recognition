// Same peak selection and pairing as fingerprint.py, using browser audio decoding.
const SR = 12000;
const FRAME = 1024;
const HOP = 240;
const BINS = FRAME / 2 + 1;
const MAX_SECONDS = 15;
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
      maximum = Math.max(maximum, magnitude);
    }
  }
  return { frames, values, maximum };
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

export async function fingerprintQuery(wave) {
  if (wave.length < SR || wave.length > SR * MAX_SECONDS) {
    throw new Error("Choose a clip at least one second long.");
  }
  const peaks = spectralPeaks(magnitudes(wave));
  const hashes = new Map();
  const pairs = [];
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
      pairs.push([hashes.get(key), frame]);
      if (pairs.length > 16000) throw new Error("Try a shorter clip.");
      if (++paired === 8) break;
    }
  }
  return {
    format: FORMAT,
    samples: wave.length,
    pairs: await Promise.all(pairs.map(async ([hash, frame]) => [await hash, frame])),
  };
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

export default function ({ parentElement, setTriggerValue }) {
  const upload = parentElement.querySelector("input");
  const record = parentElement.querySelector("button");
  const status = parentElement.querySelector("[role=status]");
  let recorder;
  let stream;
  let timer;
  let disposed = false;

  function busy(value) {
    upload.disabled = value;
    record.disabled = value;
  }

  function stopTracks() {
    clearTimeout(timer);
    stream?.getTracks().forEach(track => track.stop());
    stream = undefined;
  }

  async function analyze(blob) {
    busy(true);
    status.textContent = "Analyzing audio in your browser…";
    try {
      const wave = await decodeLocally(blob);
      const query = await fingerprintQuery(wave);
      if (!disposed) {
        status.textContent = "";
        // This is the only payload sent to Streamlit. It has no audio bytes.
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

  upload.onchange = async () => {
    const file = upload.files[0];
    upload.value = "";
    if (!file) return;
    if (file.size > 20 * 1024 * 1024) {
      status.textContent = "Choose a file under 20 MB.";
      return;
    }
    await analyze(file);
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
        stopTracks();
        status.textContent = "Recording failed. Try again or choose an audio file.";
        busy(false);
      };
      recorder.onstop = async () => {
        stopTracks();
        if (!disposed) await analyze(new Blob(chunks, { type: recorder.mimeType }));
      };
      recorder.start();
      status.textContent = "Recording five seconds…";
      timer = setTimeout(() => {
        if (recorder.state === "recording") recorder.stop();
      }, 5000);
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
