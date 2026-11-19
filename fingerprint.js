/*
 * fingerprint.js
 * JavaScript port of the "art-v1" artwork fingerprint from build_fingerprints.py
 * (SPEC / describe_art). Same numbers as Python for the same pixels, so a photo
 * fingerprinted in the browser can be compared against fingerprints.json.
 *
 * Works in the browser (window.UVSFingerprint) and in Node (require).
 *
 * Browser use:
 *   const db = UVSFingerprint.loadFingerprints(await (await fetch("fingerprints.json")).json());
 *   const q = await UVSFingerprint.describeArtFromSource(imageOrBlobOrCanvas);
 *   const best = UVSFingerprint.rank(db, q, { limit: 5 });            // whole catalog
 *   const best2 = UVSFingerprint.rank(db, q, { candidates: [...ids] }); // only these printings
 *
 * The input must be the card itself, upright, filling the image (as the card
 * scans in the art folder do). Finding and straightening the card in a camera
 * frame happens before this step.
 *
 * Matching Python exactly needs three things, all handled here:
 *   1. Crop box rounded like Python's round() (ties to even).
 *   2. Shrink with Pillow's BOX resampling, reproduced integer for integer
 *      (two passes, 22-bit fixed point, rounded to 8 bits between passes).
 *   3. Decode the image without colour-profile conversion and without
 *      premultiplied alpha, because Pillow ignores both.
 */
(function (root) {
  "use strict";

  // ------------------------------------------------------------ the recipe
  // Must match the descriptor block in build_fingerprints.py.
  const ART_BOX = [0.06, 0.04, 0.94, 0.56];
  const WORK_W = 48, WORK_H = 42;
  const LAYOUT_COLS = 8, LAYOUT_ROWS = 6;
  const HOG_COLS = 4, HOG_ROWS = 3, HOG_BINS = 8;
  const DIM = LAYOUT_COLS * LAYOUT_ROWS * 3 + HOG_COLS * HOG_ROWS * HOG_BINS; // 240
  const METHOD = "art-v1";

  // Python 3 round(): halves go to the even neighbour.
  function pyRound(x) {
    const f = Math.floor(x), d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }

  function artCropRect(w, h) {
    const [l, t, r, b] = ART_BOX;
    const x0 = pyRound(w * l), y0 = pyRound(h * t);
    return { x: x0, y: y0, w: pyRound(w * r) - x0, h: pyRound(h * b) - y0 };
  }

  // ------------------------------------------- Pillow BOX resize (8 bit)
  // Mirrors precompute_coeffs / normalize_coeffs_8bpc / ImagingResample*_8bpc
  // in Pillow's src/libImaging/Resample.c.
  const PRECISION_BITS = 32 - 8 - 2;
  const ONE = 1 << PRECISION_BITS;

  function boxFilter(x) {
    return x > -0.5 && x <= 0.5 ? 1.0 : 0.0;
  }

  function coeffs(inSize, outSize) {
    const support0 = 0.5; // box filter support
    const scale = inSize / outSize;
    const filterscale = scale < 1.0 ? 1.0 : scale;
    const support = support0 * filterscale;
    const ksize = Math.ceil(support) * 2 + 1;
    const kk = new Int32Array(outSize * ksize);
    const bounds = new Int32Array(outSize * 2);
    const pre = new Float64Array(ksize);
    const ss = 1.0 / filterscale;
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      let xmin = Math.trunc(center - support + 0.5);
      if (xmin < 0) xmin = 0;
      let xmax = Math.trunc(center + support + 0.5);
      if (xmax > inSize) xmax = inSize;
      xmax -= xmin;
      let ww = 0.0;
      for (let x = 0; x < xmax; x++) {
        const w = boxFilter((x + xmin - center + 0.5) * ss);
        pre[x] = w;
        ww += w;
      }
      for (let x = 0; x < xmax; x++) {
        const w = ww !== 0.0 ? pre[x] / ww : pre[x];
        kk[xx * ksize + x] = w < 0 ? Math.trunc(-0.5 + w * ONE) : Math.trunc(0.5 + w * ONE);
      }
      bounds[xx * 2] = xmin;
      bounds[xx * 2 + 1] = xmax;
    }
    return { kk, bounds, ksize };
  }

  function clip8(ss) {
    const v = Math.floor(ss / ONE);
    return v < 0 ? 0 : v > 255 ? 255 : v;
  }

  /**
   * Shrink a region of an RGBA pixel buffer exactly like
   * img.crop(rect).resize((outW, outH), Image.Resampling.BOX) on an RGB image.
   * Returns a Uint8Array of outW*outH*3 (RGB).
   */
  function resizeBoxPIL(data, imgW, rect, outW, outH) {
    const hz = coeffs(rect.w, outW), vt = coeffs(rect.h, outH);
    // horizontal pass: rect.h rows x outW, RGB
    const tmp = new Uint8Array(rect.h * outW * 3);
    for (let y = 0; y < rect.h; y++) {
      const rowBase = ((rect.y + y) * imgW + rect.x) * 4;
      for (let xx = 0; xx < outW; xx++) {
        const xmin = hz.bounds[xx * 2], xmax = hz.bounds[xx * 2 + 1], kb = xx * hz.ksize;
        let s0 = ONE >> 1, s1 = ONE >> 1, s2 = ONE >> 1;
        for (let x = 0; x < xmax; x++) {
          const k = hz.kk[kb + x], p = rowBase + (x + xmin) * 4;
          s0 += data[p] * k; s1 += data[p + 1] * k; s2 += data[p + 2] * k;
        }
        const o = (y * outW + xx) * 3;
        tmp[o] = clip8(s0); tmp[o + 1] = clip8(s1); tmp[o + 2] = clip8(s2);
      }
    }
    // vertical pass
    const out = new Uint8Array(outH * outW * 3);
    for (let yy = 0; yy < outH; yy++) {
      const ymin = vt.bounds[yy * 2], ymax = vt.bounds[yy * 2 + 1], kb = yy * vt.ksize;
      for (let xx = 0; xx < outW; xx++) {
        let s0 = ONE >> 1, s1 = ONE >> 1, s2 = ONE >> 1;
        for (let y = 0; y < ymax; y++) {
          const k = vt.kk[kb + y], p = ((y + ymin) * outW + xx) * 3;
          s0 += tmp[p] * k; s1 += tmp[p + 1] * k; s2 += tmp[p + 2] * k;
        }
        const o = (yy * outW + xx) * 3;
        out[o] = clip8(s0); out[o + 1] = clip8(s1); out[o + 2] = clip8(s2);
      }
    }
    return out;
  }

  // ------------------------------------------------------- the descriptor
  function normalize(v) {
    let n = 0;
    for (let i = 0; i < v.length; i++) n += v[i] * v[i];
    n = Math.sqrt(n) + 1e-8;
    for (let i = 0; i < v.length; i++) v[i] /= n;
    return v;
  }

  /** Fingerprint from already-shrunk 48x42 RGB pixels. */
  function describeSmall(rgb) {
    const N = WORK_W * WORK_H;
    const ch = [new Float64Array(N), new Float64Array(N), new Float64Array(N)]; // Y, Cb, Cr
    for (let i = 0; i < N; i++) {
      // float64, as build_fingerprints.py does it
      const R = rgb[i * 3] / 255, G = rgb[i * 3 + 1] / 255, B = rgb[i * 3 + 2] / 255;
      const Y = 0.299 * R + 0.587 * G + 0.114 * B;
      ch[0][i] = Y;
      ch[1][i] = B - Y;
      ch[2][i] = R - Y;
    }
    for (const c of ch) {
      let m = 0;
      for (let i = 0; i < N; i++) m += c[i];
      m /= N;
      let v = 0;
      for (let i = 0; i < N; i++) v += (c[i] - m) * (c[i] - m);
      const sd = Math.max(Math.sqrt(v / N), 0.02);
      for (let i = 0; i < N; i++) c[i] = (c[i] - m) / sd;
    }

    // (a) layout: 8x6 grid x 3 channels = 144
    const cellH = Math.floor(WORK_H / LAYOUT_ROWS), cellW = Math.floor(WORK_W / LAYOUT_COLS);
    const layout = new Float64Array(LAYOUT_ROWS * LAYOUT_COLS * 3);
    for (let c = 0; c < 3; c++) {
      for (let r = 0; r < LAYOUT_ROWS; r++) {
        for (let q = 0; q < LAYOUT_COLS; q++) {
          let s = 0;
          for (let y = r * cellH; y < (r + 1) * cellH; y++)
            for (let x = q * cellW; x < (q + 1) * cellW; x++) s += ch[c][y * WORK_W + x];
          layout[c * LAYOUT_ROWS * LAYOUT_COLS + r * LAYOUT_COLS + q] = s / (cellH * cellW);
        }
      }
    }
    normalize(layout);

    // (b) edge directions: 4x3 cells x 8 bins = 96
    const z = ch[0];
    const at = (y, x) => z[y * WORK_W + x];
    const hcH = Math.floor(WORK_H / HOG_ROWS), hcW = Math.floor(WORK_W / HOG_COLS);
    const hist = new Float64Array(HOG_ROWS * HOG_COLS * HOG_BINS);
    for (let y = 0; y < WORK_H; y++) {
      for (let x = 0; x < WORK_W; x++) {
        // numpy.gradient: central differences inside, one-sided at the edges
        const gy = y === 0 ? at(1, x) - at(0, x) : y === WORK_H - 1 ? at(y, x) - at(y - 1, x) : (at(y + 1, x) - at(y - 1, x)) / 2;
        const gx = x === 0 ? at(y, 1) - at(y, 0) : x === WORK_W - 1 ? at(y, x) - at(y, x - 1) : (at(y, x + 1) - at(y, x - 1)) / 2;
        const mag = Math.hypot(gx, gy);
        let ang = Math.atan2(gy, gx) % Math.PI;
        if (ang < 0) ang += Math.PI;
        // + 1e-9: exactly diagonal edges sit on a bin boundary; send ties to the same bin as Python does
        const bin = Math.min(Math.floor((ang / Math.PI) * HOG_BINS + 1e-9), HOG_BINS - 1);
        const cell = Math.floor(y / hcH) * HOG_COLS + Math.floor(x / hcW);
        hist[cell * HOG_BINS + bin] += mag;
      }
    }
    const hog = new Float64Array(hist.length);
    let hogMean = 0;
    for (let c = 0; c < HOG_ROWS * HOG_COLS; c++) {
      let s = 0;
      for (let b = 0; b < HOG_BINS; b++) s += hist[c * HOG_BINS + b];
      for (let b = 0; b < HOG_BINS; b++) {
        hog[c * HOG_BINS + b] = Math.sqrt(hist[c * HOG_BINS + b] / (s + 1e-6));
        hogMean += hog[c * HOG_BINS + b];
      }
    }
    hogMean /= hog.length;
    for (let i = 0; i < hog.length; i++) hog[i] -= hogMean;
    normalize(hog);

    const v = new Float64Array(DIM);
    v.set(layout, 0);
    v.set(hog, layout.length);
    return Float32Array.from(normalize(v));
  }

  /** Fingerprint from a full card image as RGBA pixels (ImageData-like: {data, width, height}). */
  function describeArt(img) {
    if (img.width > img.height * 0.9) throw new Error("image is not a portrait card");
    const rect = artCropRect(img.width, img.height);
    return describeSmall(resizeBoxPIL(img.data, img.width, rect, WORK_W, WORK_H));
  }

  // --------------------------------------------------------- browser glue
  /**
   * Decode an <img>, Blob/File, ImageBitmap, <canvas> or <video> frame and fingerprint it.
   * Blobs are decoded with colour conversion off, matching Pillow.
   */
  async function describeArtFromSource(src) {
    let bmp = src, made = false;
    if (typeof Blob !== "undefined" && src instanceof Blob) {
      bmp = await createImageBitmap(src, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
      made = true;
    }
    const w = bmp.naturalWidth || bmp.videoWidth || bmp.width, h = bmp.naturalHeight || bmp.videoHeight || bmp.height;
    const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(w, h) : Object.assign(document.createElement("canvas"), { width: w, height: h });
    const ctx = canvas.getContext("2d", { willReadFrequently: true, colorSpace: "srgb" });
    ctx.drawImage(bmp, 0, 0);
    if (made && bmp.close) bmp.close();
    return describeArt(ctx.getImageData(0, 0, w, h));
  }

  // ------------------------------------------------- fingerprints.json
  function halfToFloat(h) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) return s * f * 2 ** -24;
    if (e === 31) return f ? NaN : s * Infinity;
    return s * (1 + f / 1024) * 2 ** (e - 15);
  }

  function base64ToBytes(b64) {
    if (typeof atob === "function") {
      const bin = atob(b64), out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    }
    return new Uint8Array(Buffer.from(b64, "base64"));
  }

  /** Parse the object from fingerprints.json into a searchable database. */
  function loadFingerprints(json) {
    if (!json.spec || json.spec.method !== METHOD) throw new Error("fingerprints.json was built with method " + (json.spec && json.spec.method) + ", this file computes " + METHOD);
    if (json.dim !== DIM) throw new Error("fingerprint length " + json.dim + ", expected " + DIM);
    const bytes = base64ToBytes(json.data);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const n = json.count, matrix = new Float32Array(n * DIM);
    for (let i = 0; i < n * DIM; i++) matrix[i] = halfToFloat(view.getUint16(i * 2, true));
    const index = new Map(json.ids.map((id, i) => [id, i]));
    return { ids: json.ids, dim: DIM, count: n, matrix, index, twins: json.twins || {}, twinThreshold: json.twin_threshold };
  }

  function similarity(db, q, i) {
    let s = 0;
    const o = i * DIM;
    for (let d = 0; d < DIM; d++) s += db.matrix[o + d] * q[d];
    return s;
  }

  /**
   * Rank printings by artwork similarity (cosine; 1 = identical).
   * options.candidates: printing ids to choose among (e.g. from the text scan); default all.
   * options.limit: how many to return (default 10).
   * Each result lists its same-art twins so the app can show them together.
   */
  function rank(db, q, options) {
    const opts = options || {};
    const limit = opts.limit || 10;
    const pool = opts.candidates ? opts.candidates.map((id) => db.index.get(String(id).toLowerCase())).filter((i) => i !== undefined) : null;
    const scored = [];
    const n = pool ? pool.length : db.count;
    for (let k = 0; k < n; k++) {
      const i = pool ? pool[k] : k;
      scored.push({ id: db.ids[i], score: similarity(db, q, i) });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit).map((r) => ({ ...r, twins: db.twins[r.id] || [] }));
  }

  const api = {
    METHOD, DIM, ART_BOX, WORK_W, WORK_H,
    pyRound, artCropRect, resizeBoxPIL, describeSmall, describeArt, describeArtFromSource,
    loadFingerprints, rank, halfToFloat,
  };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.UVSFingerprint = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
