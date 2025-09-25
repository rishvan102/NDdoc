/* NiceDay PDF Jar – single-file app logic
   - Robust pdf-lib loader + fontkit
   - Unicode-safe text drawing (cascading fallback: Latin → Tamil → Symbols/Emoji)
   - Tools: note/txt→pdf, images↔pdf, compress, merge, split, reorder, rotate, delete, extract,
     page numbers, watermark, strip metadata, redact
   - Subpath-aware paths for PWA/service worker and assets
*/

/* ---------------------------- Subpath helper ---------------------------- */
(function () {
  // Optional global override: window.__ND_BASE_PATH__ = "/NiceDay/"
  const baseAttr = (document.documentElement.getAttribute("data-basepath") || "").trim();
  const base = (window.__ND_BASE_PATH__ || baseAttr || "").replace(/^\/|\/$/g, "");
  window.__ndPath__ = (p) => {
    const clean = String(p || "").replace(/^\/+/, "");
    return base ? `/${base}/${clean}` : `/${clean}`;
  };
})();

/* --------------------------- Boot coordination -------------------------- */
(function () {
  window.__ndBoot = { domReady: false, pdfLibReady: false, started: false };
  window.__ndTryStart = function () {
    // Left for compatibility; handlers are attached on DOMContentLoaded.
  };
})();

/* -------------------------- pdf.js worker path -------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  if (window.pdfjsLib) {
    // Keep cdnjs worker in sync with the <script> version used in index.html
    window.pdfjsLib.GlobalWorkerOptions.workerSrc =
      "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
  }
});

/* ------------------------ Robust pdf-lib + fontkit ---------------------- */
(function loadPdfLibAndStart() {
  function start() {
    if (!window.PDFLib) {
      alert("pdf-lib failed to load.");
      return;
    }
    window.__ndBoot.pdfLibReady = true;
    window.__ndTryStart();
  }

  function loadFrom(url, onload, onerror) {
    const s = document.createElement("script");
    s.defer = true;
    s.src = url;
    s.onload = onload;
    s.onerror = onerror;
    document.head.appendChild(s);
  }

  // Try unpkg first, fallback to jsDelivr
  loadFrom(
    "https://unpkg.com/pdf-lib@1.17.1/dist/pdf-lib.min.js",
    start,
    () =>
      loadFrom(
        "https://cdn.jsdelivr.net/npm/pdf-lib@1.17.1/dist/pdf-lib.min.js",
        start,
        () => alert("Could not load pdf-lib.")
      )
  );
})();

/* ------------------------------- App code ------------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  window.__ndBoot.domReady = true;

  /* --------- Header bits (year, tabs, search, PWA, SW, vibes) ---------- */
  const yrEl = document.getElementById("yr");
  if (yrEl) yrEl.textContent = new Date().getFullYear();

  // Search filter
  const search = document.getElementById("toolSearch");
  function applyFilter() {
    const activeTab = document.querySelector(".tabpanel:not(.hidden)");
    if (!activeTab) return;
    const cards = activeTab.querySelectorAll(".tool-card");
    const q = (search?.value || "").toLowerCase();
    cards.forEach((card) => {
      const tags = (card.dataset.tags || "").toLowerCase();
      const text = (card.innerText || "").toLowerCase();
      const ok = !q || text.includes(q) || tags.includes(q);
      card.classList.toggle("hidden-by-filter", !ok);
    });
  }
  search?.addEventListener("input", applyFilter);

  // Tabs
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".tabpanel");
  function showTab(name) {
    tabs.forEach((t) => {
      const isActive = t.dataset.tab === name;
      t.classList.toggle("active", isActive);
      t.setAttribute("aria-pressed", String(isActive));
    });
    panels.forEach((p) => p.classList.toggle("hidden", p.dataset.tabpanel !== name));
    applyFilter();
  }
  tabs.forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
  showTab("docs");

  // PWA install button
  const installBtn = document.getElementById("installBtn");
  let deferredPrompt = null;
  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    deferredPrompt = e;
    installBtn?.classList.remove("hidden");
  });
  installBtn?.addEventListener("click", async () => {
    if (!deferredPrompt) return;
    deferredPrompt.prompt();
    await deferredPrompt.userChoice;
    deferredPrompt = null;
    installBtn.classList.add("hidden");
  });
  window.addEventListener("appinstalled", () => installBtn?.classList.add("hidden"));

  // Service Worker (subpath-aware) — only register if file exists to avoid 404 console noise
  (async () => {
    if ("serviceWorker" in navigator) {
      try {
        const swUrl = window.__ndPath__("service-worker.js");
        const scope = window.__ndPath__("");
        const head = await fetch(swUrl, { method: "HEAD" });
        if (head.ok) {
          await navigator.serviceWorker.register(swUrl, { scope });
        } else {
          console.warn("Service worker not found at:", swUrl);
        }
      } catch (err) {
        console.warn("Service worker check failed:", err);
      }
    }
  })();

  // Vibes
  const QUOTES = [
    "You're doing amazing — keep going! 💪",
    "Small steps create big changes 🌱",
    "Your effort matters more than perfection ✨",
    "Today is a fresh start — make it count 🌞",
    "You are stronger than you think 💡",
    "Believe in progress, not perfection 🌟",
    "One positive action can change your whole day 💖",
  ];
  const ACTION_TITLES = {
    note: "Note saved!",
    text: "Document created!",
    merge: "PDFs merged!",
    split: "Pages split!",
    images: "Images converted!",
    rotate: "Pages rotated!",
    reorder: "Pages reordered!",
    del: "Pages deleted!",
    extract: "Pages extracted!",
    numbers: "Numbers added!",
    watermark: "Watermark added!",
    pdf2img: "Images exported!",
    meta: "Clean copy ready!",
    compress: "PDF compressed!",
    redact: "Redactions applied!",
  };
  const mq = document.getElementById("moodQuote");
  if (mq) mq.textContent = `Today’s note: ${QUOTES[Math.floor(Math.random() * QUOTES.length)]}`;

  // Motivator modal
  const overlay = document.getElementById("ndOverlay");
  const modal = document.getElementById("ndModal");
  const titleEl = document.getElementById("ndTitle");
  const msgEl = document.getElementById("ndMsg");
  const closeEl = document.getElementById("ndClose");
  function showMotivator(kind) {
    if (!overlay || !modal) return;
    if (titleEl) titleEl.textContent = ACTION_TITLES[kind] || "Nice work!";
    if (msgEl) msgEl.textContent = QUOTES[Math.floor(Math.random() * QUOTES.length)];
    overlay.classList.add("show");
    modal.style.visibility = "visible";
    modal.classList.add("show");
    closeEl?.focus();
  }
  function hideMotivator() {
    overlay?.classList.remove("show");
    modal?.classList.remove("show");
    setTimeout(() => {
      if (modal) modal.style.visibility = "hidden";
    }, 180);
  }
  overlay?.addEventListener("click", hideMotivator);
  closeEl?.addEventListener("click", hideMotivator);
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") hideMotivator();
  });

  /* ----------------------------- Fonts setup ---------------------------- */
  let FONT_REG_BYTES = null,
    FONT_BOLD_BYTES = null,
    FONT_TAMIL_REG_BYTES = null,
    FONT_TAMIL_BOLD_BYTES = null,
    FONT_SYM_BYTES = null;

  async function fetchTTF(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error("Font fetch failed: " + url);
    return new Uint8Array(await r.arrayBuffer());
  }

  async function ensureFontkit() {
    if (window.fontkit) return window.fontkit;
    await new Promise((res) => {
      const s = document.createElement("script");
      s.defer = true;
      s.src = "https://unpkg.com/@pdf-lib/fontkit@1.1.1/dist/fontkit.umd.min.js";
      s.onload = res;
      s.onerror = res;
      document.head.appendChild(s);
    });
    return window.fontkit;
  }

  async function preloadFontBytes() {
    if (FONT_REG_BYTES && FONT_BOLD_BYTES && FONT_TAMIL_REG_BYTES && FONT_TAMIL_BOLD_BYTES) return;
    const p = (x) => window.__ndPath__("fonts/" + x);

    // Primary (Latin)
    const latinReg = fetchTTF(p("NotoSans-Regular.ttf"));
    const latinBold = fetchTTF(p("NotoSans-Bold.ttf"));
    // Tamil
    const tamilReg = fetchTTF(p("NotoSansTamil-Regular.ttf"));
    const tamilBold = fetchTTF(p("NotoSansTamil-Bold.ttf"));

    [FONT_REG_BYTES, FONT_BOLD_BYTES, FONT_TAMIL_REG_BYTES, FONT_TAMIL_BOLD_BYTES] = await Promise.all([
      latinReg,
      latinBold,
      tamilReg,
      tamilBold,
    ]);

    // Symbols/emoji fallback (optional)
    try {
      FONT_SYM_BYTES = await fetchTTF(p("symbola.ttf"));
    } catch {
      try {
        FONT_SYM_BYTES = await fetchTTF(p("NotoSansSymbols2-Regular.ttf"));
      } catch {
        FONT_SYM_BYTES = null; // fallback to primary if missing
      }
    }
  }

  async function getFontsFor(pdf) {
    await preloadFontBytes();
    const fk = await ensureFontkit();
    if (!pdf._ndFontkitRegistered) {
      pdf.registerFontkit(fk);
      pdf._ndFontkitRegistered = true;
    }
    // subset:false for stable width measurements
    const latin = await pdf.embedFont(FONT_REG_BYTES, { subset: false });
    const latinBold = await pdf.embedFont(FONT_BOLD_BYTES, { subset: false });
    const tamil = await pdf.embedFont(FONT_TAMIL_REG_BYTES, { subset: false });
    const tamilBold = await pdf.embedFont(FONT_TAMIL_BOLD_BYTES, { subset: false });
    const sym = FONT_SYM_BYTES ? await pdf.embedFont(FONT_SYM_BYTES, { subset: false }) : latin;

    return {
      // Regular-weight cascade
      fonts: [latin, tamil, sym],
      // Bold-weight cascade
      fontsBold: [latinBold, tamilBold, sym],
      // single-font convenience
      font: latin,
      fontBold: latinBold,
      fontSym: sym,
    };
  }

  // Cascading width measure (use best supporting font per char)
  function measureWidthWithCascade(text, size, fontList) {
    if (!text) return 0;
    const hasGlyph = (f, cp) => (f.hasGlyphForUnicode ? f.hasGlyphForUnicode(cp) : true);
    let w = 0;
    for (const ch of Array.from(text)) {
      const cp = ch.codePointAt(0);
      const f = fontList.find((fn) => hasGlyph(fn, cp)) || fontList[0];
      w += f.widthOfTextAtSize(ch, size);
    }
    return w;
  }

  // Unicode-aware wrapper with cascading fallback array (e.g., [Latin, Tamil, Symbols])
  function drawUnicodeWrappedMulti(page, text, x, y, width, size, leading, fontList, color) {
    const hasGlyph = (f, cp) => (f.hasGlyphForUnicode ? f.hasGlyphForUnicode(cp) : true);
    const chars = Array.from(text || "");
    let ty = y,
      line = "",
      currentFont = fontList[0] || null;

    function flush() {
      if (!line) return;
      page.drawText(line, { x, y: ty, size, font: currentFont, color });
      ty -= leading;
      line = "";
      currentFont = fontList[0] || null;
    }

    for (const ch of chars) {
      const cp = ch.codePointAt(0);
      const f = fontList.find((fn) => hasGlyph(fn, cp)) || fontList[0];

      if (!line) currentFont = f;
      if (f !== currentFont) {
        flush(); // different font mid-line → flush to keep widths correct
        currentFont = f;
      }

      const candidate = line + ch;
      const w = currentFont.widthOfTextAtSize(candidate, size);
      if (w > width) {
        flush();
        line = ch;
      } else {
        line = candidate;
      }
    }
    if (line) {
      page.drawText(line, { x, y: ty, size, font: currentFont, color });
      ty -= leading;
    }
    return ty;
  }

  /* --------------------------- Small utilities -------------------------- */
  async function fileToBytes(file) {
    const buf = await file.arrayBuffer();
    return new Uint8Array(buf);
  }
  function copyBytes(u8) {
    return u8.slice(0);
  }
  function saveBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function downloadPdf(pdfDoc, filename, kind) {
    const bytes = await pdfDoc.save();
    saveBlob(new Blob([bytes], { type: "application/pdf" }), filename);
    showMotivator(kind);
  }

  // ---------- NEW: server fetch helpers ----------
  async function ndFetchJSON(url, options = {}) {
    const res = await fetch(url, options);
    if (!res.ok) {
      const msg = await res.text().catch(() => res.statusText);
      throw new Error(msg || `HTTP ${res.status}`);
    }
    return res.json();
  }
  async function ndFetchBlob(url, options = {}) {
    const res = await fetch(url, options);
    if (!res.ok) {
      const msg = await res.text().catch(() => res.statusText);
      throw new Error(msg || `HTTP ${res.status}`);
    }
    return res.blob();
  }

  /* ----------------------- ==== PDF tool handlers ==== ----------------------- */

  /* DOCS: Note → PDF (SERVER) */
  (function () {
    const noteBtn = document.getElementById("noteBtn");
    const noteInput = document.getElementById("noteInput");
    const noteTitle = document.getElementById("noteTitle");
    const noteStatus = document.getElementById("noteStatus");

    noteBtn?.addEventListener("click", async () => {
      const text = (noteInput?.value || "").trim();
      if (!text) {
        if (noteStatus) noteStatus.textContent = "Type something first.";
        return;
      }
      try {
        if (noteStatus) noteStatus.textContent = "Creating PDF on server…";
        const payload = {
          title: (noteTitle?.value || "My Note").trim(),
          body: text,
          page_size: "A4",
        };
        const blob = await ndFetchBlob("/api/note-pdf", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        saveBlob(blob, `NiceDay_Note_${Date.now()}.pdf`);
        if (noteStatus) noteStatus.textContent = "Done ✅";
        showMotivator("note");
      } catch (e) {
        console.error(e);
        if (noteStatus) noteStatus.textContent = `Error: ${e.message}`;
      }
    });
  })();

  /* DOCS: Text/Markdown files → PDF (Unicode-safe, client-side) */
  (function () {
    const btn = document.getElementById("txtFilesBtn");
    const input = document.getElementById("txtFilesInput");
    const statusEl = document.getElementById("txtFilesStatus");

    btn?.addEventListener("click", async () => {
      if (!input?.files?.length) {
        statusEl && (statusEl.textContent = "Choose .txt or .md files.");
        return;
      }
      try {
        const pdf = await PDFLib.PDFDocument.create();
        const { fonts, fontsBold } = await getFontsFor(pdf);
        for (const f of input.files) {
          const text = await f.text();
          let page = pdf.addPage([595.28, 841.89]);
          const margin = 50,
            width = page.getSize().width - margin * 2;
          let y = page.getSize().height - margin - 20;

          page.drawText(f.name, {
            x: margin,
            y,
            size: 16,
            font: fontsBold[0],
            color: PDFLib.rgb(0.1, 0.1, 0.1),
          });
          y -= 24;

          const blocks = text.split(/\n{2,}/);
          for (const b of blocks) {
            const body = b.replace(/^#.*$/gm, "").replace(/\*\*(.*?)\*\*/g, "$1").split(/\n/).join(" ");
            y = drawUnicodeWrappedMulti(
              page,
              body,
              margin,
              y,
              width,
              12,
              16,
              fonts,
              PDFLib.rgb(0.2, 0.2, 0.2)
            ) - 18;
            if (y < 60) {
              page = pdf.addPage([595.28, 841.89]);
              y = page.getSize().height - margin;
            }
          }
        }
        await downloadPdf(pdf, `NiceDay_Text_${Date.now()}.pdf`, "text"); // fixed Date.now()
        statusEl && (statusEl.textContent = "Done.");
      } catch (e) {
        console.error(e);
        statusEl && (statusEl.textContent = "Error creating PDF.");
      }
    });
  })();

  /* DOCS: Document Viewer (SERVER) */
  (function attachViewer() {
    const input = document.getElementById("viewUpload");
    const btn = document.getElementById("viewBtn");
    const frame = document.getElementById("viewerFrame");
    const statusEl = document.getElementById("viewStatus");
    if (!input || !btn || !frame) return;

    btn.addEventListener("click", async () => {
      try {
        if (!input.files || !input.files.length) {
          statusEl && (statusEl.textContent = "Please choose a file.");
          return;
        }
        statusEl && (statusEl.textContent = "Uploading…");
        const fd = new FormData();
        fd.append("file", input.files[0]);
        const res = await ndFetchJSON("/api/view/upload", { method: "POST", body: fd });
        frame.src = res.url; // Inline preview via browser PDF/image viewer
        statusEl && (statusEl.textContent = "Preview ready ✅");
      } catch (e) {
        statusEl && (statusEl.textContent = `Error: ${e.message}`);
      }
    });
  })();

  /* CONVERT: Images → PDF (client-side) */
  (function () {
    const btn = document.getElementById("imgBtn");
    const input = document.getElementById("imgInput");
    const statusEl = document.getElementById("imgStatus");

    btn?.addEventListener("click", async () => {
      if (!input?.files?.length) {
        statusEl && (statusEl.textContent = "Choose images.");
        return;
      }
      try {
        const out = await PDFLib.PDFDocument.create();
        for (const f of input.files) {
          const bytes = await f.arrayBuffer();
          const isPng = (f.type || "").toLowerCase().includes("png");
          const img = isPng ? await out.embedPng(bytes) : await out.embedJpg(bytes);
          const page = out.addPage([595.28, 841.89]);
          const maxW = 595.28 - 60,
            maxH = 841.89 - 60;
          const r = Math.min(maxW / img.width, maxH / img.height);
          const w = img.width * r,
            h = img.height * r;
          page.drawImage(img, { x: (595.28 - w) / 2, y: (841.89 - h) / 2, width: w, height: h });
        }
        await downloadPdf(out, `NiceDay_Images_${Date.now()}.pdf`, "images");
        statusEl && (statusEl.textContent = "Done.");

              } catch (e) {
        console.error(e);
        statusEl && (statusEl.textContent = "Error creating PDF.");
      }
    });
  })();

  /* CONVERT: PDF → Images (ZIP) */
  (function () {
    const btn = document.getElementById("pdf2imgBtn");
    const input = document.getElementById("pdf2imgInput");
    const formatSel = document.getElementById("pdf2imgFormat");
    const scaleSel = document.getElementById("pdf2imgScale");
    const statusEl = document.getElementById("pdf2imgStatus");

    btn?.addEventListener("click", async () => {
      const file = input?.files && input.files[0];
      if (!file) {
        statusEl && (statusEl.textContent = "Choose a PDF.");
        return;
      }
      statusEl && (statusEl.textContent = "Rendering pages…");
      try {
        const data = new Uint8Array(await file.arrayBuffer());
        const pdf = await pdfjsLib.getDocument({ data }).promise;
        const zip = new JSZip();
        const format = formatSel?.value || "png"; // png or jpeg
        const scale = parseFloat(scaleSel?.value || "1");

        for (let n = 1; n <= pdf.numPages; n++) {
          const page = await pdf.getPage(n);
          const viewport = page.getViewport({ scale: 2 * scale });
          const canvas = document.createElement("canvas");
          const ctx = canvas.getContext("2d");
          canvas.width = viewport.width;
          canvas.height = viewport.height;
          await page.render({ canvasContext: ctx, viewport }).promise;

          const mime = format === "jpeg" ? "image/jpeg" : "image/png";
          const dataURL = canvas.toDataURL(mime, 0.92);
          const base64 = dataURL.split(",")[1];
          const fname = `page_${String(n).padStart(3, "0")}.${format === "jpeg" ? "jpg" : "png"}`;
          zip.file(fname, base64, { base64: true });
        }
        const blob = await zip.generateAsync({ type: "blob" });
        saveBlob(blob, `NiceDay_Pages_${Date.now()}.zip`);
        statusEl && (statusEl.textContent = "Done.");
        showMotivator("pdf2img");
      } catch (e) {
        console.error(e);
        statusEl && (statusEl.textContent = "Error exporting images.");
      }
    });
  })();

  /* CONVERT: Document Scanner (SERVER) */
  (function attachScanner() {
    const input = document.getElementById("scanInput");
    const btn = document.getElementById("scanBtn");
    const statusEl = document.getElementById("scanStatus");
    const deskewEl = document.getElementById("scanDeskew");
    const threshEl = document.getElementById("scanThresh");
    const ocrEl = document.getElementById("scanOCR");
    if (!input || !btn) return;

    btn.addEventListener("click", async () => {
      try {
        if (!input.files || !input.files.length) {
          statusEl && (statusEl.textContent = "Please choose one PDF or one/more images.");
          return;
        }
        statusEl && (statusEl.textContent = "Scanning on server…");
        const fd = new FormData();
        // If first file is a PDF, send only that (server expects either ONE PDF or many images)
        const first = input.files[0];
        if (first.type === "application/pdf") {
          fd.append("files", first);
        } else {
          for (const f of input.files) fd.append("files", f);
        }
        const qs = new URLSearchParams({
          deskew: deskewEl?.checked ? "1" : "0",
          threshold: threshEl?.checked ? "1" : "0",
          ocr: ocrEl?.checked ? "1" : "0",
        }).toString();

        const blob = await ndFetchBlob(`/api/scan?${qs}`, { method: "POST", body: fd });
        saveBlob(blob, `NiceDay_Scanned_${Date.now()}.pdf`);
        statusEl && (statusEl.textContent = "Downloaded ✅");
        showMotivator("images");
      } catch (e) {
        statusEl && (statusEl.textContent = `Error: ${e.message}`);
      }
    });
  })();

  /* CONVERT: Compress PDF (rasterize) */
  (function () {
    const btn = document.getElementById("cmpBtn");
    const input = document.getElementById("cmpInput");
    const scaleSel = document.getElementById("cmpScale");
    const qualityInput = document.getElementById("cmpQuality");
    const fmtSel = document.getElementById("cmpFormat");
    const statusEl = document.getElementById("cmpStatus");

    btn?.addEventListener("click", async () => {
      if (!input?.files?.length) {
        statusEl && (statusEl.textContent = "Choose a PDF.");
        return;
      }
      statusEl && (statusEl.textContent = "Compressing…");
      try {
        const scale = parseFloat(scaleSel?.value || "1");
        const quality = Math.min(0.95, Math.max(0.3, parseFloat(qualityInput?.value || "0.8")));
        const fmt = fmtSel?.value || "jpeg";
        const data = new Uint8Array(await input.files[0].arrayBuffer());
        const pdf = await pdfjsLib.getDocument({ data }).promise;

        const out = await PDFLib.PDFDocument.create();
        for (let n = 1; n <= pdf.numPages; n++) {
          const page = await pdf.getPage(n);
          const viewport = page.getViewport({ scale: 2 * scale });
          const canvas = document.createElement("canvas");
          const ctx = canvas.getContext("2d", { willReadFrequently: true });
          canvas.width = viewport.width;
          canvas.height = viewport.height;
          await page.render({ canvasContext: ctx, viewport }).promise;

          let bytes, img;
          if (fmt === "png") {
            const b64 = canvas.toDataURL("image/png").split(",")[1];
            bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
            img = await out.embedPng(bytes);
          } else {
            const b64 = canvas.toDataURL("image/jpeg", quality).split(",")[1];
            bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
            img = await out.embedJpg(bytes);
          }
          const pw = img.width,
            ph = img.height;
          const pageOut = out.addPage([pw, ph]);
          pageOut.drawImage(img, { x: 0, y: 0, width: pw, height: ph });
        }
        await downloadPdf(out, `NiceDay_Compressed_${Date.now()}.pdf`, "compress");
        statusEl && (statusEl.textContent = "Done.");
      } catch (e) {
        console.error(e);
        statusEl && (statusEl.textContent = "Error compressing.");
      }
    });
  })();

  /* ORGANIZE: Merge PDFs */
  (function () {
    const btn = document.getElementById("mergeBtn");
    const input = document.getElementById("mergeInput");
    const statusEl = document.getElementById("mergeStatus");

    btn?.addEventListener("click", async () => {
      if (!input?.files || input.files.length < 2) {
        statusEl && (statusEl.textContent = "Select at least 2 PDFs.");
        return;
      }
      statusEl && (statusEl.textContent = "Merging…");
      try {
        const out = await PDFLib.PDFDocument.create();
        for (const f of input.files) {
          const bytes = await f.arrayBuffer();
          const src = await PDFLib.PDFDocument.load(bytes);
          const pages = await out.copyPages(src, src.getPageIndices());
          pages.forEach((p) => out.addPage(p));
        }
        await downloadPdf(out, `NiceDay_Merged_${Date.now()}.pdf`, "merge");
        statusEl && (statusEl.textContent = "Done.");
      } catch (e) {
        console.error(e);
        statusEl && (statusEl.textContent = "Error merging.");
      }
    });
  })();

  /* ORGANIZE: Split PDF */
  (function () {
    const btn = document.getElementById("splitBtn");
    const input = document.getElementById("splitInput");
    const rangesEl = document.getElementById("splitRanges");
    const statusEl = document.getElementById("splitStatus");

    function parseRanges(text) {
      const parts = String(text || "")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      const out = [];
      for (const p of parts) {
        if (p.includes("-")) {
          let [a, b] = p.split("-").map((n) => parseInt(n, 10));
          if (!isNaN(a) && !isNaN(b)) out.push([a, b]);
        } else {
          let n = parseInt(p, 10);
          if (!isNaN(n)) out.push([n, n]);
        }
      }
      return out;
    }

    btn?.addEventListener("click", async () => {
      if (!input?.files?.length) {
        statusEl && (statusEl.textContent = "Choose a PDF.");
        return;
      }
      const ranges = parseRanges(rangesEl?.value);
      if (!ranges.length) {
        statusEl && (statusEl.textContent = "Enter valid ranges.");
        return;
      }
      try {
        const bytes = await input.files[0].arrayBuffer();
        const src = await PDFLib.PDFDocument.load(bytes);
        const total = src.getPageCount();
        for (const [a, b] of ranges) {
          const start = Math.max(1, a),
            end = Math.min(total, b);
          const out = await PDFLib.PDFDocument.create();
          const idxs = Array.from({ length: end - start + 1 }, (_, i) => start - 1 + i);
          const pages = await out.copyPages(src, idxs);
          pages.forEach((p) => out.addPage(p));
          await downloadPdf(out, `NiceDay_Split_${start}-${end}_${Date.now()}.pdf`, "split");
        }
        statusEl && (statusEl.textContent = "Done.");
      } catch (e) {
        console.error(e);
        statusEl && (statusEl.textContent = "Error splitting.");
      }
    });
  })();

  /* ORGANIZE: Reorder Pages (Drag & Drop) */
  (function () {
    const input = document.getElementById("reorderDnDInput");
    const grid = document.getElementById("reorderGrid");
    const btnReset = document.getElementById("reorderReset");
    const btnExport = document.getElementById("reorderExport");
    const statusEl = document.getElementById("reorderDnDStatus");

    const reorderState = { order: [], originalBytes: null, pageCount: 0 };

    function copyBytes(u8) {
      return u8.slice(0);
    }

    input?.addEventListener("change", async () => {
      if (!grid) return;
      grid.innerHTML = "";
      statusEl && (statusEl.textContent = "");
      const file = input.files?.[0];
      if (!file) return;

      reorderState.originalBytes = new Uint8Array(await file.arrayBuffer());
      const pdf = await pdfjsLib.getDocument({ data: copyBytes(reorderState.originalBytes) }).promise;
      reorderState.pageCount = pdf.numPages;
      reorderState.order = Array.from({ length: pdf.numPages }, (_, i) => i + 1);

      for (let n = 1; n <= pdf.numPages; n++) {
        const page = await pdf.getPage(n);
        const viewport = page.getViewport({ scale: 0.2 });
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        await page.render({ canvasContext: ctx, viewport }).promise;

        const item = document.createElement("div");
        item.className = "border rounded-lg p-1 text-center select-none bg-white";
        item.dataset.page = n;

        const img = document.createElement("img");
        img.src = canvas.toDataURL("image/png");
        img.alt = `Page ${n} thumbnail`;
        img.className = "thumb w-full rounded";
        const label = document.createElement("div");
        label.className = "text-xs mt-1 text-gray-600";
        label.textContent = `Page ${n}`;

        item.appendChild(img);
        item.appendChild(label);
        grid.appendChild(item);
      }

      new Sortable(grid, {
        animation: 150,
        ghostClass: "drag-ghost",
        onSort: () => {
          reorderState.order = Array.from(grid.children).map((n) => parseInt(n.dataset.page, 10));
        },
      });
    });

    btnReset?.addEventListener("click", () => {
      if (!reorderState.pageCount || !grid) return;
      const items = Array.from(grid.children).sort(
        (a, b) => parseInt(a.dataset.page) - parseInt(b.dataset.page)
      );
      items.forEach((i) => grid.appendChild(i));
      reorderState.order = Array.from({ length: reorderState.pageCount }, (_, i) => i + 1);
    });

    btnExport?.addEventListener("click", async () => {
      if (!reorderState.originalBytes) {
        statusEl && (statusEl.textContent = "Upload a PDF first.");
        return;
      }
      try {
        const src = await PDFLib.PDFDocument.load(copyBytes(reorderState.originalBytes));
        const out = await PDFLib.PDFDocument.create();
        const total = src.getPageCount();
        const order = reorderState.order.length
          ? reorderState.order
          : Array.from({ length: total }, (_, i) => i + 1);
        const zeroIdx = order.map((n) => Math.min(total, Math.max(1, n)) - 1);
        const pages = await out.copyPages(src, zeroIdx);
        pages.forEach((p) => out.addPage(p));
        await downloadPdf(out, `NiceDay_Reordered_${Date.now()}.pdf`, "reorder");
        statusEl && (statusEl.textContent = "Done.");
      } catch (err) {
        console.error(err);
        statusEl && (statusEl.textContent = "Error exporting.");
      }
    });
  })();

  /* EDIT: Delete Pages */
  (function () {
    const btn = document.getElementById("delBtn");
    const input = document.getElementById("delInput");
    const listEl = document.getElementById("delList");
    const statusEl = document.getElementById("delStatus");

    function parseRanges(text) {
      const parts = String(text || "")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      const out = [];
      for (const p of parts) {
        if (p.includes("-")) {
          let [a, b] = p.split("-").map((n) => parseInt(n, 10));
          if (!isNaN(a) && !isNaN(b)) out.push([a, b]);
        } else {
          let n = parseInt(p, 10);
          if (!isNaN(n)) out.push([n, n]);
        }
      }
      return out;
    }

    btn?.addEventListener("click", async () => {
      if (!input?.files?.length) {
        statusEl && (statusEl.textContent = "Choose a PDF.");
        return;
      }
      try {
        const bytes = await input.files[0].arrayBuffer();
        const src = await PDFLib.PDFDocument.load(bytes);
        const total = src.getPageCount();
        const delRanges = parseRanges(listEl?.value);
        const toDel = new Set();
        for (const [a, b] of delRanges) {
          const s = Math.max(1, a),
            e = Math.min(total, b);
          for (let i = s; i <= e; i++) toDel.add(i - 1);
        }
        const out = await PDFLib.PDFDocument.create();
        const keepIdx = src.getPageIndices().filter((i) => !toDel.has(i));
        const pages = await out.copyPages(src, keepIdx);
        pages.forEach((p) => out.addPage(p));
        await downloadPdf(out, `NiceDay_Deleted_${Date.now()}.pdf`, "del");
        statusEl && (statusEl.textContent = "Done.");
      } catch (e) {
        console.error(e);
        statusEl && (statusEl.textContent = "Error deleting.");
      }
    });
  })();

  /* EDIT: Extract Pages */
  (function () {
    const btn = document.getElementById("extBtn");
    const input = document.getElementById("extInput");
    const listEl = document.getElementById("extList");
    const statusEl = document.getElementById("extStatus");

    function parseRanges(text) {
      const parts = String(text || "")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      const out = [];
      for (const p of parts) {
        if (p.includes("-")) {
          let [a, b] = p.split("-").map((n) => parseInt(n, 10));
          if (!isNaN(a) && !isNaN(b)) out.push([a, b]);
        } else {
          let n = parseInt(p, 10);
          if (!isNaN(n)) out.push([n, n]);
        }
      }
      return out;
    }

    btn?.addEventListener("click", async () => {
      if (!input?.files?.length) {
        statusEl && (statusEl.textContent = "Choose a PDF.");
        return;
      }
      try {
        const bytes = await input.files[0].arrayBuffer();
        const src = await PDFLib.PDFDocument.load(bytes);
        const total = src.getPageCount();
        const ranges = parseRanges(listEl?.value);
        if (!ranges.length) {
          statusEl && (statusEl.textContent = "Enter valid ranges.");
          return;
        }
        const out = await PDFLib.PDFDocument.create();
        const idxs = [];
        for (const [a, b] of ranges) {
          const s = Math.max(1, a),
            e = Math.min(total, b);
          for (let i = s; i <= e; i++) idxs.push(i - 1);
        }
        const pages = await out.copyPages(src, idxs);
        pages.forEach((p) => out.addPage(p));
        await downloadPdf(out, `NiceDay_Extract_${Date.now()}.pdf`, "extract");
        statusEl && (statusEl.textContent = "Done.");
      } catch (e) {
        console.error(e);
        statusEl && (statusEl.textContent = "Error extracting.");
      }
    });
  })();

  /* EDIT: Rotate Pages */
  (function () {
    const btn = document.getElementById("rotBtn");
    const input = document.getElementById("rotInput");
    const rangesEl = document.getElementById("rotRanges");
    const angleEl = document.getElementById("rotAngle");
    const statusEl = document.getElementById("rotStatus");

    function parseRanges(text) {
      const parts = String(text || "")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      const out = [];
      for (const p of parts) {
        if (p.includes("-")) {
          let [a, b] = p.split("-").map((n) => parseInt(n, 10));
          if (!isNaN(a) && !isNaN(b)) out.push([a, b]);
        } else {
          let n = parseInt(p, 10);
          if (!isNaN(n)) out.push([n, n]);
        }
      }
      return out;
    }

    btn?.addEventListener("click", async () => {
      if (!input?.files?.length) {
        statusEl && (statusEl.textContent = "Choose a PDF.");
        return;
      }
      const angle = parseInt(angleEl?.value || "0", 10);
      const ranges = parseRanges(rangesEl?.value);
      if (![90, 180, 270].includes(angle) || !ranges.length) {
        statusEl && (statusEl.textContent = "Angle 90/180/270 + valid ranges.");
        return;
      }
      try {
        const bytes = await input.files[0].arrayBuffer();
        const src = await PDFLib.PDFDocument.load(bytes);
        const out = await PDFLib.PDFDocument.create();
        const total = src.getPageCount();
        const targets = new Set();
        for (const [a, b] of ranges) {
          const s = Math.max(1, a),
            e = Math.min(total, b);
          for (let i = s; i <= e; i++) targets.add(i - 1);
        }
        const pages = await out.copyPages(src, src.getPageIndices());
        pages.forEach((p, idx) => {
          if (targets.has(idx)) p.setRotation(PDFLib.degrees(angle));
          out.addPage(p);
        });
        await downloadPdf(out, `NiceDay_Rotated_${Date.now()}.pdf`, "rotate");
        statusEl && (statusEl.textContent = "Done.");
      } catch (e) {
        console.error(e);
        statusEl && (statusEl.textContent = "Error rotating.");
      }
    });
  })();

  /* EDIT: Add Page Numbers (Unicode-safe) */
  (function () {
    const btn = document.getElementById("numBtn");
    const input = document.getElementById("numInput");
    const posSel = document.getElementById("numPos");
    const marginEl = document.getElementById("numMargin");
    const statusEl = document.getElementById("numStatus");

    btn?.addEventListener("click", async () => {
      if (!input?.files?.length) {
        statusEl && (statusEl.textContent = "Choose a PDF.");
        return;
      }
      try {
        const bytes = await input.files[0].arrayBuffer();
        const src = await PDFLib.PDFDocument.load(bytes);
        const out = await PDFLib.PDFDocument.create();
        const { fonts } = await getFontsFor(out); // head of cascade for digits
        const pages = await out.copyPages(src, src.getPageIndices());
        const margin = parseFloat(marginEl?.value || "24");
        const pos = posSel?.value || "br";
        const size = 10;
        pages.forEach((p, i) => {
          out.addPage(p);
          const { width, height } = p.getSize();
          const text = `${i + 1} / ${pages.length}`;
          const tw = measureWidthWithCascade(text, size, fonts);
          let x = 0,
            y = 0,
            m = margin;
          if (pos === "br") {
            x = width - tw - m;
            y = m;
          }
          if (pos === "bc") {
            x = (width - tw) / 2;
            y = m;
          }
          if (pos === "bl") {
            x = m;
            y = m;
          }
          if (pos === "tr") {
            x = width - tw - m;
            y = height - size - m;
          }
          if (pos === "tc") {
            x = (width - tw) / 2;
            y = height - size - m;
          }
          if (pos === "tl") {
            x = m;
            y = height - size - m;
          }
          const numFont = fonts[0];
          p.drawText(text, { x, y, size, font: numFont });
        });
        await downloadPdf(out, `NiceDay_Numbered_${Date.now()}.pdf`, "numbers");
        statusEl && (statusEl.textContent = "Done.");
      } catch (e) {
        console.error(e);
        statusEl && (statusEl.textContent = "Error numbering.");
      }
    });
  })();

  /* EDIT: Watermark (Unicode-safe bold) */
  (function () {
    const btn = document.getElementById("wmBtn");
    const input = document.getElementById("wmInput");
    const txtEl = document.getElementById("wmText");
    const sizeEl = document.getElementById("wmSize");
    const angleEl = document.getElementById("wmAngle");
    const alphaEl = document.getElementById("wmAlpha");
    const statusEl = document.getElementById("wmStatus");

    btn?.addEventListener("click", async () => {
      if (!input?.files?.length) {
        statusEl && (statusEl.textContent = "Choose a PDF.");
        return;
      }
      const text = (txtEl?.value || "CONFIDENTIAL").trim();
      const angle = parseFloat(angleEl?.value || "45");
      const alpha = Math.max(0, Math.min(1, parseFloat(alphaEl?.value || "0.15")));
      const fontSize = Math.max(16, parseInt(sizeEl?.value || "64", 10));
      try {
        const bytes = await input.files[0].arrayBuffer();
        const src = await PDFLib.PDFDocument.load(bytes);
        const out = await PDFLib.PDFDocument.create();
        const { fontsBold } = await getFontsFor(out);
        const pages = await out.copyPages(src, src.getPageIndices());
        pages.forEach((p) => {
          out.addPage(p);
          const { width, height } = p.getSize();
          const tw = measureWidthWithCascade(text, fontSize, fontsBold);
          const drawFont = fontsBold[0];
          p.drawText(text, {
            x: (width - tw) / 2,
            y: height / 2,
            size: fontSize,
            font: drawFont,
            color: PDFLib.rgb(0.2, 0.2, 0.2),
            rotate: PDFLib.degrees(angle),
            opacity: alpha,
          });
        });
        await downloadPdf(out, `NiceDay_Watermark_${Date.now()}.pdf`, "watermark");
        statusEl && (statusEl.textContent = "Done.");
      } catch (e) {
        console.error(e);
        statusEl && (statusEl.textContent = "Error watermarking.");
      }
    });
  })();

  /* SECURE: Strip Metadata */
  (function () {
    const btn = document.getElementById("metaBtn");
    const input = document.getElementById("metaInput");
    const statusEl = document.getElementById("metaStatus");

    btn?.addEventListener("click", async () => {
      if (!input?.files?.length) {
        statusEl && (statusEl.textContent = "Choose a PDF.");
        return;
      }
      try {
        const bytes = await input.files[0].arrayBuffer();
        const src = await PDFLib.PDFDocument.load(bytes);
        const out = await PDFLib.PDFDocument.create();
        const pages = await out.copyPages(src, src.getPageIndices());
        pages.forEach((p) => out.addPage(p));
        await downloadPdf(out, `NiceDay_Clean_${Date.now()}.pdf`, "meta");
        statusEl && (statusEl.textContent = "Done.");
      } catch (e) {
        console.error(e);
        statusEl && (statusEl.textContent = "Error cleaning.");
      }
    });
  })();

  /* SECURE: Redact (draw boxes) */
  (function () {
    const rInput = document.getElementById("redactInput");
    const rCanvas = document.getElementById("redactStage");
    const rPrev = document.getElementById("redPrev");
    const rNext = document.getElementById("redNext");
    const rClear = document.getElementById("redClear");
    const rApply = document.getElementById("redApply");
    const rInfo = document.getElementById("redInfo");
    const rStatus = document.getElementById("redStatus");

    let rPdf = null,
      rPage = 1;
    const rectsByPage = [];
    const renderState = Object.create(null);

    function drawCurrentPage() {
      if (!rPdf || !rCanvas) return;
      rPdf.getPage(rPage).then(async (page) => {
        const viewport = page.getViewport({ scale: 1.5 });
        rCanvas.width = viewport.width;
        rCanvas.height = viewport.height;
        const ctx = rCanvas.getContext("2d");
        ctx.clearRect(0, 0, rCanvas.width, rCanvas.height);
        await page.render({ canvasContext: ctx, viewport }).promise;
        renderState[rPage] = { canvasW: rCanvas.width, canvasH: rCanvas.height };
        const rects = rectsByPage[rPage] || [];
        ctx.fillStyle = "rgba(0,0,0,0.45)";
        rects.forEach((rc) => ctx.fillRect(rc.x, rc.y, rc.w, rc.h));
        rInfo && (rInfo.textContent = `Page ${rPage} / ${rPdf.numPages}`);
      });
    }

    rInput?.addEventListener("change", async () => {
      if (!rCanvas) return;
      rStatus && (rStatus.textContent = "");
      const f = rInput.files?.[0];
      if (!f) return;
      const bytes = new Uint8Array(await f.arrayBuffer());
      rCanvas._ndOriginalBytes = bytes.slice(0);
      rPdf = await pdfjsLib.getDocument({ data: bytes.slice(0) }).promise;
      rectsByPage.length = rPdf.numPages + 1;
      rPage = 1;
      drawCurrentPage();
    });

    rPrev?.addEventListener("click", () => {
      if (!rPdf) return;
      rPage = Math.max(1, rPage - 1);
      drawCurrentPage();
    });
    rNext?.addEventListener("click", () => {
      if (!rPdf) return;
      rPage = Math.min(rPdf.numPages, rPage + 1);
      drawCurrentPage();
    });
    rClear?.addEventListener("click", () => {
      if (!rPdf) return;
      rectsByPage[rPage] = [];
      drawCurrentPage();
    });

    // Draw rectangles with mouse
    (function enableRectDrawing() {
      if (!rCanvas) return;
      let start = null,
        current = null,
        drawing = false;
      rCanvas.addEventListener("mousedown", (e) => {
        if (!rPdf) return;
        const rect = rCanvas.getBoundingClientRect();
        start = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        drawing = true;
      });
      rCanvas.addEventListener("mousemove", (e) => {
        if (!drawing) return;
        const rect = rCanvas.getBoundingClientRect();
        current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
        drawCurrentPage();
        const ctx = rCanvas.getContext("2d");
        ctx.fillStyle = "rgba(0,0,0,0.45)";
        const x = Math.min(start.x, current.x),
          y = Math.min(start.y, current.y);
        const w = Math.abs(start.x - current.x),
          h = Math.abs(start.y - current.y);
        ctx.fillRect(x, y, w, h);
      });
      window.addEventListener("mouseup", () => {
        if (!drawing) return;
        drawing = false;
        if (!start || !current) return;
        const x = Math.min(start.x, current.x),
          y = Math.min(start.y, current.y);
        const w = Math.abs(start.x - current.x),
          h = Math.abs(start.y - current.y);
        if (!rectsByPage[rPage]) rectsByPage[rPage] = [];
        rectsByPage[rPage].push({ x, y, w, h });
        drawCurrentPage();
      });
    })();

    rApply?.addEventListener("click", async () => {
      if (!rCanvas || !rCanvas._ndOriginalBytes) {
        rStatus && (rStatus.textContent = "Load a PDF first.");
        return;
      }
      try {
        rStatus && (rStatus.textContent = "Applying…");
        const src = await PDFLib.PDFDocument.load(rCanvas._ndOriginalBytes.slice(0), {
          ignoreEncryption: true,
        });
        const out = await PDFLib.PDFDocument.create();
        const idxs = src.getPageIndices();
        const copied = await out.copyPages(src, idxs);
        copied.forEach((p) => out.addPage(p));

        function dimsFor(i1) {
          const rs = renderState[i1];
          if (rs && Number.isFinite(rs.canvasW) && Number.isFinite(rs.canvasH)) {
            return { w: rs.canvasW, h: rs.canvasH };
          }
          return { w: Math.max(1, rCanvas.width), h: Math.max(1, rCanvas.height) };
        }

        for (let i = 0; i < copied.length; i++) {
          const p = out.getPage(i);
          const { width, height } = p.getSize();
          const list = rectsByPage[i + 1] || [];
          if (!list.length) continue;

          const dims = dimsFor(i + 1);
          const scaleX = width / dims.w;
          const scaleY = height / dims.h;

          for (const rc of list) {
            const x = Math.max(0, (rc?.x ?? 0) * scaleX);
            const y = Math.max(0, (dims.h - (rc?.y ?? 0) - (rc?.h ?? 0)) * scaleY);
            const w = Math.max(0, (rc?.w ?? 0) * scaleX);
            const h = Math.max(0, (rc?.h ?? 0) * scaleY);
            if (!Number.isFinite(x + y + w + h)) continue;
            p.drawRectangle({ x, y, width: w, height: h, color: PDFLib.rgb(0, 0, 0) });
          }
        }

        const bytes = await out.save();
        saveBlob(new Blob([bytes], { type: "application/pdf" }), `NiceDay_Redacted_${Date.now()}.pdf`);
        rStatus && (rStatus.textContent = "Done.");
        showMotivator("redact");
      } catch (err) {
        console.error("Redact apply error:", err);
        rStatus && (rStatus.textContent = "Error applying redactions.");
      }
    });
  })();

  /* ------------------------- kick off if ready -------------------------- */
  window.__ndTryStart?.();
});

     
