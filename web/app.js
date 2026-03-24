(() => {
  const page = document.body.dataset.page;
  if (page === "input") {
    initInputPage();
  } else if (page === "canvas") {
    initCanvasPage();
  }

  function $(id) {
    return document.getElementById(id);
  }

  function initInputPage() {
    const confirmBtn = $("confirmBtn");
    const errorMsg = $("errorMsg");
    const uploadZone = $("uploadZone");
    const referenceFile = $("referenceFile");
    const referencePreview = $("referencePreview");
    const referenceStatus = $("referenceStatus");
    const samBackend = $("samBackend");
    const samApiKeyGroup = $("samApiKeyGroup");
    const samApiKeyInput = $("samApiKey");
    const inputImageZone = $("inputImageZone");
    const inputImageFile = $("inputImageFile");
    const inputImagePreview = $("inputImagePreview");
    const inputImageStatus = $("inputImageStatus");
    const inputImageNote = $("inputImageNote");
    const inputImageText = $("inputImageText");
    const imageGenSection = $("imageGenSection");
    let uploadedReferencePath = null;
    let uploadedInputImagePath = null;

    function syncSamApiKeyVisibility() {
      const shouldShow =
        samBackend &&
        (samBackend.value === "fal" || samBackend.value === "roboflow");
      if (samApiKeyGroup) {
        samApiKeyGroup.hidden = !shouldShow;
      }
      if (!shouldShow && samApiKeyInput) {
        samApiKeyInput.value = "";
      }
    }

    function syncImageGenVisibility() {
      if (!imageGenSection) return;
      if (uploadedInputImagePath) {
        // Clear image gen fields so autofilled values don't get sent to backend
        const imageApiKey = $("imageApiKey");
        const imageBaseUrl = $("imageBaseUrl");
        const imageModel = $("imageModel");
        if (imageApiKey) imageApiKey.value = "";
        if (imageBaseUrl) imageBaseUrl.value = "";
        if (imageModel) imageModel.value = "";
        imageGenSection.style.opacity = "0.45";
        imageGenSection.style.pointerEvents = "none";
        if (inputImageNote) inputImageNote.style.display = "";
      } else {
        imageGenSection.style.opacity = "";
        imageGenSection.style.pointerEvents = "";
        if (inputImageNote) inputImageNote.style.display = "none";
      }
    }

    if (samBackend) {
      samBackend.addEventListener("change", syncSamApiKeyVisibility);
      syncSamApiKeyVisibility();
    }

    // Reference image upload zone
    if (uploadZone && referenceFile) {
      uploadZone.addEventListener("click", () => referenceFile.click());
      uploadZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        uploadZone.classList.add("dragging");
      });
      uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("dragging");
      });
      uploadZone.addEventListener("drop", async (event) => {
        event.preventDefault();
        uploadZone.classList.remove("dragging");
        const file = event.dataTransfer.files[0];
        if (file) {
          const uploadedPath = await uploadReference(file, confirmBtn, referencePreview, referenceStatus);
          if (uploadedPath) uploadedReferencePath = uploadedPath;
        }
      });
      referenceFile.addEventListener("change", async () => {
        const file = referenceFile.files[0];
        if (file) {
          const uploadedPath = await uploadReference(file, confirmBtn, referencePreview, referenceStatus);
          if (uploadedPath) uploadedReferencePath = uploadedPath;
        }
      });
    }

    // Input image upload zone (skip generation)
    function bindUploadZone(zone, fileInput, onFile) {
      if (!zone || !fileInput) return;
      zone.addEventListener("click", () => fileInput.click());
      zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragging"); });
      zone.addEventListener("dragleave", () => zone.classList.remove("dragging"));
      zone.addEventListener("drop", async (e) => {
        e.preventDefault();
        zone.classList.remove("dragging");
        const file = e.dataTransfer.files[0];
        if (file) await onFile(file);
      });
      fileInput.addEventListener("change", async () => {
        const file = fileInput.files[0];
        if (file) await onFile(file);
      });
    }

    bindUploadZone(inputImageZone, inputImageFile, async (file) => {
      const uploadedPath = await uploadReference(file, confirmBtn, inputImagePreview, inputImageStatus);
      if (uploadedPath) {
        uploadedInputImagePath = uploadedPath;
        if (inputImageZone) inputImageZone.classList.add("has-file");
        if (inputImageText) {
          inputImageText.innerHTML = `<strong style="color:var(--accent-strong)">${file.name}</strong><br/><span style="opacity:0.6;font-size:11px;">\u5df2\u4e0a\u4f20 \u00b7 \u5c06\u8df3\u8fc7 AI \u751f\u56fe\u9636\u6bb5</span>`;
        }
        syncImageGenVisibility();
      }
    });

    syncImageGenVisibility();

    confirmBtn.addEventListener("click", async () => {
      errorMsg.textContent = "";
      let methodText = $("methodText").value.trim();
      
      if (!methodText && !uploadedInputImagePath) {
        errorMsg.textContent = "Please provide method text or upload an input image.";
        return;
      }
      
      if (!methodText && uploadedInputImagePath) {
        methodText = "Direct image upload. No method text provided.";
      }

      confirmBtn.disabled = true;
      confirmBtn.textContent = "Starting...";

      const imageProvider = $("imageProvider").value.trim() || "gemini";
      const imageBaseUrl = $("imageBaseUrl").value.trim() || "";
      const imageApiKey = $("imageApiKey").value.trim() || "";

      const svgProvider = $("svgProvider").value.trim() || "gemini";
      const svgBaseUrl = $("svgBaseUrl").value.trim() || "";
      const svgApiKey = $("svgApiKey").value.trim() || "";

      const samPromptVal = $("samPrompt") ? $("samPrompt").value.trim() : "";

      const payload = {
        method_text: methodText,
        image_provider: imageProvider,
        image_base_url: imageBaseUrl,
        image_api_key: imageApiKey,
        image_model: $("imageModel").value.trim() || null,
        svg_provider: svgProvider,
        svg_base_url: svgBaseUrl,
        svg_api_key: svgApiKey,
        svg_model: $("svgModel").value.trim() || null,
        optimize_iterations: parseInt($("optimizeIterations").value, 10),
        reference_image_path: uploadedReferencePath,
        input_image_path: uploadedInputImagePath || null,
        sam_backend: $("samBackend").value,
        sam_api_key: $("samApiKey").value.trim() || null,
        sam_prompt: samPromptVal || null,
      };
      if (payload.sam_backend === "local") {
        payload.sam_api_key = null;
      }

      try {
        const response = await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || "Request failed");
        }

        const data = await response.json();
        window.location.href = `/canvas.html?job=${encodeURIComponent(data.job_id)}`;
      } catch (err) {
        errorMsg.textContent = err.message || "Failed to start job";
        confirmBtn.disabled = false;
        confirmBtn.textContent = "Confirm → Canvas";
      }
    });
  }

  async function uploadReference(file, confirmBtn, previewEl, statusEl) {
    if (!file.type.startsWith("image/")) {
      statusEl.textContent = "Only image files are supported.";
      return null;
    }

    confirmBtn.disabled = true;
    statusEl.textContent = "Uploading reference...";

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Upload failed");
      }

      const data = await response.json();
      statusEl.textContent = `Using uploaded reference: ${data.name}`;
      if (previewEl) {
        previewEl.src = data.url || "";
        previewEl.classList.add("visible");
      }
      return data.path || null;
    } catch (err) {
      statusEl.textContent = err.message || "Upload failed";
      return null;
    } finally {
      confirmBtn.disabled = false;
    }
  }

  async function initCanvasPage() {
    const params = new URLSearchParams(window.location.search);
    const jobId = params.get("job");
    const statusText = $("statusText");
    const jobIdEl = $("jobId");
    const artifactPanel = $("artifactPanel");
    const artifactList = $("artifactList");
    const toggle = $("artifactToggle");
    const logToggle = $("logToggle");
    const logPanel = $("logPanel");
    const logBody = $("logBody");
    const iframe = $("svgEditorFrame");
    const fallback = $("svgFallback");
    const fallbackObject = $("fallbackObject");
    const saveSvgBtn = $("saveSvgBtn");
    const downloadSvgBtn = $("downloadSvgBtn");
    const exportPngBtn = $("exportPngBtn");
    const exportPptxBtn = $("exportPptxBtn");

    if (!jobId) {
      statusText.textContent = "Missing job id";
      return;
    }

    jobIdEl.textContent = jobId;

    toggle.addEventListener("click", () => {
      artifactPanel.classList.toggle("open");
    });

    logToggle.addEventListener("click", () => {
      logPanel.classList.toggle("open");
    });

    function getCurrentSvgText() {
      try {
        const win = iframe.contentWindow;
        if (win && win.svgEditor) {
          // svgEditor.svgCanvas is the canvas object (correct property name)
          if (win.svgEditor.svgCanvas && typeof win.svgEditor.svgCanvas.getSvgString === "function") {
            return win.svgEditor.svgCanvas.getSvgString();
          }
          if (win.svgEditor.svgCanvas && typeof win.svgEditor.svgCanvas.svgCanvasToString === "function") {
            return win.svgEditor.svgCanvas.svgCanvasToString();
          }
          // Direct method on editor instance (some versions)
          if (typeof win.svgEditor.svgCanvasToString === "function") {
            return win.svgEditor.svgCanvasToString();
          }
        }
        if (win && win.svgCanvas && typeof win.svgCanvas.getSvgString === "function") {
          return win.svgCanvas.getSvgString();
        }
      } catch (e) {
        console.error(e);
      }
      return null;
    }

    saveSvgBtn.addEventListener("click", async () => {
      const svgText = getCurrentSvgText();
      if (!svgText) {
        alert("无法获取 SVG 内容，请确认 SVG-Edit 已加载。");
        return;
      }
      saveSvgBtn.textContent = "Saving…";
      saveSvgBtn.disabled = true;
      try {
        const res = await fetch(`/api/save-svg/${jobId}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ svg: svgText }),
        });
        if (res.ok) {
          saveSvgBtn.textContent = "Saved ✓";
          setTimeout(() => { saveSvgBtn.textContent = "Save SVG"; saveSvgBtn.disabled = false; }, 2000);
        } else {
          alert("保存失败: " + res.statusText);
          saveSvgBtn.textContent = "Save SVG";
          saveSvgBtn.disabled = false;
        }
      } catch (e) {
        alert("保存出错: " + e);
        saveSvgBtn.textContent = "Save SVG";
        saveSvgBtn.disabled = false;
      }
    });

    downloadSvgBtn.addEventListener("click", () => {
      const svgText = getCurrentSvgText();
      if (!svgText) {
        alert("无法获取 SVG 内容，请确认 SVG-Edit 已加载。");
        return;
      }
      const blob = new Blob([svgText], { type: "image/svg+xml" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `final_edited_${jobId}.svg`;
      a.click();
      URL.revokeObjectURL(url);
    });

    exportPngBtn.addEventListener("click", () => {
      const svgText = getCurrentSvgText();
      if (!svgText) {
        alert("无法获取 SVG 内容，请确认 SVG-Edit 已加载。");
        return;
      }
      const parser = new DOMParser();
      const doc = parser.parseFromString(svgText, "image/svg+xml");
      const svgEl = doc.querySelector("svg");
      const scale = 4;
      const w = parseFloat(svgEl.getAttribute("width") || svgEl.viewBox.baseVal.width || 1200);
      const h = parseFloat(svgEl.getAttribute("height") || svgEl.viewBox.baseVal.height || 800);
      const canvas = document.createElement("canvas");
      canvas.width = w * scale;
      canvas.height = h * scale;
      const ctx = canvas.getContext("2d");
      ctx.scale(scale, scale);
      const img = new Image();
      const blob = new Blob([svgText], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      img.onload = () => {
        ctx.drawImage(img, 0, 0, w, h);
        URL.revokeObjectURL(url);
        canvas.toBlob((pngBlob) => {
          const a = document.createElement("a");
          a.href = URL.createObjectURL(pngBlob);
          a.download = `final_edited_${jobId}_4x.png`;
          a.click();
        }, "image/png");
      };
      img.onerror = () => { URL.revokeObjectURL(url); alert("PNG 渲染失败，SVG 可能含有不支持的外部资源。"); };
      img.src = url;
    });

    if (exportPptxBtn) {
      exportPptxBtn.addEventListener("click", async () => {
        const svgText = getCurrentSvgText();
        if (!svgText) {
          alert("无法获取 SVG 内容，请确认 SVG-Edit 已加载。");
          return;
        }
        exportPptxBtn.textContent = "Exporting…";
        exportPptxBtn.disabled = true;
        try {
          const res = await fetch(`/api/export-pptx/${jobId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ svg: svgText }),
          });
          if (!res.ok) {
            const msg = await res.text();
            alert("PPTX 导出失败: " + msg);
            return;
          }
          const blob = await res.blob();
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = `figure_${jobId}.pptx`;
          a.click();
          URL.revokeObjectURL(url);
          exportPptxBtn.textContent = "Exported ✓";
          setTimeout(() => { exportPptxBtn.textContent = "Export PPTX (Editable SVG)"; exportPptxBtn.disabled = false; }, 2000);
        } catch (e) {
          alert("PPTX 导出出错: " + e);
          exportPptxBtn.textContent = "Export PPTX (Editable SVG)";
          exportPptxBtn.disabled = false;
        }
      });
    }

    let svgEditAvailable = false;
    let svgEditPath = null;
    try {
      const configRes = await fetch("/api/config");
      if (configRes.ok) {
        const config = await configRes.json();
        svgEditAvailable = Boolean(config.svgEditAvailable);
        svgEditPath = config.svgEditPath || null;
      }
    } catch (err) {
      svgEditAvailable = false;
    }

    if (svgEditAvailable && svgEditPath) {
      iframe.src = svgEditPath;
    } else {
      fallback.classList.add("active");
      iframe.style.display = "none";
    }

    let svgReady = false;
    let pendingSvgText = null;

    iframe.addEventListener("load", () => {
      svgReady = true;
      if (pendingSvgText) {
        tryLoadSvg(pendingSvgText);
        pendingSvgText = null;
      }
    });

    const stepMap = {
      figure: { step: 1, label: "Figure generated" },
      samed: { step: 2, label: "SAM3 segmentation" },
      icon_raw: { step: 3, label: "Icons extracted" },
      icon_nobg: { step: 3, label: "Icons refined" },
      template_svg: { step: 4, label: "Template SVG ready" },
      final_svg: { step: 5, label: "Final SVG ready" },
    };

    let currentStep = 0;

    const artifacts = new Set();
    const eventSource = new EventSource(`/api/events/${jobId}`);
    let isFinished = false;

    eventSource.addEventListener("artifact", async (event) => {
      const data = JSON.parse(event.data);
      if (!artifacts.has(data.path)) {
        artifacts.add(data.path);
        addArtifactCard(artifactList, data);
      }

      if (data.kind === "template_svg" || data.kind === "final_svg") {
        await loadSvgAsset(data.url);
      }

      if (data.kind === "final_svg") {
        saveSvgBtn.style.display = "";
        downloadSvgBtn.style.display = "";
        exportPngBtn.style.display = "";
        if (exportPptxBtn) exportPptxBtn.style.display = "";
      }

      if (stepMap[data.kind] && stepMap[data.kind].step > currentStep) {
        currentStep = stepMap[data.kind].step;
        statusText.textContent = `Step ${currentStep}/5 - ${stepMap[data.kind].label}`;
      }
    });

    eventSource.addEventListener("status", (event) => {
      const data = JSON.parse(event.data);
      if (data.state === "started") {
        statusText.textContent = "Running";
      } else if (data.state === "finished") {
        isFinished = true;
        if (typeof data.code === "number" && data.code !== 0) {
          statusText.textContent = `Failed (code ${data.code})`;
        } else {
          statusText.textContent = "Done";
        }
      }
    });

    eventSource.addEventListener("log", (event) => {
      const data = JSON.parse(event.data);
      appendLogLine(logBody, data);
    });

    eventSource.onerror = () => {
      if (isFinished) {
        eventSource.close();
        return;
      }
      statusText.textContent = "Disconnected";
    };

    async function loadSvgAsset(url) {
      let svgText = "";
      try {
        const response = await fetch(url);
        svgText = await response.text();
      } catch (err) {
        return;
      }

      if (svgEditAvailable) {
        if (!svgEditPath) {
          return;
        }
        if (!svgReady) {
          pendingSvgText = svgText;
          return;
        }

        const loaded = tryLoadSvg(svgText);
        if (!loaded) {
          iframe.src = `${svgEditPath}?url=${encodeURIComponent(url)}`;
        }
      } else {
        fallbackObject.data = url;
      }
    }

    function tryLoadSvg(svgText) {
      if (!iframe.contentWindow) {
        return false;
      }

      const win = iframe.contentWindow;
      if (win.svgEditor && typeof win.svgEditor.loadFromString === "function") {
        win.svgEditor.loadFromString(svgText);
        return true;
      }
      if (win.svgCanvas && typeof win.svgCanvas.setSvgString === "function") {
        win.svgCanvas.setSvgString(svgText);
        return true;
      }
      return false;
    }
  }

  function appendLogLine(container, data) {
    const line = `[${data.stream}] ${data.line}`;
    const lines = container.textContent.split("\n").filter(Boolean);
    lines.push(line);
    if (lines.length > 200) {
      lines.splice(0, lines.length - 200);
    }
    container.textContent = lines.join("\n");
    container.scrollTop = container.scrollHeight;
  }

  function addArtifactCard(container, data) {
    const card = document.createElement("a");
    card.className = "artifact-card";
    card.href = data.url;
    card.target = "_blank";
    card.rel = "noreferrer";

    const img = document.createElement("img");
    img.src = data.url;
    img.alt = data.name;
    img.loading = "lazy";

    const meta = document.createElement("div");
    meta.className = "artifact-meta";

    const name = document.createElement("div");
    name.className = "artifact-name";
    name.textContent = data.name;

    const badge = document.createElement("div");
    badge.className = "artifact-badge";
    badge.textContent = formatKind(data.kind);

    meta.appendChild(name);
    meta.appendChild(badge);
    card.appendChild(img);
    card.appendChild(meta);
    container.prepend(card);
  }

  function formatKind(kind) {
    switch (kind) {
      case "figure":
        return "figure";
      case "samed":
        return "samed";
      case "icon_raw":
        return "icon raw";
      case "icon_nobg":
        return "icon no-bg";
      case "template_svg":
        return "template";
      case "final_svg":
        return "final";
      default:
        return "artifact";
    }
  }
})();
