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
      const samMinScoreRaw = $("samMinScore") ? parseFloat($("samMinScore").value) : null;
      const samMinScore = (!isNaN(samMinScoreRaw) && samMinScoreRaw !== null) ? samMinScoreRaw : null;
      const maxBoxAreaRatioRaw = $("maxBoxAreaRatio") ? parseFloat($("maxBoxAreaRatio").value) : null;
      const maxBoxAreaRatio = (!isNaN(maxBoxAreaRatioRaw) && maxBoxAreaRatioRaw !== null) ? maxBoxAreaRatioRaw : null;
      const mergeThresholdRaw = $("mergeThreshold") ? parseFloat($("mergeThreshold").value) : null;
      const mergeThreshold = (!isNaN(mergeThresholdRaw) && mergeThresholdRaw !== null) ? mergeThresholdRaw : null;

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
        sam_min_score: samMinScore,
        max_box_area_ratio: maxBoxAreaRatio,
        merge_threshold: mergeThreshold,
      };
      if (payload.sam_backend === "local") {
        payload.sam_api_key = null;
      }

      // New flow: if using AI image generation (no uploaded input image),
      // only run step 1 first, then let user confirm before continuing.
      const usePreviewFlow = !uploadedInputImagePath;
      if (usePreviewFlow) {
        payload.stop_after = 1;
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

        // Save config for confirm/regenerate actions on canvas page
        try {
          sessionStorage.setItem("autofigure_last_config", JSON.stringify(payload));
          sessionStorage.setItem("autofigure_last_method", methodText);
        } catch (_) {
          /* sessionStorage might be unavailable */
        }

        const params = new URLSearchParams({ job: data.job_id });
        if (usePreviewFlow) {
          params.set("mode", "preview");
        }
        window.location.href = `/canvas.html?${params.toString()}`;
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

    // ===== Preview mode: confirm AI image before continuing =====
    const previewMode = params.get("mode") === "preview";
    const confirmOverlay = $("confirmOverlay");
    const confirmImage = $("confirmImage");
    const confirmContinueBtn = $("confirmContinueBtn");
    const confirmAdjustBtn = $("confirmAdjustBtn");
    const adjustPanel = $("adjustPanel");
    const adjustMethodText = $("adjustMethodText");
    const regenerateBtn = $("regenerateBtn");
    const confirmStatus = $("confirmStatus");
    let figureUrl = null;

    function maybeShowConfirmModal() {
      if (!previewMode || !figureUrl || !isFinished) return;
      if (confirmOverlay && confirmOverlay.hasAttribute("hidden")) {
        confirmImage.src = figureUrl;
        confirmOverlay.removeAttribute("hidden");
        // Pre-fill adjust textarea with last method text
        try {
          const lastMethod = sessionStorage.getItem("autofigure_last_method");
          if (lastMethod && adjustMethodText) {
            adjustMethodText.value = lastMethod;
          }
        } catch (_) { /* ignore */ }
      }
    }

    function setConfirmStatus(text) {
      if (confirmStatus) confirmStatus.textContent = text || "";
    }

    function getLastConfig() {
      try {
        const raw = sessionStorage.getItem("autofigure_last_config");
        return raw ? JSON.parse(raw) : null;
      } catch (_) {
        return null;
      }
    }

    if (confirmContinueBtn) {
      confirmContinueBtn.addEventListener("click", async () => {
        const config = getLastConfig();
        if (!config) {
          setConfirmStatus("无法读取配置，请返回首页重新操作");
          return;
        }
        confirmContinueBtn.disabled = true;
        confirmAdjustBtn.disabled = true;
        setConfirmStatus("正在启动后续步骤...");
        try {
          const res = await fetch("/api/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ...config,
              resume_dir: jobId,
              stop_after: 5,
              input_image_path: null,
              reference_image_path: null,
            }),
          });
          if (!res.ok) throw new Error(await res.text() || "Request failed");
          const data = await res.json();
          window.location.href = `/canvas.html?job=${encodeURIComponent(data.job_id)}`;
        } catch (err) {
          setConfirmStatus(err.message || "启动失败");
          confirmContinueBtn.disabled = false;
          confirmAdjustBtn.disabled = false;
        }
      });
    }

    if (confirmAdjustBtn) {
      confirmAdjustBtn.addEventListener("click", () => {
        if (adjustPanel) adjustPanel.hidden = false;
        if (adjustMethodText) adjustMethodText.focus();
      });
    }

    if (regenerateBtn) {
      regenerateBtn.addEventListener("click", async () => {
        const config = getLastConfig();
        if (!config) {
          setConfirmStatus("无法读取配置，请返回首页重新操作");
          return;
        }
        const newMethod = adjustMethodText ? adjustMethodText.value.trim() : "";
        if (!newMethod) {
          setConfirmStatus("请输入文本内容");
          return;
        }
        regenerateBtn.disabled = true;
        confirmContinueBtn.disabled = true;
        setConfirmStatus("正在重新生成图片...");
        try {
          const res = await fetch("/api/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ...config,
              method_text: newMethod,
              stop_after: 1,
              resume_dir: null,
              input_image_path: null,
              reference_image_path: null,
            }),
          });
          if (!res.ok) throw new Error(await res.text() || "Request failed");
          const data = await res.json();
          try {
            sessionStorage.setItem("autofigure_last_method", newMethod);
            sessionStorage.setItem("autofigure_last_config", JSON.stringify({
              ...config,
              method_text: newMethod,
            }));
          } catch (_) { /* ignore */ }
          window.location.href = `/canvas.html?job=${encodeURIComponent(data.job_id)}&mode=preview`;
        } catch (err) {
          setConfirmStatus(err.message || "重新生成失败");
          regenerateBtn.disabled = false;
          confirmContinueBtn.disabled = false;
        }
      });
    }
    // ===== End preview mode setup =====

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
          setTimeout(() => { exportPptxBtn.textContent = "Export PPTX"; exportPptxBtn.disabled = false; }, 2000);
        } catch (e) {
          alert("PPTX 导出出错: " + e);
          exportPptxBtn.textContent = "Export PPTX";
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
      // 禁用 SVG-Edit 的本地存储（storagePrompt=false 阻止弹窗请求存储；
      // noStorageOnLoad=true 禁止其从 localStorage 恢复上一次的 SVG 内容，
      // 避免新任务画布先显示上一次结果、直到新结果推送才被覆盖）。
      iframe.src = svgEditPath + (svgEditPath.includes("?") ? "&" : "?") + "storagePrompt=false&noStorageOnLoad=true";
    } else {
      fallback.classList.add("active");
      iframe.style.display = "none";
    }

    // 清理 SVG-Edit 曾写入的遗留内容（用户以前同意过存储时），
    // 保证画布从空白开始。cookie 需按 iframe 路径一并清除。
    try {
      if (window.localStorage) {
        localStorage.removeItem("svgedit-default");
        localStorage.removeItem("title-svgedit-default");
      }
      ["/", "/vendor/svg-edit/", "/vendor/svg-edit/editor/", "/vendor/svg-edit/editor/index.html"].forEach((p) => {
        document.cookie = "svgeditstore=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=" + p;
      });
    } catch (_) { /* localStorage 可能被禁用，忽略 */ }

    let svgReady = false;
    let pendingSvgText = null;
    /** 2 = final.svg, 1 = template — 避免 template 覆盖已排队的 final */
    let pendingSvgPriority = 0;

    function setPendingSvg(svgText, priority) {
      if (priority >= pendingSvgPriority) {
        pendingSvgText = svgText;
        pendingSvgPriority = priority;
      }
    }

    /**
     * SVG-Edit 的线框模式会通过 CSS 强制 fill:none，覆盖图形上的填色。
     * 用户偏好或误触可能持久化开启线框；在画布页强制恢复到正常着色模式。
     */
    function ensureSvgEditWireframeOff(win) {
      try {
        const ed = win.svgEditor;
        if (!ed?.workarea || !ed.topPanel || typeof ed.topPanel.clickWireframe !== "function") {
          return;
        }
        if (ed.workarea.classList.contains("wireframe")) {
          ed.topPanel.clickWireframe();
        }
      } catch (_) {
        /* ignore cross-origin or partial init */
      }
    }

    function scheduleSvgEditWireframeOff(win) {
      if (!win) {
        return;
      }
      [0, 80, 250, 500].forEach((ms) => {
        setTimeout(() => ensureSvgEditWireframeOff(win), ms);
      });
    }

    /**
     * 将视图缩放至完整显示 SVG 画布内容（默认视图改进）。
     * SVG-Edit 默认以 100% 加载，大尺寸 SVG 会被裁切；这里计算 workarea
     * 视口与画布的比例，自动缩放到 ~95% 适配并居中。
     */
    function fitSvgToView(win) {
      if (!win) return;
      try {
        const ed = win.svgEditor;
        const canvas = ed?.svgCanvas || win.svgCanvas;
        if (!canvas || !ed?.workarea) return;
        const res = canvas.getResolution();
        if (!res || !res.w || !res.h) return;
        const vw = ed.workarea.clientWidth;
        const vh = ed.workarea.clientHeight;
        if (!vw || !vh) return;
        // 留 5% 边距，保证完整可见
        const fitZoom = Math.min(vw / res.w, vh / res.h) * 0.95;
        // 限制在合理范围 5%–800%
        const clamped = Math.max(0.05, Math.min(8, fitZoom));
        if (canvas.setCurrentZoom) {
          canvas.setCurrentZoom(clamped);
        } else if (canvas.setZoom) {
          canvas.setZoom(clamped);
          const zoomInput = win.document.getElementById("zoom");
          if (zoomInput) zoomInput.value = (clamped * 100).toFixed(1);
        }
        // 居中画布
        if (typeof ed.updateCanvas === "function") {
          ed.updateCanvas(true);
        }
      } catch (_) { /* iframe 可能尚未就绪，忽略 */ }
    }

    /**
     * 注入 Ctrl+滚轮缩放支持到 SVG-Edit iframe。
     * SVG-Edit 原生仅支持 Shift+滚轮缩放；现代用户更习惯 Ctrl+滚轮。
     * 使用 capture 阶段拦截，避免触发浏览器页面缩放和 SVG-Edit 的 shift 处理。
     */
    let ctrlWheelAttached = false;
    function attachCtrlWheelZoom(win) {
      if (!win || ctrlWheelAttached) return;
      try {
        const doc = win.document;
        if (!doc) return;
        doc.addEventListener("wheel", (e) => {
          if (!e.ctrlKey) return;
          e.preventDefault();
          e.stopPropagation();
          const ed = win.svgEditor;
          if (!ed || typeof ed.zoomImage !== "function") return;
          // 向上滚（deltaY < 0）放大，向下滚缩小；每档 1.1 倍
          const factor = e.deltaY < 0 ? 1.1 : 0.9;
          ed.zoomImage(factor);
        }, { capture: true, passive: false });
        ctrlWheelAttached = true;
      } catch (_) { /* 跨域或部分初始化，忽略 */ }
    }

    /**
     * 修复 Del/Backspace 删除元素。
     * SVG-Edit 内置的删除快捷键仅在 keydown 目标为 BODY 时生效；
     * 点击画布上的图形后焦点落在 SVG 元素上，导致 Del 无效。
     * 这里在 iframe 文档上补一个监听，直接调用 svgCanvas.deleteSelectedElements()。
     */
    let deleteKeyAttached = false;
    function attachDeleteKey(win) {
      if (!win || deleteKeyAttached) return;
      try {
        const doc = win.document;
        if (!doc) return;
        doc.addEventListener("keydown", (e) => {
          if (e.key !== "Delete" && e.key !== "Backspace") return;
          // 不干扰文本输入框/多行文本的退格
          const node = e.target;
          const tag = (node && node.tagName || "").toLowerCase();
          if (tag === "input" || tag === "textarea" || (node && node.isContentEditable)) return;
          const ed = win.svgEditor;
          if (!ed || !ed.svgCanvas) return;
          if (ed.selectedElement || ed.multiselected) {
            e.preventDefault();
            e.stopPropagation();
            ed.svgCanvas.deleteSelectedElements();
          }
        });
        deleteKeyAttached = true;
      } catch (_) { /* 跨域或部分初始化，忽略 */ }
    }

    iframe.addEventListener("load", () => {
      svgReady = true;
      const win = iframe.contentWindow;
      scheduleSvgEditWireframeOff(win);
      attachCtrlWheelZoom(win);
      attachDeleteKey(win);
      const queued = pendingSvgText;
      pendingSvgText = null;
      pendingSvgPriority = 0;
      if (queued) {
        void tryLoadSvg(queued).then((ok) => {
          if (ok) setTimeout(() => fitSvgToView(win), 60);
        });
      } else {
        // iframe 直接通过 ?url= 加载时，SVG-Edit 自己会装入；给一点时间后适配
        setTimeout(() => fitSvgToView(win), 200);
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

      if (data.kind === "figure" && !figureUrl) {
        figureUrl = data.url;
        maybeShowConfirmModal();
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
        maybeShowConfirmModal();
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
          const pri = url.includes("final.svg") ? 2 : 1;
          setPendingSvg(svgText, pri);
          return;
        }

        const loaded = await tryLoadSvg(svgText);
        if (loaded) {
          // SSE 推送新 SVG 后也需适配视图
          setTimeout(() => fitSvgToView(iframe.contentWindow), 60);
        } else {
          iframe.src = `${svgEditPath}?storagePrompt=false&noStorageOnLoad=true&url=${encodeURIComponent(url)}`;
        }
      } else {
        fallbackObject.data = url;
      }
    }

    async function tryLoadSvg(svgText) {
      if (!iframe.contentWindow) {
        return false;
      }

      const win = iframe.contentWindow;
      if (win.svgEditor && typeof win.svgEditor.loadFromString === "function") {
        await win.svgEditor.loadFromString(svgText);
        scheduleSvgEditWireframeOff(win);
        return true;
      }
      if (win.svgCanvas && typeof win.svgCanvas.setSvgString === "function") {
        const maybePromise = win.svgCanvas.setSvgString(svgText);
        if (maybePromise && typeof maybePromise.then === "function") {
          await maybePromise;
        }
        scheduleSvgEditWireframeOff(win);
        return true;
      }
      return false;
    }

    /**
     * 打开画布时若 job 已在磁盘上生成 final.svg（刷新、重启 SSE 无回放），主动拉取并装入编辑器。
     */
    async function bootstrapFinalFromDisk() {
      if (!jobId || !svgEditAvailable) {
        return;
      }
      const url = `/api/artifacts/${encodeURIComponent(jobId)}/final.svg`;
      try {
        const response = await fetch(url);
        if (!response.ok) {
          return;
        }
        const svgText = await response.text();
        saveSvgBtn.style.display = "";
        downloadSvgBtn.style.display = "";
        exportPngBtn.style.display = "";
        if (exportPptxBtn) exportPptxBtn.style.display = "";
        if (!svgReady) {
          setPendingSvg(svgText, 2);
          return;
        }
        const ok = await tryLoadSvg(svgText);
        if (ok) setTimeout(() => fitSvgToView(iframe.contentWindow), 60);
      } catch (_) {
        /* 磁盘上无 final 或网络错误 — 忽略，由 SSE 后续推送 */
      }
    }

    void bootstrapFinalFromDisk();

    /**
     * Preview mode: check if figure.png already exists on disk (page refresh / SSE miss).
     */
    async function bootstrapFigureFromDisk() {
      if (!previewMode || !jobId) return;
      const url = `/api/artifacts/${encodeURIComponent(jobId)}/figure.png`;
      try {
        const response = await fetch(url, { method: "HEAD" });
        if (response.ok && !figureUrl) {
          figureUrl = url;
          // Assume job is finished if figure.png exists in preview mode
          isFinished = true;
          maybeShowConfirmModal();
        }
      } catch (_) { /* ignore */ }
    }

    void bootstrapFigureFromDisk();
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
