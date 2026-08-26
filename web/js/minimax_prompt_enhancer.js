/** LLM prompt enhancer panel for MiniMax H3 Director (Ollama / Zhipu). */

import { api } from "../../scripts/api.js";
import { resolveTaskKey, taskUsesReferenceImages, taskUsesReferenceVideo } from "./minimax_gen_timeline.js";
import { stripFl2vPromptBody } from "./minimax_fl2v.js";

export const PE_PANEL_COLLAPSED_H = 34;
export const PE_PANEL_EXPANDED_H = 348;

const DEFAULT_LLM_URL = "http://127.0.0.1:11434/v1";
const DEFAULT_LLM_MODEL = "qwen3.5";
const DEFAULT_ZHIPU_URL = "https://open.bigmodel.cn/api/paas/v4";
const DEFAULT_ZHIPU_MODEL = "glm-4.6v-flash";
const DEFAULT_OPENAI_COMPAT_URL = "http://127.0.0.1:8080/v1";
const DEFAULT_API_FORMAT = "Ollama";
const API_OLLAMA = "Ollama";
const API_ZHIPU = "智谱 GLM";
const API_OPENAI_COMPAT = "OpenAI Compatible";
const OPENAI_COMPAT_STANDARD = "标准";
const OPENAI_COMPAT_LLAMA_SWAP = "llama-swap";
const DEFAULT_OUTPUT_LANGUAGE = "中文";
const OUTPUT_LANGUAGE_ZH = "中文";
const CHARACTER_DETAIL_NORMAL = "一般";
const CHARACTER_DETAIL_DETAILED = "详尽";
const LEGACY_OPENAI_FORMAT = "OpenAI / vLLM";

const STATUS_COLORS = {
    info: "#9aa3b5",
    loading: "#fbbf24",
    success: "#4ade80",
    error: "#f87171",
};

function coerceLlmUrl(value, defaultUrl = DEFAULT_LLM_URL) {
    const s = String(value ?? "").trim();
    if (/^https?:\/\//i.test(s)) return s.replace(/\/+$/, "");
    return defaultUrl;
}

function coerceLlmModel(value) {
    const s = String(value ?? "").trim();
    if (!s || s === "true" || s === "false") return DEFAULT_LLM_MODEL;
    return s;
}

function normalizeApiFormat(fmt) {
    if (fmt === LEGACY_OPENAI_FORMAT) return API_OPENAI_COMPAT;
    if (fmt === API_ZHIPU || fmt === API_OLLAMA || fmt === API_OPENAI_COMPAT) return fmt;
    return DEFAULT_API_FORMAT;
}

function inferApiFormat(url, explicit) {
    const fmt = normalizeApiFormat(explicit);
    if (fmt === API_ZHIPU || fmt === API_OLLAMA || fmt === API_OPENAI_COMPAT) return fmt;
    const u = coerceLlmUrl(url);
    if (/bigmodel\.cn/i.test(u)) return API_ZHIPU;
    return DEFAULT_API_FORMAT;
}

function defaultsForApiFormat(fmt) {
    if (fmt === API_ZHIPU) return { url: DEFAULT_ZHIPU_URL, model: DEFAULT_ZHIPU_MODEL };
    if (fmt === API_OPENAI_COMPAT) return { url: DEFAULT_OPENAI_COMPAT_URL, model: DEFAULT_LLM_MODEL };
    return { url: "http://127.0.0.1:11434", model: DEFAULT_LLM_MODEL };
}

function normalizeOpenAiCompatMode(mode) {
    return String(mode || "").trim().toLowerCase() === OPENAI_COMPAT_LLAMA_SWAP
        ? OPENAI_COMPAT_LLAMA_SWAP
        : OPENAI_COMPAT_STANDARD;
}

function ensurePeStyles() {
    if (document.getElementById("minimax-pe-styles")) return;
    const style = document.createElement("style");
    style.id = "minimax-pe-styles";
    style.textContent = `
@keyframes minimax-pe-pulse { 0%,100%{opacity:1} 50%{opacity:.65} }
.minimax-pe-loading { animation: minimax-pe-pulse 1.2s ease-in-out infinite !important; }
.minimax-pe-label { font-size: 11px; color: #b8c0d0; flex-shrink: 0; white-space: nowrap; }
.minimax-pe-input, .minimax-pe-select {
    font-size: 11px; line-height: 1.35; min-height: 28px; box-sizing: border-box;
    background: #12151b; color: #e8ecf4; border: 1px solid #2a3140; border-radius: 4px;
}
.minimax-pe-input { padding: 5px 8px; }
.minimax-pe-select { padding: 4px 8px; cursor: pointer; }
.minimax-pe-btn-sm {
    font-size: 11px; line-height: 1.35; min-height: 28px; box-sizing: border-box;
    background: #252a34; color: #e8ecf4; border: 1px solid #2a3140; border-radius: 4px;
    padding: 4px 10px; cursor: pointer; flex-shrink: 0; white-space: nowrap;
}
.minimax-pe-api-row { display: flex; gap: 6px; align-items: center; flex-wrap: nowrap; }
.minimax-pe-api-row .minimax-pe-select { flex: 0 1 38%; min-width: 132px; max-width: 220px; }
.minimax-pe-api-row .minimax-pe-input { flex: 1 1 120px; min-width: 0; }
.minimax-pe-options-row { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
.minimax-pe-check-item { display: flex; gap: 4px; align-items: center; }
.minimax-pe-check-item span { font-size: 11px; color: #b8c0d0; }
.minimax-pe-cancel-btn {
    font-size: 10px; min-height: 24px; box-sizing: border-box;
    background: #7f1d1d !important; color: #fecaca !important;
    border: 1px solid #b91c1c !important; border-radius: 4px;
    padding: 3px 9px; cursor: pointer; flex-shrink: 0; white-space: nowrap;
}
.minimax-pe-progress { font-size: 10px; color: #94a3b8; white-space: nowrap; min-width: 42px; text-align: right; }
`;
    document.head.appendChild(style);
}

function el(style, text, tag = "div") {
    const node = document.createElement(tag);
    if (style) Object.assign(node.style, style);
    if (text != null) node.textContent = text;
    return node;
}

function swallowKeys(input) {
    input.addEventListener("keydown", (e) => e.stopPropagation());
    input.addEventListener("keyup", (e) => e.stopPropagation());
}

function isAbortError(e) { return e?.name === "AbortError"; }

async function readApiError(resp, fallback) {
    let data = {};
    try { data = await resp.json(); } catch {}
    return { data, error: data.error || `${fallback} (HTTP ${resp.status})` };
}

function networkHint(e) {
    if (isAbortError(e)) return "已取消操作";
    const message = String(e?.message || e || "");
    if (/Failed to fetch|NetworkError|Load failed/i.test(message)) {
        return `${message}（请确认 LLM 服务已启动、地址/端口正确，且浏览器可访问）`;
    }
    return message;
}

async function fetchImageB64(imageFile) {
    const resp = await api.fetchApi("/minimax/director/image_b64", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ imageFile }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || resp.statusText);
    return data.image;
}

function resolveOutputLanguage(pe) {
    const widgetLang = pe.widget?.("llm_output_language")?.value;
    if (widgetLang) {
        if (pe.langSelect) pe.langSelect.value = widgetLang;
        return widgetLang;
    }
    return pe.langSelect?.value || DEFAULT_OUTPUT_LANGUAGE;
}

function coerceFeatureEnhanceValue(value) {
    if (value === true || value === 1) return true;
    if (value === false || value === 0 || value == null || value === "") return false;
    const v = String(value).trim().toLowerCase();
    if (v === "true" || v === "yes" || v === "on") return true;
    // 旧下拉迁移到 BOOLEAN 时可能残留「一般」「详尽」字符串，仅「详尽」视为开启
    if (v.includes("详尽") || v === "detailed" || v === "verbose" || v === "full") return true;
    return false;
}

function resolveCharacterFeatureEnhance(pe, { preferWidget = false } = {}) {
    const enhanceWidget = pe.widget?.("llm_character_feature_enhance");
    const fromWidgetEnhance = enhanceWidget?.value;
    const fromCheck = pe.detailCheck?.checked;
    if (preferWidget && enhanceWidget) {
        return coerceFeatureEnhanceValue(fromWidgetEnhance);
    }
    if (fromCheck != null) return !!fromCheck;
    if (enhanceWidget) return coerceFeatureEnhanceValue(fromWidgetEnhance);
    return false;
}

function formatEnhanceSuccessStatus(taskKey, result) {
    let msg = `扩写成功 (${taskKey})，${result.text?.length ?? 0} 字符`;
    if (result.detailedMode) {
        const han = result.hanCount ?? 0;
        const target = result.detailTargetHan ?? 300;
        msg += `，${han} 汉字（角色特征增强目标≥${target}）`;
        if (han < target) {
            msg += "；仍偏短请换更强 Vision 模型或确认参考图已上传";
        }
    }
    return msg;
}

function resolveCharacterDetailLevel(pe, opts) {
    return resolveCharacterFeatureEnhance(pe, opts) ? CHARACTER_DETAIL_DETAILED : "一般";
}

export function mountPromptEnhancerPanel(editor, parentEl) {
    ensurePeStyles();
    const pe = {
        editor, open: false, _currentDefaultTemplate: "", _busy: false,
        _abortController: null,
    };

    pe.setStatus = (text, kind = "info", action = null) => {
        pe.statusEl.textContent = text || "";
        pe.statusEl.style.color = STATUS_COLORS[kind] || STATUS_COLORS.info;
        pe.statusEl.style.fontWeight = kind === "error" ? "600" : "400";
        pe.statusEl.style.whiteSpace = "pre-wrap";
        pe.statusEl.style.lineHeight = "1.35";
        pe.actionEl.textContent = action?.label || "";
        pe.actionEl.onclick = action?.run || null;
        pe.actionEl.style.display = action?.label ? "inline-block" : "none";
    };

    pe.cancelEnhance = (message = "已取消扩写") => {
        const controller = pe._abortController;
        if (!controller) return false;
        controller.abort();
        pe._cancelled = true;
        pe.setStatus(message, "info");
        return true;
    };

    pe.setProgressText = (text) => {
        pe.progressEl.textContent = text || "";
    };

    pe.updateProgress = (done, total, label = "") => {
        const prefix = label ? `${label} · ` : "";
        pe.setProgressText(total > 0 ? `${prefix}${done}/${total}` : "");
    };

    pe.setEnhanceLoading = (loading, activeBtn = null, label = "扩写中…") => {
        const wasBusy = !!pe._busy;
        pe._busy = loading;
        pe.enhanceCurrentBtn.disabled = loading;
        pe.enhanceAllBtn.disabled = loading;
        pe.refreshBtn.disabled = loading;
        pe.unloadBtn.disabled = loading;
        if (loading && activeBtn) {
            if (!wasBusy) {
                pe._cancelled = false;
                pe._activeButton = activeBtn;
                pe._activeButtonLabel = activeBtn.textContent;
                pe._abortController = new AbortController();
                pe.cancelBtn.style.display = "inline-block";
                pe.progressEl.style.display = "inline-block";
            }
            if (activeBtn !== pe.cancelBtn) {
                activeBtn.textContent = label;
                activeBtn.setAttribute("aria-busy", "true");
            }
            activeBtn.classList.add("minimax-pe-loading");
            if (activeBtn !== pe.cancelBtn) activeBtn.style.cursor = "wait";
        } else if (!loading) {
            const btn = pe._activeButton;
            const savedLabel = pe._activeButtonLabel || (btn === pe.enhanceAllBtn ? "扩写全部提示词" : "扩写当前提示词");
            if (btn && btn !== pe.cancelBtn) {
                btn.textContent = savedLabel;
                btn.removeAttribute("aria-busy");
                btn.classList.remove("minimax-pe-loading");
                btn.style.cursor = "pointer";
            }
            pe._activeButton = null;
            pe._activeButtonLabel = "";
            pe._abortController = null;
            pe.cancelBtn.style.display = "none";
            pe.setProgressText("");
            pe.progressEl.style.display = "none";
        }
    };

    pe.isCancelled = () => !!pe._cancelled;

    pe.throwIfCancelled = () => {
        if (pe.isCancelled()) throw new DOMException("Aborted", "AbortError");
    };

    const header = el({
        display: "flex", justifyContent: "space-between", alignItems: "center",
        background: "#1a1d24", border: "1px solid #2a3140", borderRadius: "4px",
        padding: "6px 8px", cursor: "pointer", userSelect: "none", marginTop: "6px",
    });
    header.appendChild(el({ fontWeight: "600", fontSize: "10px", color: "#9aa3b5", textTransform: "uppercase" }, "LLM 提示词增强 Prompt Enhancer"));
    pe.arrow = el({ fontSize: "10px", color: "#9aa3b5" }, "\u25B6");
    header.appendChild(pe.arrow);

    pe.body = el({
        background: "#1a1d24", border: "1px solid #2a3140", borderTop: "none",
        borderRadius: "0 0 4px 4px", padding: "8px", display: "none",
        flexDirection: "column", gap: "6px", marginTop: "-5px",
    });
    pe.body.appendChild(el({ fontSize: "9px", color: "#7d8698", lineHeight: "1.4" },
        "按 MiniMax H3 官方 task 模板扩写短提示词。「当前」仅扩写选中片段/全局；「全部」在分段模式下依次扩写各片段。"));

    const fmtRow = el({});
    fmtRow.className = "minimax-pe-api-row";
    fmtRow.appendChild(el({}, "API:", "span")).className = "minimax-pe-label";
    pe.apiSelect = document.createElement("select");
    pe.apiSelect.className = "minimax-pe-select";
    for (const [val, label] of [
        [API_OLLAMA, "Ollama (/api/chat)"],
        [API_ZHIPU, "智谱 GLM (/paas/v4/chat)"],
        [API_OPENAI_COMPAT, "OpenAI Compatible (/v1/chat/completions)"],
    ]) {
        const o = document.createElement("option");
        o.value = val; o.textContent = label;
        pe.apiSelect.appendChild(o);
    }
    pe.urlInput = document.createElement("input");
    pe.urlInput.type = "text";
    pe.urlInput.placeholder = DEFAULT_LLM_URL;
    pe.urlInput.className = "minimax-pe-input";
    pe.urlInput.oninput = () => {
        pe.apiSelect.value = inferApiFormat(pe.urlInput.value, pe.apiSelect.value);
        pe.updateApiFormatUI();
        pe.syncToWidgets();
    };
    swallowKeys(pe.urlInput);
    pe.apiSelect.onchange = () => {
        const d = defaultsForApiFormat(pe.apiSelect.value);
        pe.urlInput.value = d.url;
        if (!pe.modelInput.value.trim() || pe._lastApiFormat !== pe.apiSelect.value) {
            pe.modelInput.value = d.model;
        }
        pe._lastApiFormat = pe.apiSelect.value;
        pe.updateApiFormatUI();
        pe.syncToWidgets();
        pe.fetchModels();
    };
    fmtRow.appendChild(pe.apiSelect);
    fmtRow.appendChild(pe.urlInput);
    pe.refreshBtn = el({}, "刷新模型", "button");
    pe.refreshBtn.className = "minimax-pe-btn-sm";
    pe.refreshBtn.onclick = () => pe.fetchModels();
    fmtRow.appendChild(pe.refreshBtn);
    pe.visionBadge = el({ fontSize: "9px", color: "#4ade80", flexShrink: "0", display: "none" });
    fmtRow.appendChild(pe.visionBadge);
    pe.body.appendChild(fmtRow);

    const compatRow = el({ display: "none", gap: "6px", alignItems: "center" });
    compatRow.appendChild(el({}, "OpenAI 特性:", "span")).className = "minimax-pe-label";
    pe.compatSelect = document.createElement("select");
    pe.compatSelect.className = "minimax-pe-select";
    Object.assign(pe.compatSelect.style, { flex: "1" });
    for (const [val, label] of [
        [OPENAI_COMPAT_STANDARD, "标准"],
        [OPENAI_COMPAT_LLAMA_SWAP, "llama-swap"],
    ]) {
        const o = document.createElement("option");
        o.value = val; o.textContent = label;
        pe.compatSelect.appendChild(o);
    }
    pe.compatSelect.title = "仅 OpenAI Compatible 生效。选择 llama-swap 后启用 /api/models/unload/{model_id} 卸载接口。";
    pe.compatSelect.onchange = () => {
        pe.updateApiFormatUI();
        pe.syncToWidgets();
    };
    compatRow.appendChild(pe.compatSelect);
    pe.compatRow = compatRow;
    pe.body.appendChild(compatRow);

    const keyRow = el({ display: "flex", gap: "6px", alignItems: "center" });
    keyRow.dataset.r = "pe-key-row";
    keyRow.appendChild(el({}, "API Key:", "span")).className = "minimax-pe-label";
    pe.apiKeyInput = document.createElement("input");
    pe.apiKeyInput.type = "password";
    pe.apiKeyInput.placeholder = "智谱 API Key";
    pe.apiKeyInput.autocomplete = "off";
    pe.apiKeyInput.className = "minimax-pe-input";
    Object.assign(pe.apiKeyInput.style, { flex: "1" });
    pe.apiKeyInput.oninput = () => pe.syncToWidgets();
    swallowKeys(pe.apiKeyInput);
    keyRow.appendChild(pe.apiKeyInput);
    pe.keyRow = keyRow;
    pe.body.appendChild(keyRow);

    const modelRow = el({ display: "flex", gap: "6px", alignItems: "center" });
    modelRow.appendChild(el({}, "模型:", "span")).className = "minimax-pe-label";
    pe.modelInput = document.createElement("input");
    pe.modelInput.type = "text";
    pe.modelInput.placeholder = DEFAULT_LLM_MODEL;
    pe.modelInput.className = "minimax-pe-input";
    Object.assign(pe.modelInput.style, { flex: "1" });
    pe.modelInput.oninput = () => pe.syncToWidgets();
    swallowKeys(pe.modelInput);
    modelRow.appendChild(pe.modelInput);
    pe.modelList = document.createElement("datalist");
    pe.modelList.id = `minimax-pe-models-${editor.node.id}`;
    pe.modelInput.setAttribute("list", pe.modelList.id);
    modelRow.appendChild(pe.modelList);
    pe.body.appendChild(modelRow);

    const langRow = el({ display: "flex", gap: "6px", alignItems: "center" });
    langRow.appendChild(el({}, "扩写语言:", "span")).className = "minimax-pe-label";
    pe.langSelect = document.createElement("select");
    pe.langSelect.className = "minimax-pe-select";
    Object.assign(pe.langSelect.style, { flex: "1" });
    for (const [val, label] of [
        [OUTPUT_LANGUAGE_ZH, "中文（简体中文）"],
        ["English", "English（官方推荐）"],
    ]) {
        const o = document.createElement("option");
        o.value = val; o.textContent = label;
        pe.langSelect.appendChild(o);
    }
    pe.langSelect.title =
        "LLM 扩写输出语言。MiniMax H3 官方示例与 T5 系统提示词为英文；"
        + "选中文时扩写结果为简体中文。";
    pe.langSelect.onchange = () => {
        pe._lastOutputLanguage = pe.langSelect.value || DEFAULT_OUTPUT_LANGUAGE;
        pe.syncToWidgets();
        pe.fetchTemplate(true);
    };
    langRow.appendChild(pe.langSelect);
    pe.body.appendChild(langRow);

    const AUTO_ENHANCE_TIP =
        "Queue 时在服务端自动用 LLM 扩写每段正向提示词（MiniMax H3 官方 task 模板；"
        + "可附带源视频帧与参考图）。扩写失败则使用原文，不中断生成。"
        + "多段任务会每段各调用一次 LLM，耗时会增加。";

    const optionsRow = el({});
    optionsRow.className = "minimax-pe-options-row";
    const detailItem = el({});
    detailItem.className = "minimax-pe-check-item";
    pe.detailCheck = document.createElement("input");
    pe.detailCheck.type = "checkbox";
    pe.detailCheck.checked = false;
    pe.detailCheck.title =
        "rv2v/r2v/r2i 等含参考图任务：未勾选时按 MiniMax H3 官方模板扩写；"
        + "勾选后启用角色特征增强（≥300汉字详尽外观描述）。";
    pe.detailCheck.onchange = () => pe.syncToWidgets();
    detailItem.appendChild(pe.detailCheck);
    const detailLabel = el({ cursor: "help" }, "角色特征增强", "span");
    detailLabel.title = pe.detailCheck.title;
    detailItem.appendChild(detailLabel);
    optionsRow.appendChild(detailItem);

    const autoItem = el({ cursor: "help" });
    autoItem.className = "minimax-pe-check-item";
    autoItem.title = AUTO_ENHANCE_TIP;
    pe.autoCheck = document.createElement("input");
    pe.autoCheck.type = "checkbox";
    pe.autoCheck.checked = false;
    pe.autoCheck.title = AUTO_ENHANCE_TIP;
    pe.autoCheck.onchange = () => pe.syncToWidgets();
    autoItem.appendChild(pe.autoCheck);
    const autoLabel = el({ cursor: "help" }, "自动扩写", "span");
    autoLabel.title = AUTO_ENHANCE_TIP;
    autoItem.appendChild(autoLabel);
    optionsRow.appendChild(autoItem);

    pe.unloadWrap = el({});
    pe.unloadWrap.className = "minimax-pe-check-item";
    pe.unloadCheck = document.createElement("input");
    pe.unloadCheck.type = "checkbox";
    pe.unloadCheck.onchange = () => pe.syncToWidgets();
    pe.unloadWrap.appendChild(pe.unloadCheck);
    pe.unloadCheckLabel = el({}, "用后卸载 Ollama", "span");
    pe.unloadWrap.appendChild(pe.unloadCheckLabel);
    optionsRow.appendChild(pe.unloadWrap);
    pe.body.appendChild(optionsRow);

    const btnRow = el({ display: "flex", gap: "6px", flexDirection: "column" });
    const enhanceRow = el({ display: "flex", gap: "6px" });
    pe.enhanceCurrentBtn = el({
        flex: "1", background: "#3b82f6", color: "#fff", border: "none", borderRadius: "4px",
        padding: "6px", fontWeight: "600", fontSize: "10px", cursor: "pointer",
    }, "扩写当前提示词", "button");
    pe.enhanceCurrentBtn.onclick = () => pe.enhancePrompt("current");
    enhanceRow.appendChild(pe.enhanceCurrentBtn);
    pe.enhanceAllBtn = el({
        flex: "1", background: "#6366f1", color: "#fff", border: "none", borderRadius: "4px",
        padding: "6px", fontWeight: "600", fontSize: "10px", cursor: "pointer",
    }, "扩写全部提示词", "button");
    pe.enhanceAllBtn.onclick = () => pe.enhancePrompt("all");
    enhanceRow.appendChild(pe.enhanceAllBtn);
    btnRow.appendChild(enhanceRow);
    const utilRow = el({ display: "flex", gap: "6px" });
    pe.unloadBtn = el({ background: "#252a34", color: "#e8ecf4", border: "1px solid #2a3140", borderRadius: "4px", padding: "6px 10px", fontSize: "10px", cursor: "pointer" }, "卸载 Ollama", "button");
    pe.unloadBtn.onclick = () => pe.unloadOllama();
    utilRow.appendChild(pe.unloadBtn);
    pe.unloadBtnRow = utilRow;
    btnRow.appendChild(utilRow);

    const progressRow = el({ display: "flex", gap: "6px", alignItems: "center", justifyContent: "space-between" });
    pe.progressEl = el({ fontSize: "10px", color: "#94a3b8", whiteSpace: "nowrap" }, "", "span");
    pe.progressEl.className = "minimax-pe-progress";
    pe.progressEl.style.display = "none";
    pe.cancelBtn = el({
        background: "#7f1d1d", color: "#fecaca", border: "1px solid #b91c1c",
        borderRadius: "4px", padding: "3px 9px", fontWeight: "600", fontSize: "10px",
    }, "取消", "button");
    pe.cancelBtn.className = "minimax-pe-cancel-btn";
    pe.cancelBtn.style.display = "none";
    pe.cancelBtn.onclick = () => pe.cancelEnhance();
    progressRow.appendChild(pe.progressEl);
    progressRow.appendChild(pe.cancelBtn);
    btnRow.appendChild(progressRow);
    pe.body.appendChild(btnRow);

    const statusRow = el({ display: "flex", gap: "8px", alignItems: "flex-start" });
    pe.statusEl = el({ fontSize: "10px", color: STATUS_COLORS.info, minHeight: "16px", padding: "2px 0", flex: "1" });
    pe.actionEl = el({ fontSize: "10px", color: "#93c5fd", cursor: "pointer", textDecoration: "underline", whiteSpace: "nowrap" }, "", "button");
    Object.assign(pe.actionEl.style, { display: "none", background: "transparent", border: "0", padding: "2px 0" });
    statusRow.appendChild(pe.statusEl);
    statusRow.appendChild(pe.actionEl);
    pe.body.appendChild(statusRow);

    pe.templateArea = document.createElement("textarea");
    pe.templateArea.rows = 4;
    Object.assign(pe.templateArea.style, { width: "100%", fontSize: "9px", display: "none", background: "#12151b", color: "#d6dbe6", border: "1px solid #2a3140", borderRadius: "3px" });
    pe.templateArea.oninput = () => pe.syncToWidgets();
    swallowKeys(pe.templateArea);
    pe.body.appendChild(pe.templateArea);

    header.onclick = () => {
        pe.open = !pe.open;
        pe.body.style.display = pe.open ? "flex" : "none";
        pe.arrow.style.transform = pe.open ? "rotate(90deg)" : "";
        editor.updateDomWidgetHeight?.();
        if (pe.open && !pe.modelList.options.length) pe.fetchModels(true);
    };

    parentEl.appendChild(header);
    parentEl.appendChild(pe.body);

    pe.widget = (name) => editor.widget(name);

    pe.supportsUnload = () => {
        const fmt = pe.apiSelect.value;
        return fmt === API_OLLAMA
            || (fmt === API_OPENAI_COMPAT && normalizeOpenAiCompatMode(pe.compatSelect?.value) === OPENAI_COMPAT_LLAMA_SWAP);
    };

    pe.updateApiFormatUI = () => {
        const fmt = pe.apiSelect.value;
        const isOpenAi = fmt === API_OPENAI_COMPAT;
        const showKey = fmt === API_ZHIPU || isOpenAi;
        const supportsUnload = pe.supportsUnload();
        if (pe.compatRow) pe.compatRow.style.display = isOpenAi ? "flex" : "none";
        if (pe.keyRow) pe.keyRow.style.display = showKey ? "flex" : "none";
        if (pe.apiKeyInput) {
            pe.apiKeyInput.placeholder = fmt === API_ZHIPU ? "智谱 API Key" : "OpenAI / llama-swap API Key（可选）";
        }
        if (pe.unloadWrap) pe.unloadWrap.style.display = supportsUnload ? "flex" : "none";
        if (pe.unloadBtnRow) pe.unloadBtnRow.style.display = supportsUnload ? "flex" : "none";
        const unloadText = fmt === API_OLLAMA ? "用后卸载 Ollama" : "用后卸载模型 (llama-swap)";
        if (pe.unloadCheckLabel) pe.unloadCheckLabel.textContent = unloadText;
        if (pe.unloadBtn) pe.unloadBtn.textContent = fmt === API_OLLAMA ? "卸载 Ollama" : "卸载模型 (llama-swap)";
        pe.urlInput.placeholder = defaultsForApiFormat(fmt).url;
        pe.modelInput.placeholder = defaultsForApiFormat(fmt).model;
    };

    pe.syncFromWidgets = () => {
        const w = (n) => pe.widget(n);
        const explicitFmt = w("llm_api_format")?.value || DEFAULT_API_FORMAT;
        const fmt = inferApiFormat(w("llm_url")?.value, explicitFmt);
        const url = coerceLlmUrl(w("llm_url")?.value, defaultsForApiFormat(fmt).url);
        pe.urlInput.value = url;
        pe.apiSelect.value = fmt;
        pe._lastApiFormat = pe.apiSelect.value;
        if (pe.compatSelect) {
            pe.compatSelect.value = normalizeOpenAiCompatMode(w("llm_openai_compat_mode")?.value);
        }
        pe.modelInput.value = coerceLlmModel(w("llm_model")?.value);
        if (w("llm_api_key")) pe.apiKeyInput.value = w("llm_api_key").value || "";
        if (w("llm_auto_enhance")) pe.autoCheck.checked = !!w("llm_auto_enhance").value;
        else pe.autoCheck.checked = false;
        if (w("llm_unload_after")) pe.unloadCheck.checked = !!w("llm_unload_after").value;
        const lang = resolveOutputLanguage(pe);
        const prevLang = pe._lastOutputLanguage;
        pe._lastOutputLanguage = lang;
        if (pe.detailCheck) {
            pe.detailCheck.checked = resolveCharacterFeatureEnhance(pe, { preferWidget: true });
        }
        if (w("llm_custom_template")) pe.templateArea.value = w("llm_custom_template").value || "";
        pe.updateApiFormatUI();
        if (prevLang !== null && lang !== prevLang) pe.fetchTemplate(true);
    };

    pe.syncToWidgets = () => {
        const set = (n, v) => { const w = pe.widget(n); if (w) w.value = v; };
        const url = coerceLlmUrl(pe.urlInput.value, defaultsForApiFormat(pe.apiSelect.value).url);
        set("llm_api_format", pe.apiSelect.value);
        set("llm_openai_compat_mode", normalizeOpenAiCompatMode(pe.compatSelect?.value));
        set("llm_url", url);
        set("llm_api_key", pe.apiKeyInput.value || "");
        set("llm_model", coerceLlmModel(pe.modelInput.value));
        set("llm_output_language", pe.langSelect.value || DEFAULT_OUTPUT_LANGUAGE);
        set("llm_character_feature_enhance", !!pe.detailCheck?.checked);
        set("llm_auto_enhance", !!pe.autoCheck.checked);
        set("llm_unload_after", pe.supportsUnload() && !!pe.unloadCheck.checked);
        const custom = pe.templateArea.value.trim();
        set("llm_custom_template", custom !== pe._currentDefaultTemplate ? custom : "");
        editor._markNodeDirtyLight?.();
    };

    pe.fetchModels = async (silent = false) => {
        if (pe._busy) return;
        pe._modelAbortController?.abort();
        const modelAbort = new AbortController();
        pe._modelAbortController = modelAbort;
        try {
            const llmUrl = coerceLlmUrl(pe.urlInput.value, defaultsForApiFormat(pe.apiSelect.value).url);
            pe.urlInput.value = llmUrl;
            pe.apiSelect.value = inferApiFormat(llmUrl, pe.apiSelect.value);
            pe.updateApiFormatUI();
            if (!silent) pe.setStatus("正在获取模型列表…", "loading");
            const resp = await api.fetchApi("/minimax/director/enhance_models", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    llm_url: llmUrl, api_format: pe.apiSelect.value,
                    openai_compat_mode: normalizeOpenAiCompatMode(pe.compatSelect?.value),
                    api_key: pe.apiKeyInput.value || "",
                }),
                signal: modelAbort.signal,
            });
            if (!resp.ok) {
                const { error } = await readApiError(resp, "获取模型失败");
                if (!silent) pe.setStatus(error, "error");
                return;
            }
            const data = await resp.json();
            pe.modelList.innerHTML = "";
            for (const name of data.models || []) {
                const o = document.createElement("option");
                o.value = name;
                pe.modelList.appendChild(o);
            }
            if (!pe.modelInput.value.trim()) pe.modelInput.value = defaultsForApiFormat(pe.apiSelect.value).model;
            pe.syncToWidgets();
            if (!silent) pe.setStatus(`${(data.models || []).length} 个模型（可手动输入名称）`, "success");
        } catch (e) {
            if (!silent && !isAbortError(e)) pe.setStatus(`连接失败: ${networkHint(e)}`, "error");
        } finally {
            if (pe._modelAbortController === modelAbort) pe._modelAbortController = null;
        }
    };

    pe.fetchTemplate = async (resetIfDefault = false) => {
        const task = resolveTaskKey(editor.getTaskKey?.() || "rv2v");
        const outputLanguage = resolveOutputLanguage(pe);
        try {
            const resp = await api.fetchApi("/minimax/director/get_template", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ task_type: task, output_language: outputLanguage }),
            });
            const data = await resp.json();
            if (data.template) {
                pe._currentDefaultTemplate = data.template;
                if (resetIfDefault || !pe.templateArea.value || pe.templateArea.value === pe._lastFetchedTemplate) {
                    pe.templateArea.value = data.template;
                }
                pe._lastFetchedTemplate = data.template;
            }
        } catch (e) { /* ignore */ }
    };

    pe.getPromptBlock = (segmentIndex = null) => {
        if (editor.isGlobalMode?.()) {
            editor.timeline.global = editor.timeline.global || {};
            return { block: editor.timeline.global, taskKey: resolveTaskKey(editor.getTaskKey?.() || "rv2v"), isGlobal: true };
        }
        const idx = segmentIndex ?? editor.selectedIndex ?? 0;
        const seg = editor.timeline.segments?.[idx];
        const global = editor.timeline.global || {};
        const taskKey = resolveTaskKey(seg?.taskType || global.taskType || editor.getTaskKey?.() || "rv2v");
        return { block: seg || global, taskKey, isGlobal: false, segmentIndex: idx };
    };

    pe.getPromptTextForBlock = (segmentIndex = null) => {
        if (editor.isGlobalMode?.()) {
            return (editor.globalPrompt?.value || editor.timeline.global?.prompt || editor.globalPromptWidget?.value || "").trim();
        }
        const idx = segmentIndex ?? editor.selectedIndex ?? 0;
        const seg = editor.timeline.segments?.[idx];
        const globalPrompt = (editor.timeline.global?.prompt || editor.globalPrompt?.value || "").trim();
        return (seg?.prompt || globalPrompt || "").trim();
    };

    pe.setPromptTextForBlock = (text, segmentIndex = null) => {
        if (editor.isGlobalMode?.()) {
            if (editor.globalPrompt) editor.globalPrompt.value = text;
            editor.timeline.global = editor.timeline.global || {};
            editor.timeline.global.prompt = text;
            if (editor.globalPromptWidget) editor.globalPromptWidget.value = text;
            return;
        }
        const idx = segmentIndex ?? editor.selectedIndex ?? 0;
        const seg = editor.timeline.segments?.[idx];
        if (seg) seg.prompt = text;
        if (idx === editor.selectedIndex && editor.segPrompt) editor.segPrompt.value = text;
    };

    pe.getActivePromptText = () => pe.getPromptTextForBlock();

    pe.setActivePromptText = (text) => {
        pe.setPromptTextForBlock(text);
        editor.commit?.(false, { syncTimeline: true });
    };

    pe.getLlmConfig = () => {
        if (pe.detailCheck) {
            pe.detailCheck.checked = resolveCharacterFeatureEnhance(pe, { preferWidget: true });
        }
        pe.syncToWidgets();
        const llmUrl = coerceLlmUrl(pe.urlInput.value, defaultsForApiFormat(pe.apiSelect.value).url);
        pe.urlInput.value = llmUrl;
        pe.apiSelect.value = inferApiFormat(llmUrl, pe.apiSelect.value);
        pe.updateApiFormatUI();
        const model = coerceLlmModel(pe.modelInput.value);
        pe.modelInput.value = model;
        const outputLanguage = resolveOutputLanguage(pe);
        const characterFeatureEnhance = resolveCharacterFeatureEnhance(pe, { preferWidget: true });
        const customTemplate = pe.templateArea.value.trim() !== pe._currentDefaultTemplate ? pe.templateArea.value.trim() : "";
        return {
            llmUrl, model, apiFormat: pe.apiSelect.value,
            openaiCompatMode: normalizeOpenAiCompatMode(pe.compatSelect?.value),
            apiKey: pe.apiKeyInput.value || "",
            outputLanguage,
            characterFeatureEnhance,
            customTemplate,
        };
    };

    pe.collectVisionImagesForBlock = async (block, taskKey) => {
        const images = [];
        let sourceCount = 0;
        let refCount = 0;
        const refSlots = [];
        const video = editor.timeline?.video || {};
        const videoFile = video.videoFile || video.fileName;
        const isOllama = pe.apiSelect?.value === API_OLLAMA;
        const sourceFrameCount = isOllama ? 2 : 3;
        if (videoFile && editor.getDirectorMode?.() === "video") {
            const resp = await api.fetchApi("/minimax/director/extract_frames", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    filename: videoFile,
                    subfolder: video.subfolder || "",
                    num_frames: sourceFrameCount,
                }),
            });
            const data = await resp.json();
            if (data.frames?.length) { sourceCount = data.frames.length; images.push(...data.frames); }
        }
        const global = editor.timeline.global || {};
        const refsBlock = block || global;
        if (taskUsesReferenceImages(taskKey) && refsBlock?.refs?.length) {
            const sortedRefs = [...refsBlock.refs]
                .filter((r) => r.imageFile || r.imageB64)
                .sort((a, b) => Number(a.index ?? a.slot ?? 0) - Number(b.index ?? b.slot ?? 0));
            for (const ref of sortedRefs) {
                const slot = Number(ref.index ?? ref.slot ?? 0);
                if (ref.imageFile) {
                    const b64 = await fetchImageB64(ref.imageFile);
                    if (b64) { images.push(b64); refCount += 1; refSlots.push(slot); }
                } else if (ref.imageB64) {
                    images.push(ref.imageB64.startsWith("data:") ? ref.imageB64.split(",", 2)[1] : ref.imageB64);
                    refCount += 1;
                    refSlots.push(slot);
                }
            }
        }
        let refVideoCount = 0;
        if (taskUsesReferenceVideo(taskKey)) {
            const rv = refsBlock?.referenceVideo || global.referenceVideo || {};
            const refVid = rv.videoFile || rv.fileName;
            if (refVid) {
                const resp = await api.fetchApi("/minimax/director/extract_frames", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        filename: refVid,
                        subfolder: rv.subfolder || "",
                        num_frames: isOllama ? 1 : 2,
                    }),
                });
                const data = await resp.json();
                if (data.frames?.length) {
                    refVideoCount = data.frames.length;
                    images.push(...data.frames);
                }
            }
        }
        return { images, sourceCount, refCount, refSlots, refVideoCount };
    };

    pe.callEnhanceApi = async (prompt, taskKey, block, cfg, signal = pe._abortController?.signal) => {
        let images = []; let refCount = 0; let sourceCount = 0; let refSlots = []; let refVideoCount = 0;
        try {
            pe.throwIfCancelled();
            ({
                images, refCount, sourceCount, refSlots, refVideoCount,
            } = await pe.collectVisionImagesForBlock(block, taskKey));
        } catch (e) {
            if (isAbortError(e)) throw e;
            console.warn("[MiniMax H3 PE] vision collect failed:", e);
            pe.setStatus("素材读取失败，将不附带图片继续扩写", "info");
        }
        pe.throwIfCancelled();
        pe.setStatus(images.length ? `正在扩写（附带 ${images.length} 张参考图）…` : "正在扩写…", "loading");
        try {
            pe.throwIfCancelled();
            const resp = await api.fetchApi("/minimax/director/enhance", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                    llm_url: cfg.llmUrl, model: cfg.model, prompt, task_type: taskKey,
                    image_num: Math.max(1, refCount), images, api_format: cfg.apiFormat,
                    openai_compat_mode: cfg.openaiCompatMode,
                    api_key: cfg.apiKey, output_language: cfg.outputLanguage,
                    character_feature_enhance: cfg.characterFeatureEnhance,
                    source_count: sourceCount, ref_slots: refSlots, ref_video_count: refVideoCount,
                        llm_unload_after: pe.supportsUnload() && !!pe.unloadCheck.checked, custom_template: cfg.customTemplate,
                    }),
                signal,
            });
                if (!resp.ok) {
                const { error } = await readApiError(resp, "扩写请求失败");
                return { ok: false, text: "", error, vision: { images, sourceCount, refCount } };
            }
            let data = {};
            try { data = await resp.json(); } catch { data = {}; }
            return {
                ok: !!data.response,
                text: data.response || "",
                error: data.error || (data.response ? "" : "LLM 扩写返回为空；请更换非思考型模型或查看 ComfyUI 日志"),
                hanCount: data.han_count,
                detailedMode: !!data.detailed_mode,
                detailTargetHan: data.detail_target_han,
                vision: { images, sourceCount, refCount },
            };
        } catch (e) {
            if (isAbortError(e)) throw e;
            throw new Error(networkHint(e));
        }
    };

    pe.enhanceOneTarget = async (segmentIndex, cfg, activeBtn, label, ordinal = 0, total = 0) => {
        const { block, taskKey } = pe.getPromptBlock(segmentIndex);
        const prompt = pe.getPromptTextForBlock(segmentIndex);
        if (!prompt) return { ok: false, skipped: true, reason: "empty" };
        pe.setEnhanceLoading(true, activeBtn, label);
        pe.updateProgress(ordinal, total, label);
        pe.setStatus(`正在扩写: ${label}…`, "loading");
        const result = await pe.callEnhanceApi(prompt, taskKey, block, cfg);
        if (result.ok) {
            const text = taskKey === "fl2v" ? stripFl2vPromptBody(result.text) : result.text;
            pe.setPromptTextForBlock(text, segmentIndex);
            return { ok: true, chars: text.length, taskKey, result: { ...result, text } };
        }
        return { ok: false, error: result.error, taskKey };
    };

    pe.enhancePrompt = async (scope = "current") => {
        if (pe._busy) return;
        const cfg = pe.getLlmConfig();
        if (!cfg.model) { pe.setStatus("请输入模型名称", "error"); return; }
        if ((cfg.apiFormat === API_ZHIPU) && !cfg.apiKey) {
            pe.setStatus("请填写 API Key（或设置环境变量）", "error");
            return;
        }

        const activeBtn = scope === "all" ? pe.enhanceAllBtn : pe.enhanceCurrentBtn;

        if (scope === "current") {
            const prompt = pe.getActivePromptText();
            if (!prompt) { pe.setStatus("请先输入提示词", "error"); return; }
            pe.setEnhanceLoading(true, activeBtn, "准备中…");
            try {
                const { block, taskKey } = pe.getPromptBlock();
                pe.setEnhanceLoading(true, activeBtn, "收集素材…");
                const result = await pe.callEnhanceApi(prompt, taskKey, block, cfg);
                const v = result.vision || {};
                if (v.images?.length) {
                    pe.visionBadge.textContent = `${v.sourceCount ? v.sourceCount + " 视频帧" : ""}${v.sourceCount && v.refCount ? " + " : ""}${v.refCount ? v.refCount + " 参考图" : ""}`;
                    pe.visionBadge.style.display = "inline";
                } else {
                    pe.visionBadge.style.display = "none";
                }
                if (result.ok) {
                    const text = taskKey === "fl2v" ? stripFl2vPromptBody(result.text) : result.text;
                    pe.setActivePromptText(text);
                    pe.setStatus(formatEnhanceSuccessStatus(taskKey, { ...result, text }), "success");
                } else {
                    pe.setStatus(result.error, "error");
                }
            } catch (e) {
                pe.setStatus(isAbortError(e) ? "已取消扩写，原文未修改" : `请求失败: ${networkHint(e)}`, isAbortError(e) ? "info" : "error");
            } finally {
                pe.setEnhanceLoading(false);
            }
            return;
        }

        // scope === "all"
        if (editor.isGlobalMode?.()) {
            const prompt = pe.getActivePromptText();
            if (!prompt) { pe.setStatus("请先输入全局提示词", "error"); return; }
            try {
                const r = await pe.enhanceOneTarget(null, cfg, activeBtn, "全局提示词");
                if (r.ok) {
                    editor.commit?.(false, { syncTimeline: true });
                    pe.setStatus(formatEnhanceSuccessStatus(r.taskKey, r.result || {}), "success");
                } else if (!r.skipped) {
                    pe.setStatus(r.error || "扩写失败", "error");
                }
            } catch (e) {
                pe.setStatus(isAbortError(e) ? "已取消扩写，原文未修改" : `请求失败: ${networkHint(e)}`, isAbortError(e) ? "info" : "error");
            } finally {
                pe.setEnhanceLoading(false);
            }
            return;
        }

        const segments = editor.timeline.segments || [];
        const targets = segments.map((_, i) => i).filter((i) => pe.getPromptTextForBlock(i));
        if (!targets.length) {
            pe.setStatus("没有可扩写的分段提示词（请先填写各片段或全局提示词）", "error");
            return;
        }

        let okCount = 0;
        let lastError = "";
        let failedTarget = -1;
        try {
            for (let n = 0; n < targets.length; n++) {
                pe.throwIfCancelled();
                const idx = targets[n];
                const label = `片段 ${idx + 1}/${segments.length}`;
                const r = await pe.enhanceOneTarget(idx, cfg, activeBtn, label, n + 1, targets.length);
                if (r.ok) {
                    okCount += 1;
                    pe.updateProgress(okCount, targets.length, label);
                    pe.setStatus(`${label} 扩写成功 (${okCount}/${targets.length})`, "loading");
                } else if (!r.skipped) {
                    lastError = r.error || "未知错误";
                    failedTarget = idx;
                    pe.updateProgress(okCount, targets.length, label);
                    pe.setStatus(`${label} 失败：${lastError}`, "error", {
                        label: "重试这一段",
                        run: () => pe.retryFailedBatch(targets, cfg, activeBtn, failedTarget, okCount),
                    });
                    break;
                }
            }
            editor.commit?.(false, { syncTimeline: true });
            editor.updateSelectionUI?.();
            if (failedTarget < 0 && okCount === targets.length) {
                pe.setStatus(`全部扩写完成：${okCount} 个分段`, "success");
            } else if (failedTarget >= 0 && okCount > 0 && lastError) {
                pe.setStatus(`已暂停：${okCount}/${targets.length} 成功。${lastError}`, "error", {
                    label: "重试剩余片段",
                    run: () => pe.retryFailedBatch(targets, cfg, activeBtn, failedTarget, okCount),
                });
            } else if (failedTarget >= 0 && lastError) {
                pe.setStatus(`扩写失败：${lastError}`, "error", {
                    label: "重试全部",
                    run: () => pe.enhancePrompt("all"),
                });
            }
        } catch (e) {
            if (isAbortError(e)) {
                editor.commit?.(false, { syncTimeline: true });
                pe.setStatus(`已取消扩写：${okCount}/${targets.length} 成功，未处理片段保留原文`, "info");
            } else {
                pe.setStatus(`请求失败: ${networkHint(e)}`, "error", {
                    label: failedTarget >= 0 ? "重试剩余片段" : "重试全部",
                    run: () => (failedTarget >= 0
                        ? pe.retryFailedBatch(targets, cfg, activeBtn, failedTarget, okCount)
                        : pe.enhancePrompt("all")),
                });
            }
        } finally {
            pe.setEnhanceLoading(false);
        }
    };

    pe.retryFailedBatch = async (targets, cfg, activeBtn, failedTarget, alreadyOk = 0) => {
        if (pe._busy || failedTarget == null || failedTarget < 0) return;
        const remaining = targets.filter((idx) => Number(idx) >= Number(failedTarget));
        const totalRemaining = remaining.length;
        if (!totalRemaining) return;
        const maxSegment = targets.reduce((max, idx) => Math.max(max, Number(idx) + 1), 0);
        let done = 0;
        let nextFailed = -1;
        let lastError = "";
        try {
            for (const idx of remaining) {
                pe.throwIfCancelled();
                const label = `片段 ${idx + 1}/${maxSegment}`;
                const r = await pe.enhanceOneTarget(idx, cfg, activeBtn, label, ++done, totalRemaining);
                if (r.ok) continue;
                nextFailed = idx;
                lastError = r.error || "未知错误";
                pe.setStatus(`片段 ${idx + 1} 失败：${lastError}`, "error", {
                    label: "重试剩余片段",
                    run: () => pe.retryFailedBatch(targets, cfg, activeBtn, nextFailed, alreadyOk + done - 1),
                });
                break;
            }
            if (nextFailed < 0) pe.setStatus(`剩余片段扩写完成（本次 ${done} 段）`, "success");
        } catch (e) {
            pe.setStatus(isAbortError(e) ? `已取消重试：本次完成 ${Math.max(0, done - 1)} 段` : `请求失败: ${networkHint(e)}`, isAbortError(e) ? "info" : "error");
        } finally {
            pe.setEnhanceLoading(false);
        }
    };


    pe.unloadModel = async () => {
        const llmUrl = coerceLlmUrl(pe.urlInput.value, defaultsForApiFormat(pe.apiSelect.value).url);
        const apiFormat = pe.apiSelect.value;
        const openaiCompatMode = normalizeOpenAiCompatMode(pe.compatSelect?.value);
        const model = coerceLlmModel(pe.modelInput.value);
        if (!model) { pe.setStatus("请输入模型名称", "error"); return; }
        if (!pe.supportsUnload()) {
            pe.setStatus("当前 API 格式不支持手动卸载模型", "error");
            return;
        }
        pe.setStatus("正在卸载模型…", "loading");
        try {
            const resp = await api.fetchApi("/minimax/director/unload_model", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    llm_url: llmUrl,
                    model,
                    api_format: apiFormat,
                    openai_compat_mode: openaiCompatMode,
                    api_key: pe.apiKeyInput.value || "",
                }),
            });
            if (!resp.ok) {
                const { error } = await readApiError(resp, "卸载失败");
                pe.setStatus(error, "error");
                return;
            }
            let data = {};
            try { data = await resp.json(); } catch {}
            if (resp.ok && data.status === "unloaded") {
                pe.setStatus(`${data.provider || "LLM"} 模型已卸载`, "success");
            } else {
                pe.setStatus(data.error || "卸载失败", "error");
            }
        } catch (e) {
            pe.setStatus(`卸载失败: ${networkHint(e)}`, "error");
        }
    };
    pe.unloadOllama = pe.unloadModel;

    pe.onTaskTypeChanged = () => pe.fetchTemplate();
    pe.handleServerEnhanced = (payload) => {
        if (!payload || String(payload.node) !== String(editor.node.id)) return;
        let text = payload.text || "";
        if (editor.isFl2vMode?.() || resolveTaskKey(editor.getTaskKey?.() || "") === "fl2v") {
            text = stripFl2vPromptBody(text);
        }
        pe.setActivePromptText(text);
        pe.setStatus(`自动扩写已应用（${text.length} 字符）`, "success");
    };

    pe._lastOutputLanguage = null;
    pe.syncFromWidgets();
    editor._promptEnhancer = pe;
    pe.fetchTemplate(true);
    return pe;
}

export function getPromptEnhancerPanelHeight(editor) {
    const pe = editor?._promptEnhancer;
    if (!pe?.open) return PE_PANEL_COLLAPSED_H;
    return PE_PANEL_COLLAPSED_H + PE_PANEL_EXPANDED_H;
}

export function registerDirectorPromptEnhancerEvents(findDirectorNode) {
    api.addEventListener("minimax_director_enhanced", ({ detail }) => {
        findDirectorNode(detail?.node)?._minimaxEditor?._promptEnhancer?.handleServerEnhanced?.(detail);
    });
}
