import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Card UI for MiniMaxH3TimelineEditor + @-mention picker for
// MiniMaxH3ConditioningTimelineIntegration's prompt widget.
//
// The Timeline Editor card UI is deliberately built the same way
// ComfyUI-MiniMaxH3-Hybrid's timeline UI was: media is uploaded directly
// (the same /upload/image endpoint native LoadImage uses) and stored as one
// hidden JSON widget (media_json) that the Python node parses -- not real
// per-item graph sockets. A real socket per media slot means LiteGraph
// renders one input row per slot regardless of whether it's connected, so a
// node with room for many items becomes a wall of dots no matter how the
// widgets below are handled; a single hidden widget avoids that entirely.
// This file only renders the real DOM (via node.addDOMWidget), so CSS
// flexbox gives the actual side-by-side card layout.

const TIMELINE_CLASS = "MiniMaxH3TimelineEditor";
const INTEGRATION_CLASS = "MiniMaxH3ConditioningTimelineIntegration";
// No architectural limit on item count -- PackedLayout just adds more rows
// to the packed self-attention sequence per item. 40 is a practical ceiling
// well past what VRAM/sampling time makes usable, matching nodes_timeline.py's
// own MAX_MEDIA.
const MAX_ITEMS = 40;
const IMAGE_EXT = new Set(["png", "jpg", "jpeg", "webp", "gif", "bmp", "avif", "tif", "tiff"]);
const VIDEO_EXT = new Set(["mp4", "webm", "mov", "mkv", "avi", "m4v"]);
const AUDIO_EXT = new Set(["mp3", "wav", "flac", "ogg", "m4a", "aac", "opus"]);

const ROLE_START = "keyframe_start";
const ROLE_END = "keyframe_end";
const ROLE_MID = "keyframe_mid";
const ROLE_REF = "reference";
const ROLES = [ROLE_START, ROLE_END, ROLE_MID, ROLE_REF];
const ROLE_LABELS = { [ROLE_START]: "Start", [ROLE_END]: "End", [ROLE_MID]: "Mid", [ROLE_REF]: "Ref" };

let cssInstalled = false;
function installStyles() {
    if (cssInstalled) return;
    cssInstalled = true;
    const style = document.createElement("style");
    style.textContent = `
.h3c-root{box-sizing:border-box;width:100%;font:11px system-ui,sans-serif;color:#ddd;display:flex;flex-direction:column;gap:6px}
.h3c-row{display:flex;flex-direction:row;gap:8px;flex-wrap:wrap}
.h3c-card{flex:0 0 160px;display:flex;flex-direction:column;gap:4px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.12);border-radius:6px;padding:6px;position:relative}
.h3c-drop{height:90px;border:1.5px dashed #587084;border-radius:5px;display:flex;align-items:center;justify-content:center;text-align:center;color:#8fa3b2;cursor:pointer;overflow:hidden;background:#0d1217}
.h3c-drop.over{background:#1b2933;border-color:#8dd7ff}
.h3c-drop img,.h3c-drop video{width:100%;height:100%;object-fit:cover}
.h3c-remove{position:absolute;top:4px;right:4px;width:16px;height:16px;border-radius:3px;background:rgba(255,80,80,.3);color:#f88;border:1px solid rgba(255,80,80,.5);cursor:pointer;font-size:11px;line-height:14px;text-align:center;padding:0}
.h3c-modebar{display:flex;gap:3px}
.h3c-modebar button{flex:1;padding:3px 2px;font-size:10px;background:rgba(255,255,255,0.06);color:#eee;border:1px solid rgba(255,255,255,0.15);border-radius:3px;cursor:pointer}
.h3c-modebar button.active{background:rgba(120,170,255,0.35);border-color:#7aaeff}
.h3c-sub{display:flex;flex-direction:column;gap:3px}
.h3c-seconds{display:flex;align-items:center;gap:4px;font-size:10px;color:#ccc}
.h3c-seconds input{width:44px;background:#111a21;color:#ddd;border:1px solid #40515e;border-radius:3px;padding:1px 3px}
.h3c-add{flex:0 0 90px;display:flex;align-items:center;justify-content:center;border:1.5px dashed #7edca0;border-radius:6px;color:#bff3d0;cursor:pointer;font-size:24px;font-weight:700;background:rgba(126,235,167,0.06)}
.h3c-add:hover{background:rgba(126,235,167,0.16)}
.h3c-hint{font-size:10px;color:#888}
.h3c-tag{font-size:10px;color:#7edca0;font-weight:700}

.h3tl-mention-menu{position:fixed;z-index:10000;min-width:200px;max-width:320px;max-height:260px;overflow-y:auto;background:var(--comfy-menu-bg,#202020);border:1px solid rgba(255,255,255,.15);border-radius:6px;box-shadow:0 6px 20px rgba(0,0,0,.4);font-size:12px;padding:4px}
.h3tl-mention-title{padding:4px 6px 6px;color:rgba(255,255,255,.55);font-size:11px}
.h3tl-mention-item{display:flex;justify-content:space-between;gap:8px;padding:5px 7px;border-radius:4px;cursor:pointer;color:rgba(255,255,255,.92)}
.h3tl-mention-item.is-active,.h3tl-mention-item:hover{background:rgba(160,255,178,.16)}
.h3tl-mention-item .tag{font-weight:700}
.h3tl-mention-item .detail{color:rgba(255,255,255,.5);font-size:11px;white-space:nowrap}
.h3tl-mention-empty{padding:6px 7px;color:rgba(255,255,255,.55)}
`;
    document.head.appendChild(style);
}

function extOf(name) {
    return String(name || "").split(".").pop().toLowerCase();
}

function mediaTypeFor(file) {
    const mime = String(file.type || "").toLowerCase();
    if (mime.startsWith("image/")) return "image";
    if (mime.startsWith("video/")) return "video";
    if (mime.startsWith("audio/")) return "audio";
    const ext = extOf(file.name);
    if (IMAGE_EXT.has(ext)) return "image";
    if (VIDEO_EXT.has(ext)) return "video";
    if (AUDIO_EXT.has(ext)) return "audio";
    return null;
}

async function uploadFile(file) {
    const form = new FormData();
    form.append("image", file, file.name);
    form.append("type", "input");
    form.append("overwrite", "false");
    const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
    if (!response.ok) throw new Error(`upload failed (${response.status})`);
    const result = await response.json();
    return result.subfolder ? `${result.subfolder}/${result.name}` : result.name;
}

function viewUrl(filename) {
    return api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input`);
}

function defaultItem() {
    // anchor_seconds is keyframe_mid-only now (its real placement frame).
    // References had their own anchor_channel/anchor_closeness controls;
    // removed after testing showed they weren't needed for co-presence --
    // see nodes_timeline.py's module docstring.
    // noise_aug: real per-item control (unlike the removed anchor system) --
    // patches the DiT's own _cond_video_rows/_cond_audio_rows so each row's
    // value is genuinely independent, not just a global scalar. Default
    // matches the native per-modality default (visual 0.999, audio 1.0),
    // resolved server-side from the item's actual media type if left unset.
    return { filename: null, type: null, role: ROLE_REF, anchor_seconds: -1, noise_aug: 0.999 };
}

// --- Timeline Editor card UI -------------------------------------------

function installTimelineUI(node) {
    if (node.__h3cInstalled) return;
    node.__h3cInstalled = true;
    installStyles();

    const dataWidget = node.widgets?.find((w) => w.name === "media_json");
    if (!dataWidget) return;
    dataWidget.hidden = true;
    dataWidget.options = { ...(dataWidget.options || {}), hidden: true };
    dataWidget.draw = () => {};
    dataWidget.computeSize = () => [0, -4];

    let items = [];
    // Reads the widget's CURRENT value, not a snapshot -- onNodeCreated
    // (where install() runs) fires BEFORE LiteGraph applies a loaded
    // workflow's saved widgets_values via node.configure(). Reading
    // dataWidget.value only once here would capture the still-default "[]"
    // for any node restored from a saved workflow.
    function syncFromWidget() {
        try {
            const parsed = JSON.parse(dataWidget.value || "[]");
            items = Array.isArray(parsed) ? parsed.map((i) => ({ ...defaultItem(), ...i })) : [];
        } catch {
            items = [];
        }
    }
    syncFromWidget();

    const root = document.createElement("div");
    root.className = "h3c-root";
    const hint = document.createElement("div");
    hint.className = "h3c-hint";
    hint.textContent = "drop or click a slot to add an image / video / audio file";
    const row = document.createElement("div");
    row.className = "h3c-row";
    root.appendChild(hint);
    root.appendChild(row);

    const commit = () => {
        dataWidget.value = JSON.stringify(items);
        dataWidget.callback?.(dataWidget.value);
        node.graph?.setDirtyCanvas(true, true);
    };

    function buildCard(item, index) {
        const card = document.createElement("div");
        card.className = "h3c-card";

        const remove = document.createElement("button");
        remove.className = "h3c-remove";
        remove.textContent = "x";
        remove.onclick = (e) => { e.stopPropagation(); items.splice(index, 1); commit(); render(); };
        card.appendChild(remove);

        const drop = document.createElement("div");
        drop.className = "h3c-drop";
        drop.textContent = item.filename ? "" : "Input image / video / audio";
        if (item.filename) {
            if (item.type === "image") {
                const img = document.createElement("img");
                img.src = viewUrl(item.filename);
                drop.appendChild(img);
            } else if (item.type === "video") {
                const video = document.createElement("video");
                video.src = viewUrl(item.filename);
                video.muted = true;
                video.loop = true;
                video.autoplay = true;
                drop.appendChild(video);
            } else {
                drop.textContent = "\u{1F50A} " + item.filename.split("/").pop();
            }
        }
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "image/*,video/*,audio/*";
        fileInput.style.display = "none";
        fileInput.onchange = () => handleFiles(fileInput.files, index);
        drop.appendChild(fileInput);
        drop.onclick = () => fileInput.click();
        drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
        drop.ondragleave = () => drop.classList.remove("over");
        drop.ondrop = (e) => {
            e.preventDefault();
            drop.classList.remove("over");
            if (e.dataTransfer?.files?.length) handleFiles(e.dataTransfer.files, index);
        };
        card.appendChild(drop);

        // Role only matters once a file is attached -- an empty card has
        // nothing to place on the timeline yet.
        if (item.filename) {
            const modebar = document.createElement("div");
            modebar.className = "h3c-modebar";
            ROLES.forEach((roleValue) => {
                const btn = document.createElement("button");
                btn.textContent = ROLE_LABELS[roleValue];
                btn.className = item.role === roleValue ? "active" : "";
                btn.onclick = (e) => {
                    e.stopPropagation();
                    item.role = roleValue;
                    if ((roleValue === ROLE_START || roleValue === ROLE_END) && item.anchor_seconds >= 0) {
                        item.anchor_seconds = -1;
                    }
                    if (roleValue === ROLE_MID && item.anchor_seconds < 0) {
                        item.anchor_seconds = 0;
                    }
                    commit();
                    render();
                };
                modebar.appendChild(btn);
            });
            card.appendChild(modebar);

            // keyframe_start/keyframe_end always pin to the clip's first/last
            // frame server-side -- an anchor field there would show a control
            // with no effect. keyframe_mid REQUIRES an anchor (that's the only
            // way it's placed), so its field is always shown. References used
            // to have their own anchor/channel/closeness controls here;
            // removed after testing showed plain unanchored references
            // already produce clean multi-reference co-presence on their own
            // (see nodes_timeline.py's module docstring) -- nothing left for
            // a reference card to control beyond role and the file itself.
            if (item.role === ROLE_MID) {
                appendSecondsRow(card, item, "at:", "anchor_seconds", commit);
            }

            // Real per-item control, unlike the removed anchor system --
            // patches the DiT's own row-building so this item's value is
            // genuinely independent of every other item's. 1.0 = this row's
            // content is used exactly as given; lower values blend it toward
            // noise before the model sees it, and make the model treat it as
            // less "already resolved," which can soften a hard cut into a
            // mid-clip keyframe at the cost of matching it less exactly.
            const augRow = document.createElement("div");
            augRow.className = "h3c-seconds";
            const augLabel = document.createElement("span");
            augLabel.textContent = "noise_aug:";
            const augInput = document.createElement("input");
            augInput.type = "number";
            augInput.step = "0.01";
            augInput.min = "0";
            augInput.max = "1";
            augInput.value = item.noise_aug ?? 0.999;
            augInput.onclick = (e) => e.stopPropagation();
            augInput.onchange = () => {
                const v = Math.max(0, Math.min(1, Number(augInput.value)));
                item.noise_aug = Number.isNaN(v) ? 0.999 : v;
                augInput.value = item.noise_aug;
                commit();
            };
            augRow.append(augLabel, augInput);
            card.appendChild(augRow);

            const tag = document.createElement("div");
            tag.className = "h3c-tag";
            tag.textContent = String(item.type || "").toUpperCase();
            card.appendChild(tag);
        }
        return card;
    }

    function appendSecondsRow(card, item, label, field, onCommit, showUnit = true) {
        const secRow = document.createElement("div");
        secRow.className = "h3c-seconds";
        const secLabel = document.createElement("span");
        secLabel.textContent = label;
        const secInput = document.createElement("input");
        secInput.type = "number";
        secInput.step = "0.05";
        secInput.min = "0";
        secInput.value = Math.max(0, Number(item[field] ?? 0));
        secInput.onclick = (e) => e.stopPropagation();
        secInput.onchange = () => { item[field] = Math.max(0, Number(secInput.value) || 0); onCommit(); };
        secRow.append(secLabel, secInput);
        if (showUnit) {
            const s = document.createElement("span");
            s.textContent = "s";
            secRow.append(s);
        }
        card.appendChild(secRow);
    }

    async function handleFiles(fileList, index) {
        const file = fileList?.[0];
        if (!file) return;
        const type = mediaTypeFor(file);
        if (!type) return;
        try {
            const filename = await uploadFile(file);
            items[index].filename = filename;
            items[index].type = type;
            commit();
            render();
        } catch (err) {
            console.error("[MiniMax H3 Timeline] upload failed", err);
        }
    }

    const CARD_W = 160, ADD_W = 90, GAP = 8, ROW_PAD = 40;
    function widthFor() {
        const addTile = items.length < MAX_ITEMS ? ADD_W + GAP : 0;
        const content = items.length * (CARD_W + GAP) + addTile;
        return Math.max(360, content + ROW_PAD);
    }

    function render() {
        row.innerHTML = "";
        items.forEach((item, i) => row.appendChild(buildCard(item, i)));
        if (items.length < MAX_ITEMS) {
            const add = document.createElement("div");
            add.className = "h3c-add";
            add.textContent = "+";
            add.onclick = () => { items.push(defaultItem()); commit(); render(); };
            row.appendChild(add);
        }
        const w = Math.max(widthFor(), node.size?.[0] || 0);
        if (node.size && node.size[0] !== w) {
            node.setSize([w, node.size[1]]);
        }
        node.setDirtyCanvas(true, true);
    }
    render();

    const heightFor = () => 30 + (items.length ? 220 : 90);
    if (node.addDOMWidget) {
        const domWidget = node.addDOMWidget("minimax_h3_timeline_ui", "custom", root, {
            serialize: false,
            hideOnZoom: false,
            getHeight: heightFor,
        });
        domWidget.computeSize = () => [Math.max(widthFor(), node.size?.[0] || 360), heightFor()];
    }

    // Re-sync once the real saved value has actually landed -- onConfigure
    // fires after LiteGraph has applied the loaded workflow's widgets_values.
    const onConfigure = node.onConfigure;
    node.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        syncFromWidget();
        render();
    };
}

app.registerExtension({
    name: "MiniMaxH3TimelineCardUI",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TIMELINE_CLASS) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            installTimelineUI(this);
            return result;
        };
    },
});

// --- @-mention picker for MiniMaxH3ConditioningTimelineIntegration -----
//
// Reads the connected MiniMaxH3TimelineEditor node's media_json widget
// directly and computes the exact same <Picture N>/<Video N>/<Audio N>
// ordinals nodes_timeline.py's _combined_conditioning assigns (role ==
// "reference" items, in array order, bucketed by media type, numbered
// 1-based per bucket) -- so the tag you pick here is guaranteed to be the
// tag the model actually receives, not a guess.

const TYPE_LABELS = { image: "Picture", video: "Video", audio: "Audio" };

function findWidget(node, name) {
    return node?.widgets?.find((w) => w.name === name);
}

function getConnectedTimelineNode(integrationNode) {
    const input = integrationNode.inputs?.find((i) => i.name === "timeline");
    if (!input?.link) return null;
    const link = integrationNode.graph?.links?.[input.link];
    if (!link) return null;
    return integrationNode.graph?.getNodeById?.(link.origin_id) || null;
}

function getTimelineCandidates(integrationNode) {
    const timelineNode = getConnectedTimelineNode(integrationNode);
    if (!timelineNode || timelineNode.type !== TIMELINE_CLASS) return { connected: false, items: [] };
    const dataWidget = findWidget(timelineNode, "media_json");
    let raw = [];
    try {
        const parsed = JSON.parse(dataWidget?.value || "[]");
        raw = Array.isArray(parsed) ? parsed : [];
    } catch {
        raw = [];
    }

    const counters = { image: 0, video: 0, audio: 0 };
    const items = [];
    raw.forEach((entry, index) => {
        if (!entry || entry.role !== ROLE_REF || !entry.filename) return;
        const mediaType = ["image", "video", "audio"].includes(entry.type) ? entry.type : "image";
        counters[mediaType] = (counters[mediaType] || 0) + 1;
        const ordinal = counters[mediaType];
        const label = TYPE_LABELS[mediaType] || mediaType;
        items.push({
            tag: `<${label} ${ordinal}>`,
            detail: String(entry.filename).split("/").pop(),
        });
    });
    return { connected: true, items };
}

function closeMenu(state) {
    state.menu?.remove();
    state.menu = null;
    state.activeIndex = 0;
}

function findAtQuery(textarea) {
    const caret = textarea.selectionStart ?? textarea.value.length;
    const text = textarea.value.slice(0, caret);
    const at = text.lastIndexOf("@");
    if (at === -1) return null;
    const between = text.slice(at + 1, caret);
    if (/[\s\n]/.test(between)) return null;
    return { at, caret, query: between };
}

function insertTag(textarea, at, caret, tag) {
    const before = textarea.value.slice(0, at);
    const after = textarea.value.slice(caret);
    const insertion = `${tag} `;
    textarea.value = `${before}${insertion}${after}`;
    const newCaret = before.length + insertion.length;
    textarea.setSelectionRange(newCaret, newCaret);
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    textarea.focus();
}

function renderMenu(node, state, textarea) {
    installStyles();
    closeMenu(state);
    const { connected, items } = getTimelineCandidates(node);
    const match = findAtQuery(textarea);
    if (!match) return;
    const query = match.query.toLowerCase();
    const filtered = items.filter((it) => !query || it.tag.toLowerCase().includes(query));

    const menu = document.createElement("div");
    menu.className = "h3tl-mention-menu";
    const title = document.createElement("div");
    title.className = "h3tl-mention-title";
    title.textContent = connected ? "Reference media in this Timeline" : "Connect a Timeline Editor first";
    menu.appendChild(title);

    if (!filtered.length) {
        const empty = document.createElement("div");
        empty.className = "h3tl-mention-empty";
        empty.textContent = connected
            ? "No reference-role items yet -- set an item's role to \"Ref\" on the Timeline Editor"
            : "Wire MiniMax H3 Timeline Editor -> timeline into this node";
        menu.appendChild(empty);
    } else {
        filtered.forEach((option, index) => {
            const item = document.createElement("div");
            item.className = `h3tl-mention-item${index === state.activeIndex ? " is-active" : ""}`;
            const tag = document.createElement("span");
            tag.className = "tag";
            tag.textContent = option.tag;
            const detail = document.createElement("span");
            detail.className = "detail";
            detail.textContent = option.detail;
            item.append(tag, detail);
            item.addEventListener("mousedown", (event) => {
                event.preventDefault();
                insertTag(textarea, match.at, match.caret, option.tag);
                closeMenu(state);
            });
            menu.appendChild(item);
        });
    }

    const rect = textarea.getBoundingClientRect();
    menu.style.left = `${Math.round(rect.left)}px`;
    menu.style.top = `${Math.round(rect.bottom + 4)}px`;
    document.body.appendChild(menu);
    state.menu = menu;
    state.items = filtered;
}

function attachMentionPicker(node) {
    const promptWidget = findWidget(node, "prompt");
    const textarea = promptWidget?.inputEl;
    if (!textarea || textarea.__h3tlMentionBound) return;
    textarea.__h3tlMentionBound = true;

    const state = { menu: null, activeIndex: 0, items: [] };

    const refresh = () => {
        const match = findAtQuery(textarea);
        if (!match) {
            closeMenu(state);
            return;
        }
        renderMenu(node, state, textarea);
    };

    textarea.addEventListener("input", refresh);
    textarea.addEventListener("keydown", (event) => {
        if (!state.menu) return;
        if (event.key === "Escape") {
            closeMenu(state);
            return;
        }
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            if (!state.items.length) return;
            event.preventDefault();
            const dir = event.key === "ArrowDown" ? 1 : -1;
            state.activeIndex = (state.activeIndex + dir + state.items.length) % state.items.length;
            renderMenu(node, state, textarea);
            return;
        }
        if (event.key === "Enter" || event.key === "Tab") {
            if (!state.items.length) return;
            event.preventDefault();
            const match = findAtQuery(textarea);
            if (!match) return;
            insertTag(textarea, match.at, match.caret, state.items[state.activeIndex].tag);
            closeMenu(state);
        }
    });
    textarea.addEventListener("blur", () => {
        setTimeout(() => closeMenu(state), 150);
    });
}

app.registerExtension({
    name: "MiniMaxH3TimelineMentions",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== INTEGRATION_CLASS) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            attachMentionPicker(this);
            return result;
        };
    },
});
