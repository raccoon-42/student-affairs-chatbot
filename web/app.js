/* ---------- interface language ----------
 * UI chrome only — the bot's answers follow the user's own language. */
const STRINGS = {
  tr: {
    chats: "Sohbetler",
    newConversation: "Yeni konuşma",
    emptyState: "Akademik takvim, kayıtlar ve yönetmelikler hakkında soru sorabilirsin.",
    placeholder: "Sorunu yaz...",
    limitTitle: "Mesaj limitine ulaştın",
    limitFallback: "Mesaj limitine ulaştın. Lütfen bir süre sonra tekrar dene.",
    abuseTitle: "Uygunsuz dil",
    abuseNote: "Uygunsuz dil nedeniyle mesaj gönderimi engellendi.",
    profileTitle: "Seni tanıyalım",
    profileText: "Sorularını daha iyi yanıtlayabilmek için eğitim durumunu seç.",
    eduAday: "Aday öğrenci (İYTE'yi düşünüyorum)",
    eduLisans: "Lisans öğrencisi",
    eduYl: "Yüksek lisans öğrencisi",
    eduDoktora: "Doktora öğrencisi",
    save: "Kaydet",
    error: (message) => `Bir şeyler ters gitti (${message}). Tekrar dener misin?`,
    disclaimer: "İyteBot hata yapabilir. Önemli bilgileri öğrenci işlerinden doğrulayın.",
    viewSources: "Kaynakları görüntüle",
    noSources: "Bu yanıt için kaynak bilgisi yok.",
    today: "Bugün",
    micDenied: "Mikrofon izni verilmedi.",
    micError: "Ses tanıma başarısız oldu.",
    settingsTitle: "Ayarlar",
    imageError: "Görsel eklenemedi.",
    imageMessage: "Görsel",
    devMode: "Geliştirici modu",
    devModeNote: "Her yanıtın altında sunucu loglarını gösterir.",
    tabGeneral: "Genel",
    tabProfile: "Profil",
    tabDeveloper: "Geliştirici",
    rowLanguage: "Dil",
    rowTheme: "Tema",
    rowModel: "Model",
    rowModelNote: "Haiku hızlı, Sonnet daha güçlü.",
    modelHaiku: "Haiku 4.5",
    modelSonnet: "Sonnet 5",
    themeLight: "Açık",
    themeDark: "Koyu",
    yesterday: "Dün",
    previous7: "Önceki 7 gün",
    older: "Daha eski",
    stageRewriting: "ton ayarlanıyor…",
    stageSearching: "kaynaklarda aranıyor…",
    stageWriting: "yanıt yazılıyor…",
    tokenUsageHint: "bağlam + yanıt (token) · maliyet",
  },
  en: {
    chats: "Chats",
    newConversation: "New chat",
    emptyState: "Ask about the academic calendar, registration and regulations.",
    placeholder: "Type your question...",
    limitTitle: "Message limit reached",
    limitFallback: "You have reached the message limit. Please try again later.",
    abuseTitle: "Inappropriate language",
    abuseNote: "Messaging has been disabled due to inappropriate language.",
    profileTitle: "Tell us about yourself",
    profileText: "Pick your education status so answers fit you better.",
    eduAday: "Prospective student (considering IZTECH)",
    eduLisans: "Undergraduate student",
    eduYl: "Master's student",
    eduDoktora: "PhD student",
    save: "Save",
    error: (message) => `Something went wrong (${message}). Please try again.`,
    disclaimer: "İyteBot can make mistakes. Verify important info with student affairs.",
    viewSources: "View sources",
    noSources: "No source info for this answer.",
    today: "Today",
    micDenied: "Microphone permission was denied.",
    micError: "Transcription failed.",
    settingsTitle: "Settings",
    imageError: "Could not attach the image.",
    imageMessage: "Image",
    devMode: "Developer mode",
    devModeNote: "Shows server logs under every answer.",
    tabGeneral: "General",
    tabProfile: "Profile",
    tabDeveloper: "Developer",
    rowLanguage: "Language",
    rowTheme: "Theme",
    rowModel: "Model",
    rowModelNote: "Haiku is fast, Sonnet is more capable.",
    modelHaiku: "Haiku 4.5",
    modelSonnet: "Sonnet 5",
    themeLight: "Light",
    themeDark: "Dark",
    yesterday: "Yesterday",
    previous7: "Previous 7 days",
    older: "Older",
    stageRewriting: "toning your voice…",
    stageSearching: "searching sources…",
    stageWriting: "writing…",
    tokenUsageHint: "context + answer (tokens) · cost",
  },
};

let lang = localStorage.getItem("lang") || "tr";
let gisReady = false; // declared early: applyTheme() runs before the auth section

function t(key, ...args) {
  const entry = STRINGS[lang][key];
  return typeof entry === "function" ? entry(...args) : entry;
}

function applyLanguage() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
  document.getElementById("lang-toggle").textContent = lang === "tr" ? "EN" : "TR";
}

const messagesEl = document.getElementById("messages");
const listEl = document.getElementById("conversation-list");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendButton = document.getElementById("send");
const emptyState = document.getElementById("empty-state");

/* ---------- storage ----------
 * Anonymous: conversations live in localStorage:
 *   [{id, title, updated, messages: [{role, text}]}]
 * Signed in (serverMode): the same shape is loaded from /conversations;
 * transcripts fetch lazily on select and the server records exchanges,
 * so nothing is written to localStorage.
 * Either way the conversation id doubles as the API session_id. */

let serverMode = false;

function loadConversations() {
  return JSON.parse(localStorage.getItem("conversations") || "[]");
}

function saveConversations() {
  if (serverMode) return; // the server already recorded the exchange
  localStorage.setItem("conversations", JSON.stringify(conversations));
}

let conversations = loadConversations();
let currentId = localStorage.getItem("current_id");

async function enterServerMode() {
  serverMode = true;
  const rows = await fetch("/conversations").then((r) => (r.ok ? r.json() : []));
  conversations = rows.map((row) => ({ ...row, messages: null })); // transcripts load on demand
  currentId = null;
  renderSidebar();
  renderMessages();
}

function current() {
  return conversations.find((c) => c.id === currentId);
}

function newConversation() {
  const conversation = {
    id: crypto.randomUUID().replaceAll("-", ""),
    title: t("newConversation"),
    updated: Date.now(),
    messages: [],
  };
  conversations.unshift(conversation);
  select(conversation.id);
}

async function select(id) {
  currentId = id;
  if (!serverMode) localStorage.setItem("current_id", id);

  const conversation = current();
  if (serverMode && conversation && conversation.messages === null) {
    const data = await fetch(`/conversations/${id}`).then((r) => (r.ok ? r.json() : null));
    conversation.messages = (data?.messages ?? []).map((m) => ({
      role: m.role === "assistant" ? "bot" : "user",
      text: m.content,
      at: m.created_at ? m.created_at * 1000 : null,
      sources: m.sources,
      usage: m.usage,
    }));
    // totals come back from the per-message usage the server kept
    conversation.tokens = conversation.messages.reduce(
      (sum, m) => sum + (m.usage ? m.usage.prompt_tokens + m.usage.completion_tokens : 0), 0);
    conversation.cost = conversation.messages.reduce(
      (sum, m) => sum + (m.usage?.cost || 0), 0);
  }

  renderSidebar();
  renderMessages();
  closeSidebar();
  input.focus();
}

function removeConversation(id) {
  if (serverMode) fetch(`/conversations/${id}`, { method: "DELETE" });
  conversations = conversations.filter((c) => c.id !== id);
  saveConversations();
  if (currentId === id) {
    currentId = conversations[0]?.id ?? null;
    if (!serverMode) localStorage.setItem("current_id", currentId ?? "");
    renderMessages();
  }
  renderSidebar();
}

/* ---------- rendering ---------- */

function dateGroup(timestamp) {
  const day = 24 * 60 * 60 * 1000;
  const startOfToday = new Date().setHours(0, 0, 0, 0);
  if (timestamp >= startOfToday) return "today";
  if (timestamp >= startOfToday - day) return "yesterday";
  if (timestamp >= startOfToday - 7 * day) return "previous7";
  return "older";
}

function renderSidebar() {
  listEl.replaceChildren();
  // newest first, grouped under ChatGPT-style date headers
  const ordered = [...conversations].sort((a, b) => (b.updated || 0) - (a.updated || 0));
  let lastGroup = null;
  for (const conversation of ordered) {
    const group = dateGroup(conversation.updated || 0);
    if (group !== lastGroup) {
      const header = document.createElement("div");
      header.className = "conversation-group";
      header.textContent = t(group);
      listEl.appendChild(header);
      lastGroup = group;
    }
    const item = document.createElement("div");
    item.className = "conversation-item" + (conversation.id === currentId ? " active" : "");

    const title = document.createElement("span");
    title.className = "title";
    title.textContent = conversation.title;
    item.appendChild(title);

    const del = document.createElement("button");
    del.className = "delete";
    del.textContent = "×";
    del.ariaLabel = "Konuşmayı sil";
    del.addEventListener("click", (event) => {
      event.stopPropagation();
      removeConversation(conversation.id);
    });
    item.appendChild(del);

    item.addEventListener("click", () => select(conversation.id));
    listEl.appendChild(item);
  }
}

/* context size at a glance under an answer: prompt (system + history +
 * retrieved chunks) + completion tokens, with the charged cost when known */
function addUsageChip(botEl, usage) {
  const row = botEl.nextElementSibling; // .msg-actions
  const chip = document.createElement("span");
  chip.className = "token-usage";
  chip.title = t("tokenUsageHint");
  chip.textContent = `${formatTokens(usage.prompt_tokens)} + ${formatTokens(usage.completion_tokens)} token`
    + (usage.cost ? ` · ${formatCost(usage.cost)}` : "");
  row.appendChild(chip);
}

/* running token total of the current conversation, floating over the
 * send button; hidden until the first answer brings a count */
function renderTokenTotal() {
  const conversation = current();
  const total = conversation?.tokens;
  const el = document.getElementById("token-total");
  el.hidden = !total;
  if (!total) return;
  // tokens and cost are running totals. The % is the latest request's
  // prompt vs the window — the only true "how full is the context", since
  // past turns no longer occupy it (history is trimmed + rebuilt) — but
  // "% of what?" confuses non-developers, so it rides the dev switch
  let latest = null;
  if (devMode) {
    for (const message of conversation.messages || []) {
      if (message.usage) latest = message.usage;
    }
  }
  el.textContent = `${formatTokens(total)} token`
    + (conversation.cost ? ` · ${formatCost(conversation.cost)}` : "")
    + (latest ? formatFill(latest.prompt_tokens) : "");
}

function renderMessages() {
  messagesEl.replaceChildren();
  renderTokenTotal();
  const conversation = current();
  if (!conversation || !conversation.messages || conversation.messages.length === 0) {
    messagesEl.appendChild(emptyState);
    return;
  }
  for (const message of conversation.messages) {
    const el = addBubble(message.role, message.text);
    if (message.role === "bot") {
      setBubbleText(el, "bot", message.text, message.sources);
      addBotActions(el, message.text, message);
      if (devMode && message.usage) addUsageChip(el, message.usage);
    }
  }
  // jump straight to the end — smooth scrolling is for streaming only
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: "instant" });
}

/* Minimal markdown for bot answers: HTML is escaped first, then bold,
 * italic, inline code and list markers are converted — safe against
 * anything the model (or the data) emits. Newlines survive via the
 * bubble's pre-wrap. */
function renderMarkdown(text) {
  return text
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/^#{1,4} (.+)$/gm, "<strong>$1</strong>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|\s)\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/^[-*] /gm, "• ");
}

function escapeAttr(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;");
}

function chipHTML(source) {
  // the document's own title tells sources apart; hostname is the legacy
  // fallback for messages stored before titles existed
  let host = "";
  if (source.url) {
    try { host = new URL(source.url).hostname.replace(/^www\./, ""); } catch {}
  }
  const label = escapeAttr(source.title || host || source.type);
  // the hover card reads these; the excerpt is the cited chunk's text
  const data = ` data-host="${escapeAttr(host)}" data-title="${escapeAttr(source.title || source.type)}"` +
               ` data-text="${escapeAttr(source.label || "")}"`;
  if (!source.url) return `<span class="source-chip"${data}>${label}</span>`;
  // URLs from the scraper are already percent-encoded — re-encoding 404s
  // them; only neutralize characters that could break the attribute
  const href = source.url.replaceAll('"', "%22").replaceAll("<", "%3C");
  return `<a class="source-chip" href="${href}" target="_blank" rel="noopener"${data}>${label}</a>`;
}

/* ---------- citation hover card ----------
 * One shared card, repositioned to whichever chip is hovered. */
const chipCard = document.createElement("div");
chipCard.id = "chip-card";
chipCard.hidden = true;
document.body.appendChild(chipCard);

const GLOBE_ICON = '<svg viewBox="0 0 24 24" width="14" height="14"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M3 12h18M12 3c2.5 2.4 4 5.5 4 9s-1.5 6.6-4 9c-2.5-2.4-4-5.5-4-9s1.5-6.6 4-9z" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>';

function showChipCard(chip) {
  chipCard.replaceChildren();
  if (chip.dataset.host) {
    const host = document.createElement("div");
    host.className = "chip-card-host";
    host.innerHTML = `${GLOBE_ICON}<span></span>`;
    host.querySelector("span").textContent = chip.dataset.host;
    chipCard.appendChild(host);
  }
  const title = document.createElement("div");
  title.className = "chip-card-title";
  title.textContent = chip.dataset.title;
  chipCard.appendChild(title);
  if (chip.dataset.text) {
    const text = document.createElement("div");
    text.className = "chip-card-text";
    text.textContent = chip.dataset.text;
    chipCard.appendChild(text);
  }
  chipCard.hidden = false;
  const rect = chip.getBoundingClientRect();
  const card = chipCard.getBoundingClientRect();
  let left = Math.min(rect.left, window.innerWidth - card.width - 12);
  let top = rect.bottom + 8;
  if (top + card.height > window.innerHeight - 8) top = rect.top - card.height - 8;
  chipCard.style.left = `${Math.max(8, left)}px`;
  chipCard.style.top = `${Math.max(8, top)}px`;
}

messagesEl.addEventListener("mouseover", (event) => {
  const chip = event.target.closest(".source-chip");
  if (chip?.dataset.title) showChipCard(chip);
});
messagesEl.addEventListener("mouseout", (event) => {
  if (event.target.closest(".source-chip")) chipCard.hidden = true;
});

/* dev-mode logs float over the terminal icon the same way */
const debugCard = document.createElement("pre");
debugCard.id = "debug-card";
debugCard.hidden = true;
document.body.appendChild(debugCard);

/* "8.4k" reads better than "8412" in a chip this small */
function formatTokens(count) {
  return count >= 1000 ? `${(count / 1000).toFixed(1)}k` : `${count}`;
}

/* single answers cost fractions of a cent — keep the sub-cent digits */
function formatCost(usd) {
  return `$${usd.toFixed(usd >= 0.1 ? 2 : 4)}`;
}

/* context fill: what share of the model's window the prompt used */
function formatFill(promptTokens) {
  if (!appConfig.context_window) return "";
  const pct = (promptTokens / appConfig.context_window) * 100;
  return ` · ${pct < 10 ? pct.toFixed(1) : Math.round(pct)}%`;
}

function showDebugCard(anchor, text) {
  debugCard.textContent = text;
  debugCard.hidden = false;
  const rect = anchor.getBoundingClientRect();
  const card = debugCard.getBoundingClientRect();
  const left = Math.max(8, Math.min(rect.right - card.width, window.innerWidth - card.width - 8));
  let top = rect.top - card.height - 8; // above the icon by default
  if (top < 8) top = rect.bottom + 8;
  debugCard.style.left = `${left}px`;
  debugCard.style.top = `${top}px`;
}

/* the model cites reference chunks as [n]; swap each marker for a chip
 * right where it stands in the sentence. The model tends to repeat the
 * same citation on every line it touches — one chip per document (its
 * first mention) is enough, later repeats are dropped as clutter. */
function withCitations(html, sources) {
  if (!sources?.length) return html;
  const seen = new Set();
  return html.replace(/( ?)\[(\d+)\]/g, (marker, space, n) => {
    const source = sources[Number(n) - 1];
    if (!source) return marker;
    // identity is label + link: calendar chips all say "Akademik takvim"
    // but different years link different PDFs — those are NOT duplicates
    const key = `${source.title || source.type}|${source.url || ""}`;
    if (seen.has(key)) return "";
    seen.add(key);
    return space + chipHTML(source);
  });
}

function setBubbleText(el, role, text, sources) {
  if (role === "bot") el.innerHTML = withCitations(renderMarkdown(text), sources);
  else el.textContent = text;
}

function addBubble(role, text) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  setBubbleText(el, role, text);
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

const COPY_ICON = '<svg viewBox="0 0 24 24" width="15" height="15"><rect x="9" y="9" width="11" height="11" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
const CHECK_ICON = '<svg viewBox="0 0 24 24" width="15" height="15"><path d="M5 13l4 4L19 7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const MORE_ICON = '<svg viewBox="0 0 24 24" width="15" height="15"><circle cx="5" cy="12" r="1.6" fill="currentColor"/><circle cx="12" cy="12" r="1.6" fill="currentColor"/><circle cx="19" cy="12" r="1.6" fill="currentColor"/></svg>';
const BOOK_ICON = '<svg viewBox="0 0 24 24" width="16" height="16"><path d="M12 5c-2-1.5-5-2-8-2v16c3 0 6 .5 8 2 2-1.5 5-2 8-2V3c-3 0-6 .5-8 2z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 5v16" stroke="currentColor" stroke-width="2"/></svg>';
const TERMINAL_ICON = '<svg viewBox="0 0 24 24" width="15" height="15"><path d="M4 17l6-5-6-5M12 19h8" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

const msgPopover = document.getElementById("msg-popover");

function formatTimestamp(at) {
  if (!at) return "";
  const date = new Date(at);
  const time = date.toLocaleTimeString(lang === "tr" ? "tr-TR" : "en-US",
    { hour: "2-digit", minute: "2-digit" });
  const isToday = date.toDateString() === new Date().toDateString();
  if (isToday) return `${t("today")}, ${time}`;
  return `${date.toLocaleDateString(lang === "tr" ? "tr-TR" : "en-US",
    { day: "numeric", month: "long" })}, ${time}`;
}

function openMsgPopover(anchor, message) {
  msgPopover.replaceChildren();

  const header = document.createElement("div");
  header.className = "popover-time";
  header.textContent = formatTimestamp(message.at);
  msgPopover.appendChild(header);

  const sourcesItem = document.createElement("button");
  sourcesItem.className = "popover-item";
  sourcesItem.innerHTML = `${BOOK_ICON}<span>${t("viewSources")}</span>`;
  sourcesItem.addEventListener("click", (event) => {
    event.stopPropagation();
    showSources(message);
  });
  msgPopover.appendChild(sourcesItem);

  msgPopover.hidden = false;
  const rect = anchor.getBoundingClientRect();
  msgPopover.style.left = `${Math.min(rect.left, window.innerWidth - 320)}px`;
  msgPopover.style.bottom = `${window.innerHeight - rect.top + 8}px`;
}

function showSources(message) {
  msgPopover.replaceChildren();
  const list = document.createElement("div");
  list.className = "popover-sources";
  const sources = message.sources || [];
  // retrieval hands the model ~20 chunks; list only the ones the answer
  // actually cites. Messages without inline markers (old transcripts,
  // pre-citation answers) fall back to the full list.
  const cited = new Set(
    [...(message.text || "").matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])));
  const used = cited.size ? sources.filter((_, i) => cited.has(i + 1)) : sources;
  if (!used.length) {
    list.textContent = t("noSources");
  }
  for (const source of used) {
    const item = document.createElement(source.url ? "a" : "div");
    item.className = "source-item";
    if (source.url) {
      item.href = source.url;
      item.target = "_blank";
      item.rel = "noopener";
    }
    const type = document.createElement("span");
    type.className = "source-type";
    type.textContent = source.type;
    const label = document.createElement("span");
    label.textContent = source.label;
    item.append(type, label);
    list.appendChild(item);
  }
  msgPopover.appendChild(list);
}

document.addEventListener("click", (event) => {
  if (!msgPopover.hidden && !msgPopover.contains(event.target)) msgPopover.hidden = true;
});

function addBotActions(messageEl, text, message = {}) {
  const row = document.createElement("div");
  row.className = "msg-actions";

  const copy = document.createElement("button");
  copy.innerHTML = COPY_ICON;
  copy.ariaLabel = "Kopyala";
  copy.addEventListener("click", async () => {
    await navigator.clipboard.writeText(text);
    copy.innerHTML = CHECK_ICON;
    setTimeout(() => { copy.innerHTML = COPY_ICON; }, 1200);
  });
  row.appendChild(copy);

  const more = document.createElement("button");
  more.innerHTML = MORE_ICON;
  more.ariaLabel = "Daha fazla";
  more.addEventListener("click", (event) => {
    event.stopPropagation();
    openMsgPopover(more, message);
  });
  row.appendChild(more);

  messageEl.after(row);
}

/* ---------- chat ---------- */

// while streaming, the send button becomes a stop button that aborts
// the request; the partial answer stays in the chat
const SEND_ICON = sendButton.innerHTML;
const STOP_ICON = '<svg viewBox="0 0 24 24" width="16" height="16"><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/></svg>';
let activeController = null;

function setStreaming(active) {
  sendButton.innerHTML = active ? STOP_ICON : SEND_ICON;
  sendButton.ariaLabel = active ? "Durdur" : "Gönder";
}

async function send(query, image = null) {
  if (!current()) newConversation();
  const conversation = current();

  if (conversation.messages.length === 0) {
    const titleBase = query || t("imageMessage");
    conversation.title = titleBase.length > 40 ? titleBase.slice(0, 40) + "…" : titleBase;
  }
  conversation.messages.push({ role: "user", text: query, at: Date.now() });
  conversation.updated = Date.now();
  saveConversations();
  renderSidebar();

  if (emptyState.parentNode) emptyState.remove();
  const userEl = addBubble("user", query);
  if (image) {
    const thumb = document.createElement("img");
    thumb.className = "msg-image";
    thumb.src = image;
    userEl.prepend(thumb);
  }
  const botEl = addBubble("bot", "");
  botEl.classList.add("pending", "rewriting");
  botEl.dataset.stage = t("stageRewriting"); // shimmer label next to the cursor
  input.disabled = true;
  setStreaming(true); // send button stays clickable, now as stop
  activeController = new AbortController();

  let answer = "";
  try {
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, session_id: conversation.id, image, lang,
                             model_name: chatModel || null }),
      signal: activeController.signal,
    });
    if (response.status === 429) {
      const body = await response.json().catch(() => null);
      botEl.remove(); // the dialog carries the message instead of a bubble
      if (response.headers.get("X-Block-Reason") === "abuse") showAbuseDialog();
      else showLimitDialog(body?.detail);
      return;
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    // tokens arrive in whatever bursts the provider emits; buffer them in
    // `received` and reveal at a steady rate (ChatGPT-style) so the text
    // flows instead of jumping. The reveal speeds up with the backlog, so
    // it never falls far behind the wire.
    let received = "";
    let readDone = false;
    let readError = null;
    const pump = (async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          received += decoder.decode(value, { stream: true });
          // backend stage markers: \x02 = rewrite done, \x03 = gate
          // passed, \x01 = retrieval done. Cursor: rewriting (tilt) ->
          // searching (scan) -> writing (pulse) -> blink.
          // \u0002 (gate checking) and \u0003 (gate passed) both show as
          // "searching": retrieval genuinely runs through the whole gate
          // window, and surfacing moderation at every turn reads hostile
          if (received.includes("\u0002") || received.includes("\u0003")) {
            received = received.replaceAll("\u0002", "").replaceAll("\u0003", "");
            botEl.classList.remove("rewriting");
            botEl.classList.add("searching");
            botEl.dataset.stage = t("stageSearching");
          }
          if (received.includes("\u0001")) {
            received = received.replaceAll("\u0001", "");
            botEl.classList.remove("rewriting", "searching");
            botEl.classList.add("writing");
            botEl.dataset.stage = t("stageWriting");
          }
        }
      } catch (error) {
        readError = error;
      } finally {
        readDone = true;
      }
    })();

    while ((!readDone || answer.length < received.length) && !readError) {
      await new Promise((resolve) => setTimeout(resolve, 24));
      if (answer.length < received.length) {
        const backlog = received.length - answer.length;
        answer = received.slice(0, answer.length + Math.max(2, Math.round(backlog / 15)));
        botEl.classList.remove("rewriting", "searching", "writing");
        setBubbleText(botEl, "bot", answer);
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
    }
    await pump;
    if (readError) throw readError;
    answer = received;
    if (answer === appConfig.abuse_message) {
      botEl.remove(); // no bot reply in the chat — the popup is the response
      showAbuseDialog();
      return;
    }
    let sources = [];
    let usage = null;
    if (answer !== appConfig.off_topic_message) {
      const params = new URLSearchParams({ session_id: conversation.id });
      const data = await fetch(`/chat/sources?${params}`)
        .then((r) => (r.ok ? r.json() : {}))
        .catch(() => ({}));
      sources = data.sources ?? [];
      usage = data.usage ?? null;
      if (usage) {
        conversation.tokens = (conversation.tokens || 0)
          + usage.prompt_tokens + usage.completion_tokens;
        if (usage.cost) conversation.cost = (conversation.cost || 0) + usage.cost;
      }
    }
    const botMessage = { role: "bot", text: answer, at: Date.now(), sources, usage };
    conversation.messages.push(botMessage);
    conversation.updated = Date.now();
    saveConversations();
    if (usage) renderTokenTotal();
    setBubbleText(botEl, "bot", answer, sources); // final pass turns [n] into chips
    addBotActions(botEl, answer, botMessage);
    if (devMode) {
      if (usage) addUsageChip(botEl, usage);
      const params = new URLSearchParams({ session_id: conversation.id });
      const { debug } = await fetch(`/chat/debug?${params}`)
        .then((r) => (r.ok ? r.json() : { debug: [] }))
        .catch(() => ({ debug: [] }));
      if (debug?.length) {
        // a terminal icon at the right end of the actions row; this
        // answer's server logs float over it on hover
        const row = botEl.nextElementSibling; // .msg-actions
        const toggle = document.createElement("button");
        toggle.className = "debug-toggle";
        toggle.innerHTML = TERMINAL_ICON;
        toggle.ariaLabel = "Debug";
        const logText = debug.join("\n");
        toggle.addEventListener("mouseenter", () => showDebugCard(toggle, logText));
        toggle.addEventListener("mouseleave", () => { debugCard.hidden = true; });
        row.appendChild(toggle);
      }
    }
  } catch (error) {
    if (error.name === "AbortError") {
      // user hit stop — keep whatever streamed, drop an empty bubble
      if (answer) {
        const botMessage = { role: "bot", text: answer, at: Date.now(), sources: [] };
        conversation.messages.push(botMessage);
        conversation.updated = Date.now();
        saveConversations();
        setBubbleText(botEl, "bot", answer);
        addBotActions(botEl, answer, botMessage);
      } else {
        botEl.remove();
      }
    } else {
      botEl.textContent = t("error", error.message);
    }
  } finally {
    activeController = null;
    setStreaming(false);
    botEl.classList.remove("pending");
    input.disabled = sendButton.disabled = abuseBlocked;
    if (!abuseBlocked) input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (activeController) {
    activeController.abort(); // the button is a stop button mid-stream
    return;
  }
  const query = input.value.trim();
  if ((!query && !pendingImage) || input.disabled) return; // image alone is enough
  input.value = "";
  resizeInput();
  const image = pendingImage;
  clearAttachment();
  send(query, image);
});

/* the chatbar is a textarea: Enter sends, Shift+Enter breaks the line,
 * and the pill grows with the text (up to #input's max-height) */
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
}

input.addEventListener("input", resizeInput);

/* ---------- image attachment ----------
 * The + button picks an image; it is downscaled client-side (long edge
 * 1568px, JPEG) so the payload stays small, previewed above the composer,
 * and sent with the next message for that turn only. */
const fileInput = document.getElementById("file-input");
const attachmentPreview = document.getElementById("attachment-preview");
const attachmentThumb = document.getElementById("attachment-thumb");
let pendingImage = null;

function clearAttachment() {
  pendingImage = null;
  attachmentThumb.src = "";
  attachmentPreview.hidden = true;
}

async function downscaleImage(file) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, 1568 / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.85);
}

async function attachImage(file) {
  try {
    pendingImage = await downscaleImage(file);
  } catch {
    flashPlaceholder(t("imageError"));
    return;
  }
  attachmentThumb.src = pendingImage;
  attachmentPreview.hidden = false;
  input.focus();
}

document.getElementById("attach").addEventListener("click", () => fileInput.click());
document.getElementById("attachment-remove").addEventListener("click", clearAttachment);

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileInput.value = ""; // re-selecting the same file must fire change again
  if (file) attachImage(file);
});

// pasting an image (screenshot in the clipboard) attaches it too
input.addEventListener("paste", (event) => {
  const item = [...event.clipboardData.items].find((i) => i.type.startsWith("image/"));
  if (!item) return;
  event.preventDefault();
  attachImage(item.getAsFile());
});

/* ---------- voice input ----------
 * The mic swaps the composer for a live waveform strip with cancel (×)
 * and confirm (✓), ChatGPT-style. Confirm posts the blob to /transcribe
 * (Groq Whisper); the text lands in the input for review — it is never
 * sent automatically. Button shows only when the server has a key. */
const micButton = document.getElementById("mic");
const composerEl = document.getElementById("composer");
const recorderEl = document.getElementById("recorder");
const waveCanvas = document.getElementById("waveform");
let recorder = null;
let audioCtx = null;
let waveTimer = null;
let waveRAF = null;
let waveLevels = [];
let lastSampleAt = 0;
let recordingCancelled = false;
const WAVE_SAMPLE_MS = 100;

function flashPlaceholder(text) {
  input.placeholder = text;
  setTimeout(() => { input.placeholder = t("placeholder"); }, 3000);
}

function setRecordingUI(active) {
  composerEl.classList.toggle("recording", active);
  recorderEl.hidden = !active;
}

function renderWave() {
  const dpr = window.devicePixelRatio || 1;
  const width = (waveCanvas.width = waveCanvas.offsetWidth * dpr);
  const height = (waveCanvas.height = waveCanvas.offsetHeight * dpr);
  const ctx = waveCanvas.getContext("2d");
  const step = 5 * dpr; // bar + gap
  // frac is the sub-sample progress: bars glide left between samples
  // instead of jumping one slot per tick, and the newest eases in.
  const frac = Math.min((performance.now() - lastSampleAt) / WAVE_SAMPLE_MS, 1);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = getComputedStyle(document.body).color;
  for (let back = 0; back <= Math.ceil(width / step); back++) { // back = bars from newest
    const index = waveLevels.length - 1 - back;
    const level = index >= 0 ? waveLevels[index] : 0; // idle slots render as dots
    const x = width - (back + frac) * step;
    if (x < -step) break;
    let bar = Math.min(Math.max(2 * dpr, level * height * 3), height * 0.9);
    if (back === 0) bar = Math.max(2 * dpr, bar * frac);
    ctx.beginPath();
    ctx.roundRect(x, (height - bar) / 2, 2.5 * dpr, bar, 2 * dpr);
    ctx.fill();
  }
}

function startWaveform(stream) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  audioCtx = new AudioCtx();
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  audioCtx.createMediaStreamSource(stream).connect(analyser);
  const data = new Uint8Array(analyser.fftSize);
  waveLevels = [];
  lastSampleAt = performance.now();
  waveTimer = setInterval(() => {
    analyser.getByteTimeDomainData(data);
    let sum = 0;
    for (const value of data) {
      const deviation = (value - 128) / 128;
      sum += deviation * deviation;
    }
    const rms = Math.sqrt(sum / data.length);
    // blend with the previous sample so single spikes don't jag
    const previous = waveLevels.length ? waveLevels[waveLevels.length - 1] : rms;
    waveLevels.push(previous * 0.3 + rms * 0.7);
    lastSampleAt = performance.now();
  }, WAVE_SAMPLE_MS);
  const loop = () => {
    renderWave();
    waveRAF = requestAnimationFrame(loop);
  };
  waveRAF = requestAnimationFrame(loop);
}

function stopWaveform() {
  clearInterval(waveTimer);
  cancelAnimationFrame(waveRAF);
  if (audioCtx) audioCtx.close();
  audioCtx = null;
}

async function transcribe(blob, mimeType) {
  micButton.disabled = true;
  input.placeholder = "…";
  try {
    const response = await fetch("/transcribe", {
      method: "POST",
      headers: { "Content-Type": mimeType || "audio/webm" },
      body: blob,
    });
    if (response.status === 429) {
      const body = await response.json().catch(() => null);
      if (response.headers.get("X-Block-Reason") === "abuse") showAbuseDialog();
      else flashPlaceholder(body?.detail || t("micError"));
      return;
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const { text } = await response.json();
    if (text) {
      input.value = input.value ? `${input.value} ${text}` : text;
      resizeInput();
      input.focus();
    }
  } catch {
    flashPlaceholder(t("micError"));
  } finally {
    micButton.disabled = false;
    input.placeholder = t("placeholder");
  }
}

async function startRecording() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    flashPlaceholder(t("micDenied"));
    return;
  }
  recordingCancelled = false;
  const chunks = [];
  recorder = new MediaRecorder(stream); // browser picks the codec (webm/mp4)
  recorder.ondataavailable = (event) => chunks.push(event.data);
  recorder.onstop = () => {
    stream.getTracks().forEach((track) => track.stop());
    stopWaveform();
    setRecordingUI(false);
    if (recordingCancelled) return;
    transcribe(new Blob(chunks, { type: recorder.mimeType }), recorder.mimeType);
  };
  recorder.start();
  startWaveform(stream);
  setRecordingUI(true);
}

micButton.addEventListener("click", startRecording);
document.getElementById("rec-cancel").addEventListener("click", () => {
  recordingCancelled = true;
  recorder.stop();
});
document.getElementById("rec-done").addEventListener("click", () => recorder.stop());

document.getElementById("new-conversation").addEventListener("click", () => {
  const conversation = current();
  if (conversation && conversation.messages.length === 0) return select(conversation.id);
  newConversation();
});

/* ---------- theme ---------- */

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("theme", theme);
  renderSigninButton(); // the Google button carries its own theme
}

document.getElementById("theme-toggle").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme === "dark";
  applyTheme(dark ? "light" : "dark");
});

applyTheme(
  localStorage.getItem("theme") ||
  (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
);

/* ---------- auth ----------
 * Optional Google Sign-In: /auth/config says whether it's configured.
 * The Google button posts an ID token to /auth/google, which sets an
 * HttpOnly session cookie; /auth/me restores the user on reload. */

const authEl = document.getElementById("auth");
const signinEl = document.getElementById("signin-button");
const chipEl = document.getElementById("user-chip");

const profileDialog = document.getElementById("profile-dialog");
const profileForm = document.getElementById("profile-form");
const limitDialog = document.getElementById("limit-dialog");
const abuseDialog = document.getElementById("abuse-dialog");

let appConfig = { client_id: null, abuse_message: null };
let signedIn = false;
let currentUser = null;

document.querySelectorAll("dialog [data-close]").forEach((button) =>
  button.addEventListener("click", () => button.closest("dialog").close()));

function showLimitDialog(detail) {
  document.getElementById("limit-text").textContent = detail || t("limitFallback");
  const signinSlot = document.getElementById("limit-signin");
  signinSlot.replaceChildren();
  // anonymous users get a sign-in button right inside the dialog
  if (!signedIn && appConfig.client_id && window.google?.accounts?.id) {
    google.accounts.id.renderButton(signinSlot, googleButtonOptions(240));
  }
  limitDialog.showModal();
}

let abuseBlocked = false;

function showAbuseDialog() {
  if (appConfig.abuse_exempt) return; // dev bypass (ABUSE_EXEMPT)
  // remember the block across reloads; the server enforces it regardless
  const until = Number(localStorage.getItem("abuse_block_until")) || 0;
  if (until < Date.now()) {
    localStorage.setItem("abuse_block_until",
      Date.now() + (appConfig.abuse_block_seconds || 600) * 1000);
  }
  document.getElementById("abuse-text").textContent = appConfig.abuse_message;
  abuseBlocked = true;
  input.disabled = sendButton.disabled = true;
  if (!abuseDialog.open) abuseDialog.showModal();
  scheduleAbuseUnlock();
}

function scheduleAbuseUnlock() {
  const remaining = Number(localStorage.getItem("abuse_block_until")) - Date.now();
  if (remaining > 0) setTimeout(liftAbuseBlock, remaining);
}

function liftAbuseBlock() {
  localStorage.removeItem("abuse_block_until");
  abuseBlocked = false;
  input.disabled = sendButton.disabled = false;
  if (abuseDialog.open) abuseDialog.close();
  input.focus();
}

function checkAbuseBlock() {
  // re-lock on page load while the block window is still running
  if (Number(localStorage.getItem("abuse_block_until")) > Date.now()) showAbuseDialog();
}

// the abuse dialog cannot be dismissed — swallow Escape
abuseDialog.addEventListener("cancel", (event) => event.preventDefault());

function showUser(user) {
  signedIn = true;
  currentUser = user;
  signinEl.hidden = true;
  chipEl.hidden = false;
  document.getElementById("user-name").textContent = user.name || user.email;
  document.getElementById("user-picture").src = user.picture || "";
  document.getElementById("user-badge").hidden = !user.member;
  if (!user.education_type) openProfileDialog();
}

// forced once at first login; day-to-day changes live in the settings dialog
function openProfileDialog() {
  profileForm.querySelectorAll("input[name=education_type]").forEach((radio) => {
    radio.checked = radio.value === currentUser?.education_type;
  });
  profileDialog.showModal();
}

async function saveEducationType(education_type) {
  if (currentUser) currentUser.education_type = education_type;
  await fetch("/auth/profile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ education_type }),
  });
}

profileForm.addEventListener("submit", () => {
  saveEducationType(new FormData(profileForm).get("education_type"));
});

/* ---------- settings dialog ----------
 * ChatGPT-style: section tabs on the left, label/control rows on the
 * right. Every control applies immediately — there is no save button. */
const settingsDialog = document.getElementById("settings-dialog");
let devMode = localStorage.getItem("devmode") === "1";
// empty = let the server pick its default model
let chatModel = localStorage.getItem("chat_model") || "";

function openSettings() {
  document.getElementById("set-lang").value = lang;
  document.getElementById("set-theme").value = document.documentElement.dataset.theme;
  const modelSelect = document.getElementById("set-model");
  modelSelect.value = chatModel || modelSelect.options[0].value;
  document.getElementById("devmode-toggle").checked = devMode;
  settingsDialog.querySelectorAll("input[name=settings_education]").forEach((radio) => {
    radio.checked = radio.value === currentUser?.education_type;
  });
  settingsDialog.showModal();
}

settingsDialog.querySelectorAll(".settings-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    settingsDialog.querySelectorAll(".settings-tab").forEach((other) =>
      other.classList.toggle("active", other === tab));
    settingsDialog.querySelectorAll(".settings-panel").forEach((panel) => {
      panel.hidden = panel.dataset.panel !== tab.dataset.tab;
    });
    const title = document.getElementById("settings-title");
    title.dataset.i18n = tab.querySelector("span").dataset.i18n;
    title.textContent = t(title.dataset.i18n);
  });
});

document.getElementById("set-lang").addEventListener("change", (event) => {
  lang = event.target.value;
  localStorage.setItem("lang", lang);
  applyLanguage();
  renderSidebar();
  setupGoogleButton();
});

document.getElementById("set-theme").addEventListener("change", (event) => {
  applyTheme(event.target.value);
});

document.getElementById("devmode-toggle").addEventListener("change", (event) => {
  devMode = event.target.checked;
  localStorage.setItem("devmode", devMode ? "1" : "0");
  renderMessages(); // usage chips + token counter follow the switch
});

document.getElementById("set-model").addEventListener("change", (event) => {
  chatModel = event.target.value;
  localStorage.setItem("chat_model", chatModel);
});

settingsDialog.querySelectorAll("input[name=settings_education]").forEach((radio) => {
  radio.addEventListener("change", () => saveEducationType(radio.value));
});

document.querySelector(".user-info").addEventListener("click", openSettings);

async function adoptCurrentConversation() {
  // hand the active anonymous chat to the account so it continues seamlessly
  const conversation = current();
  if (!conversation?.messages?.length) return null;

  const result = await fetch("/conversations/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      id: conversation.id,
      messages: conversation.messages.map((m) => ({ role: m.role, text: m.text })),
    }),
  });
  if (!result.ok) return null;

  // it lives on the server now — drop the localStorage copy
  conversations = conversations.filter((c) => c.id !== conversation.id);
  localStorage.setItem("conversations", JSON.stringify(conversations));
  return conversation.id;
}

async function onGoogleCredential(response) {
  const result = await fetch("/auth/google", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ credential: response.credential }),
  });
  if (result.ok) {
    if (limitDialog.open) limitDialog.close();
    showUser(await result.json());
    const adoptedId = await adoptCurrentConversation();
    await enterServerMode();
    if (adoptedId) select(adoptedId);
  }
}

let gisScript = null;

function loadGoogleScript() {
  // the button iframe takes its language from the script's ?hl= param —
  // the locale render option alone doesn't switch it
  return new Promise((resolve) => {
    if (gisScript) gisScript.remove();
    gisScript = document.createElement("script");
    gisScript.src = `https://accounts.google.com/gsi/client?hl=${lang}`;
    gisScript.async = true;
    gisScript.onload = resolve;
    document.head.appendChild(gisScript);
  });
}

async function setupGoogleButton() {
  if (!appConfig.client_id || signedIn) return;
  await loadGoogleScript();
  google.accounts.id.initialize({
    client_id: appConfig.client_id,
    callback: onGoogleCredential,
  });
  gisReady = true;
  renderSigninButton();
}

function googleButtonOptions(width) {
  return {
    theme: document.documentElement.dataset.theme === "dark" ? "filled_black" : "outline",
    shape: "pill",
    size: "large",
    text: "signin_with",
    locale: lang === "tr" ? "tr" : "en",
    width,
  };
}

function renderSigninButton() {
  if (!gisReady || signedIn) return;
  signinEl.replaceChildren(); // re-render must not stack buttons
  google.accounts.id.renderButton(signinEl, googleButtonOptions(232));
}

async function initAuth() {
  appConfig = await fetch("/auth/config").then((r) => r.json());
  checkAbuseBlock(); // needs the config, so it lives here
  micButton.hidden = !(appConfig.stt && navigator.mediaDevices);
  if (!appConfig.client_id) return; // sign-in not configured — keep it hidden

  authEl.hidden = false;
  const user = await fetch("/auth/me").then((r) => r.json());
  if (user) {
    showUser(user);
    return enterServerMode();
  }

  setupGoogleButton();
}

document.getElementById("logout").addEventListener("click", async () => {
  await fetch("/auth/logout", { method: "POST" });
  location.reload(); // clean reset back to anonymous/localStorage mode
});

initAuth();

/* ---------- sidebar toggle ----------
 * Mobile: off-canvas drawer with overlay. Desktop: collapses like
 * ChatGPT's sidebar, remembered across visits. */

function isMobile() {
  return window.matchMedia("(max-width: 768px)").matches;
}

function closeSidebar() {
  document.body.classList.remove("sidebar-open");
}

document.getElementById("menu").addEventListener("click", () => {
  if (isMobile()) {
    document.body.classList.toggle("sidebar-open");
  } else {
    const collapsed = document.body.classList.toggle("sidebar-collapsed");
    localStorage.setItem("sidebar_collapsed", collapsed ? "1" : "");
  }
});

document.getElementById("overlay").addEventListener("click", closeSidebar);

document.getElementById("lang-toggle").addEventListener("click", () => {
  lang = lang === "tr" ? "en" : "tr";
  localStorage.setItem("lang", lang);
  applyLanguage();
  renderSidebar(); // date group headers are rendered text, not data-i18n
  setupGoogleButton(); // reload the Google SDK with the new ?hl= locale
});

/* ---------- init ---------- */

if (localStorage.getItem("sidebar_collapsed") === "1" && !isMobile()) {
  document.body.classList.add("sidebar-collapsed");
}
applyLanguage();
renderSidebar();
renderMessages();
input.focus();
