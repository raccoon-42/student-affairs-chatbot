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
    }));
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

function renderSidebar() {
  listEl.replaceChildren();
  for (const conversation of conversations) {
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

function renderMessages() {
  messagesEl.replaceChildren();
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

function chipHTML(source) {
  let label = source.type;
  if (source.url) {
    try { label = new URL(source.url).hostname.replace(/^www\./, ""); } catch {}
  }
  if (!source.url) return `<span class="source-chip">${label}</span>`;
  // URLs from the scraper are already percent-encoded — re-encoding 404s
  // them; only neutralize characters that could break the attribute
  const href = source.url.replaceAll('"', "%22").replaceAll("<", "%3C");
  return `<a class="source-chip" href="${href}" target="_blank" rel="noopener">${label}</a>`;
}

/* the model cites reference chunks as [n]; swap each marker for a chip
 * right where it stands in the sentence */
function withCitations(html, sources) {
  if (!sources?.length) return html;
  return html.replace(/\[(\d+)\]/g, (marker, n) => {
    const source = sources[Number(n) - 1];
    return source ? chipHTML(source) : marker;
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
  if (!sources.length) {
    list.textContent = t("noSources");
  }
  for (const source of sources) {
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

async function send(query) {
  if (!current()) newConversation();
  const conversation = current();

  if (conversation.messages.length === 0) {
    conversation.title = query.length > 40 ? query.slice(0, 40) + "…" : query;
  }
  conversation.messages.push({ role: "user", text: query, at: Date.now() });
  conversation.updated = Date.now();
  saveConversations();
  renderSidebar();

  if (emptyState.parentNode) emptyState.remove();
  addBubble("user", query);
  const botEl = addBubble("bot", "");
  botEl.classList.add("pending");
  input.disabled = sendButton.disabled = true;

  try {
    const params = new URLSearchParams({ query, session_id: conversation.id });
    const response = await fetch(`/chat/stream?${params}`);
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
    let answer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      answer += decoder.decode(value, { stream: true });
      setBubbleText(botEl, "bot", answer);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
    if (answer === appConfig.abuse_message) {
      botEl.remove(); // no bot reply in the chat — the popup is the response
      showAbuseDialog();
      return;
    }
    let sources = [];
    if (answer !== appConfig.off_topic_message) {
      const params = new URLSearchParams({ session_id: conversation.id });
      sources = await fetch(`/chat/sources?${params}`)
        .then((r) => (r.ok ? r.json() : { sources: [] }))
        .then((data) => data.sources)
        .catch(() => []);
    }
    const botMessage = { role: "bot", text: answer, at: Date.now(), sources };
    conversation.messages.push(botMessage);
    conversation.updated = Date.now();
    saveConversations();
    setBubbleText(botEl, "bot", answer, sources); // final pass turns [n] into chips
    addBotActions(botEl, answer, botMessage);
  } catch (error) {
    botEl.textContent = t("error", error.message);
  } finally {
    botEl.classList.remove("pending");
    input.disabled = sendButton.disabled = abuseBlocked;
    if (!abuseBlocked) input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query || input.disabled) return;
  input.value = "";
  send(query);
});

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
  signinEl.hidden = true;
  chipEl.hidden = false;
  document.getElementById("user-name").textContent = user.name || user.email;
  document.getElementById("user-picture").src = user.picture || "";
  document.getElementById("user-badge").hidden = !user.member;
  if (!user.education_type) profileDialog.showModal();
}

profileForm.addEventListener("submit", async () => {
  const education_type = new FormData(profileForm).get("education_type");
  await fetch("/auth/profile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ education_type }),
  });
});

// clicking your own name reopens the dialog to change the answer
document.querySelector(".user-info").addEventListener("click", () => {
  profileDialog.showModal();
});

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
