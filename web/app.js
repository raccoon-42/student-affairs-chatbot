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
    title: "Yeni konuşma",
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
    addBubble(message.role, message.text);
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
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

function setBubbleText(el, role, text) {
  if (role === "bot") el.innerHTML = renderMarkdown(text);
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

/* ---------- chat ---------- */

async function send(query) {
  if (!current()) newConversation();
  const conversation = current();

  if (conversation.messages.length === 0) {
    conversation.title = query.length > 40 ? query.slice(0, 40) + "…" : query;
  }
  conversation.messages.push({ role: "user", text: query });
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
    conversation.messages.push({ role: "bot", text: answer });
    conversation.updated = Date.now();
    saveConversations();
  } catch (error) {
    botEl.textContent = `Bir şeyler ters gitti (${error.message}). Tekrar dener misin?`;
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
  document.getElementById("limit-text").textContent =
    detail || "Mesaj limitine ulaştın. Lütfen bir süre sonra tekrar dene.";
  const signinSlot = document.getElementById("limit-signin");
  signinSlot.replaceChildren();
  // anonymous users get a sign-in button right inside the dialog
  if (!signedIn && appConfig.client_id && window.google?.accounts?.id) {
    google.accounts.id.renderButton(signinSlot, {
      theme: document.documentElement.dataset.theme === "dark" ? "filled_black" : "outline",
      size: "large",
      text: "signin_with",
    });
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

function whenGoogleReady(callback, attempts = 50) {
  if (window.google?.accounts?.id) return callback();
  if (attempts > 0) setTimeout(() => whenGoogleReady(callback, attempts - 1), 100);
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

  whenGoogleReady(() => {
    google.accounts.id.initialize({
      client_id: appConfig.client_id,
      callback: onGoogleCredential,
    });
    signinEl.replaceChildren(); // re-init after logout must not stack buttons
    google.accounts.id.renderButton(signinEl, {
      theme: document.documentElement.dataset.theme === "dark" ? "filled_black" : "outline",
      size: "medium",
      text: "signin_with",
      width: 230,
    });
  });
}

document.getElementById("logout").addEventListener("click", async () => {
  await fetch("/auth/logout", { method: "POST" });
  location.reload(); // clean reset back to anonymous/localStorage mode
});

initAuth();

/* ---------- mobile sidebar ---------- */

function closeSidebar() {
  document.body.classList.remove("sidebar-open");
}

document.getElementById("menu").addEventListener("click", () => {
  document.body.classList.toggle("sidebar-open");
});

document.getElementById("overlay").addEventListener("click", closeSidebar);

/* ---------- init ---------- */

renderSidebar();
renderMessages();
input.focus();
