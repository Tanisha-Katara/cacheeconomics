"use strict";

document.documentElement.classList.add("has-js");

const API_ROOT = "/api";
const PKCE_VERIFIER_KEY = "cacheeconomics.pkce.verifier";
const OAUTH_STATE_KEY = "cacheeconomics.oauth.state";

const state = {
  config: null,
  token: null,
  me: null,
  organizationId: null,
  sourceId: null,
  sources: [],
  health: null,
  analysis: null,
  operations: null,
  jobs: [],
  view: "overview",
  windowHours: 168,
  organizationLoad: 0,
  sourceLoad: 0,
  refreshing: false,
};

const views = {
  overview: { title: "Overview", kicker: "CACHE PERFORMANCE" },
  recommendations: { title: "Recommendations", kicker: "WHAT TO FIX" },
  operations: { title: "Operations", kicker: "REQUEST HEALTH" },
  jobs: { title: "Jobs & health", kicker: "DATA DELIVERY" },
};

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const byId = (id) => document.getElementById(id);

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function svgElement(tag, attributes = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function titleCase(value) {
  return String(value || "unknown")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function compactNumber(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(value)
    : "—";
}

function exactNumber(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? new Intl.NumberFormat().format(value)
    : "—";
}

function percent(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 1 }).format(value)
    : "—";
}

function milliseconds(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value)} ms`
    : "—";
}

function currency(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? new Intl.NumberFormat(undefined, {
      style: "currency",
      currency: "USD",
      currencyDisplay: "narrowSymbol",
      maximumFractionDigits: Math.abs(value) >= 1000 ? 0 : 2,
    }).format(value)
    : "—";
}

function timestamp(value) {
  if (!value) return "Not recorded";
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "Invalid timestamp";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function initial(value) {
  const character = String(value || "?").trim().charAt(0);
  return character ? character.toUpperCase() : "?";
}

function badge(label, tone = "neutral") {
  return element("span", `badge badge-${tone}`, label);
}

function button(label, className, handler) {
  const node = element("button", `button ${className}`, label);
  node.type = "button";
  node.addEventListener("click", handler);
  return node;
}

function showToast(message, kind = "success") {
  const toast = element("div", `toast${kind === "error" ? " error" : ""}`, message);
  byId("toast-region").append(toast);
  window.setTimeout(() => toast.remove(), 4500);
}

function showWorkspaceNotice(message) {
  const notice = byId("workspace-notice");
  notice.textContent = message || "";
  notice.hidden = !message;
}

function statePanel(icon, heading, detail, actionNode = null) {
  const panel = element("div", "state-panel");
  const content = element("div");
  content.append(element("span", "state-icon", icon));
  content.append(element("h2", "", heading));
  content.append(element("p", "", detail));
  if (actionNode) content.append(actionNode);
  panel.append(content);
  return panel;
}

function loadingPanel() {
  const grid = element("div", "skeleton-grid");
  for (let index = 0; index < 4; index += 1) grid.append(element("div", "skeleton"));
  return grid;
}

function setGlobalStatus(text, tone = "waiting") {
  const node = byId("global-status");
  node.className = `global-status${tone === "good" ? " is-good" : tone === "bad" ? " is-bad" : ""}`;
  node.lastElementChild.textContent = text;
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("Accept", "application/json");
  headers.set("Authorization", `Bearer ${state.token}`);
  if (options.organization !== false && state.organizationId) {
    headers.set("X-Organization-ID", state.organizationId);
  }
  if (options.body !== undefined) headers.set("Content-Type", "application/json");
  const response = await fetch(`${API_ROOT}${path}`, {
    method: options.method || "GET",
    headers,
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    credentials: "omit",
    redirect: "error",
  });
  let payload = null;
  if (response.status !== 204) {
    try {
      payload = await response.json();
    } catch (_error) {
      payload = null;
    }
  }
  if (!response.ok) {
    const detail = payload && typeof payload.detail === "string" ? payload.detail : "Request failed";
    throw new ApiError(response.status, detail);
  }
  return payload;
}

function base64Url(bytes) {
  let binary = "";
  bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
  return window.btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

function secureRandom(size) {
  const bytes = new Uint8Array(size);
  window.crypto.getRandomValues(bytes);
  return base64Url(bytes);
}

async function beginOidcSignIn() {
  if (!state.config || !state.config.oidc_enabled || !window.crypto?.subtle) {
    setAuthError("Secure browser cryptography is required for sign-in.");
    return;
  }
  const verifier = secureRandom(48);
  const digest = await window.crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier));
  const challenge = base64Url(new Uint8Array(digest));
  const oauthState = secureRandom(24);
  window.sessionStorage.setItem(PKCE_VERIFIER_KEY, verifier);
  window.sessionStorage.setItem(OAUTH_STATE_KEY, oauthState);

  const authorize = new URL(state.config.authorization_endpoint);
  authorize.searchParams.set("response_type", "code");
  authorize.searchParams.set("client_id", state.config.client_id);
  authorize.searchParams.set("audience", state.config.audience);
  authorize.searchParams.set("redirect_uri", state.config.redirect_uri);
  authorize.searchParams.set("scope", state.config.scopes);
  authorize.searchParams.set("state", oauthState);
  authorize.searchParams.set("code_challenge", challenge);
  authorize.searchParams.set("code_challenge_method", "S256");
  window.location.assign(authorize.toString());
}

async function exchangeAuthorizationCode(code, returnedState) {
  const expectedState = window.sessionStorage.getItem(OAUTH_STATE_KEY);
  const verifier = window.sessionStorage.getItem(PKCE_VERIFIER_KEY);
  window.sessionStorage.removeItem(OAUTH_STATE_KEY);
  window.sessionStorage.removeItem(PKCE_VERIFIER_KEY);
  window.history.replaceState({}, document.title, window.location.pathname);

  if (!expectedState || !verifier || returnedState !== expectedState) {
    throw new Error("The sign-in response could not be verified. Please start again.");
  }
  const body = new URLSearchParams({
    grant_type: "authorization_code",
    client_id: state.config.client_id,
    code,
    code_verifier: verifier,
    redirect_uri: state.config.redirect_uri,
  });
  const response = await fetch(state.config.token_endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
    credentials: "omit",
    redirect: "error",
  });
  if (!response.ok) throw new Error("The identity provider did not complete sign-in.");
  const payload = await response.json();
  if (!payload || typeof payload.access_token !== "string" || !payload.access_token) {
    throw new Error("The identity provider did not return an access token.");
  }
  state.token = payload.access_token;
}

function setAuthError(message) {
  const node = byId("auth-error");
  node.textContent = message;
  node.hidden = false;
}

function showLanding() {
  if (state.token) return;
  byId("workspace").hidden = true;
  byId("auth-view").hidden = true;
  byId("landing-view").hidden = false;
}

function showSignIn() {
  if (state.token) return;
  byId("workspace").hidden = true;
  byId("landing-view").hidden = true;
  byId("auth-view").hidden = false;
  window.scrollTo({ top: 0, behavior: "auto" });
}

function syncPublicRoute() {
  if (state.token || !byId("workspace").hidden) return;
  if (window.location.hash === "#sign-in") showSignIn();
  else showLanding();
}

async function initializeAuthentication() {
  try {
    const query = new URLSearchParams(window.location.search);
    if (query.has("code") || query.has("error") || window.location.hash === "#sign-in") {
      showSignIn();
    } else {
      showLanding();
    }
    const response = await fetch(`${API_ROOT}/v1/dashboard/config`, {
      headers: { "Accept": "application/json" },
      credentials: "omit",
      redirect: "error",
    });
    if (!response.ok) throw new Error("Dashboard configuration is unavailable.");
    state.config = await response.json();

    if (query.has("error")) {
      window.history.replaceState({}, document.title, window.location.pathname);
      throw new Error("The identity provider did not complete sign-in.");
    }
    if (query.has("code")) {
      if (!state.config.oidc_enabled) {
        window.history.replaceState({}, document.title, window.location.pathname);
        throw new Error("OIDC sign-in is not configured for this dashboard.");
      }
      await exchangeAuthorizationCode(query.get("code"), query.get("state"));
      await openWorkspace();
      return;
    }

    byId("auth-loading").hidden = true;
    byId("auth-actions").hidden = false;
    byId("oidc-sign-in").hidden = !state.config.oidc_enabled;
    byId("development-token-form").hidden = !state.config.allow_development_token;
    byId("auth-unconfigured").hidden = Boolean(
      state.config.oidc_enabled || state.config.allow_development_token,
    );
  } catch (error) {
    byId("auth-loading").hidden = true;
    setAuthError(error instanceof Error ? error.message : "Sign-in could not be initialized.");
  }
}

function currentOrganization() {
  return state.me?.organizations?.find((organization) => organization.id === state.organizationId) || null;
}

function currentSource() {
  return state.sources.find((source) => source.id === state.sourceId) || null;
}

function canRunJobs() {
  return ["analyst", "admin", "owner"].includes(currentOrganization()?.role);
}

async function openWorkspace() {
  byId("auth-error").hidden = true;
  try {
    const me = await api("/v1/me", { organization: false });
    state.me = me;
    byId("landing-view").hidden = true;
    byId("auth-view").hidden = true;
    byId("workspace").hidden = false;
    byId("user-name").textContent = me.user.display_name || me.user.email || "Signed-in user";
    byId("user-avatar").textContent = initial(me.user.display_name || me.user.email);
    populateOrganizations();
    if (me.organizations.length) {
      state.organizationId = me.organizations[0].id;
      byId("organization-select").value = state.organizationId;
      await loadOrganization();
    } else {
      setGlobalStatus("No organization", "waiting");
      renderAll();
    }
  } catch (error) {
    state.token = null;
    byId("workspace").hidden = true;
    byId("landing-view").hidden = true;
    byId("auth-view").hidden = false;
    byId("auth-actions").hidden = false;
    setAuthError(error instanceof ApiError && error.status === 401
      ? "That sign-in is no longer valid. Please authenticate again."
      : "The workspace could not be loaded.");
  }
}

function populateOrganizations() {
  const select = byId("organization-select");
  select.replaceChildren();
  const organizations = state.me?.organizations || [];
  if (!organizations.length) {
    select.append(new Option("No organizations", ""));
    select.disabled = true;
    return;
  }
  organizations.forEach((organization) => select.append(new Option(organization.name, organization.id)));
  select.disabled = false;
}

function populateSources() {
  const select = byId("source-select");
  select.replaceChildren();
  if (!state.sources.length) {
    select.append(new Option("No sources", ""));
    select.disabled = true;
    return;
  }
  state.sources.forEach((source) => {
    const suffix = source.enabled ? "" : " · disabled";
    select.append(new Option(`${source.name}${suffix}`, source.id));
  });
  select.disabled = false;
  select.value = state.sourceId;
}

async function loadOrganization() {
  const loadId = ++state.organizationLoad;
  state.sourceId = null;
  state.sources = [];
  state.health = null;
  state.analysis = null;
  state.operations = null;
  state.jobs = [];
  showWorkspaceNotice("");
  setGlobalStatus("Loading organization", "waiting");
  renderAll(true);
  try {
    const [sources, jobs] = await Promise.all([api("/v1/sources"), api("/v1/jobs")]);
    if (loadId !== state.organizationLoad) return;
    state.sources = sources;
    state.jobs = jobs;
    state.sourceId = sources[0]?.id || null;
    populateSources();
    updateIdentityContext();
    if (state.sourceId) await loadSource();
    else {
      setGlobalStatus("No sources", "waiting");
      renderAll();
    }
  } catch (error) {
    if (loadId !== state.organizationLoad) return;
    setGlobalStatus("Workspace error", "bad");
    showWorkspaceNotice(error instanceof Error ? error.message : "Organization data could not be loaded.");
    renderAll();
  }
}

async function optionalLatestAnalysis() {
  try {
    return await api(`/v1/sources/${encodeURIComponent(state.sourceId)}/analyses/latest`);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return null;
    throw error;
  }
}

async function loadSource(silent = false) {
  if (!state.sourceId) return;
  const loadId = ++state.sourceLoad;
  if (!silent) {
    state.health = null;
    state.analysis = null;
    state.operations = null;
    showWorkspaceNotice("");
    setGlobalStatus("Loading source", "waiting");
    renderAll(true);
  }
  try {
    const encoded = encodeURIComponent(state.sourceId);
    const [health, analysis, operations] = await Promise.all([
      api(`/v1/sources/${encoded}/health`),
      optionalLatestAnalysis(),
      api(`/v1/sources/${encoded}/operations?window_hours=${state.windowHours}`),
    ]);
    if (loadId !== state.sourceLoad) return;
    state.health = health;
    state.analysis = analysis;
    state.operations = operations;
    setSourceStatus();
    renderAll();
  } catch (error) {
    if (loadId !== state.sourceLoad) return;
    setGlobalStatus("Source error", "bad");
    showWorkspaceNotice(error instanceof Error ? error.message : "Source data could not be loaded.");
    if (!silent) renderAll();
  }
}

async function refreshWorkspace(silent = false) {
  if (state.refreshing || !state.token || !state.organizationId) return;
  state.refreshing = true;
  const organizationId = state.organizationId;
  const trigger = byId("refresh-dashboard");
  if (!silent) trigger.disabled = true;
  try {
    const jobs = await api("/v1/jobs");
    if (organizationId !== state.organizationId) return;
    state.jobs = jobs;
    if (state.sourceId) await loadSource(true);
    else renderAll();
    if (!silent) showToast("Workspace refreshed.");
  } catch (error) {
    if (!silent) showToast(error instanceof Error ? error.message : "Refresh failed.", "error");
  } finally {
    state.refreshing = false;
    trigger.disabled = false;
  }
}

async function loadOperations() {
  if (!state.sourceId) return;
  const loadId = ++state.sourceLoad;
  state.operations = null;
  renderOperations(true);
  try {
    const encoded = encodeURIComponent(state.sourceId);
    const operations = await api(`/v1/sources/${encoded}/operations?window_hours=${state.windowHours}`);
    if (loadId !== state.sourceLoad) return;
    state.operations = operations;
    renderOperations();
  } catch (error) {
    if (loadId !== state.sourceLoad) return;
    showWorkspaceNotice(error instanceof Error ? error.message : "Operations could not be loaded.");
    renderOperations();
  }
}

function updateIdentityContext() {
  const organization = currentOrganization();
  byId("user-role").textContent = organization?.role || "No role";
  byId("run-analysis").hidden = !(canRunJobs() && state.sourceId);
}

function jobsForSource() {
  return state.jobs.filter((job) => job.source_id === state.sourceId);
}

function setSourceStatus() {
  const source = currentSource();
  const failedJobs = jobsForSource().filter((job) => ["failed", "dead_letter"].includes(job.state));
  if (!source?.enabled || state.health?.consecutive_failures > 0 || failedJobs.length) {
    setGlobalStatus("Attention needed", "bad");
  } else if (!state.analysis && (state.health?.accepted_events || 0) === 0) {
    setGlobalStatus("Waiting for data", "waiting");
  } else {
    setGlobalStatus("Source healthy", "good");
  }
}

function switchView(name) {
  if (!views[name]) return;
  state.view = name;
  document.querySelectorAll(".nav-item").forEach((item) => {
    const active = item.dataset.view === name;
    item.classList.toggle("is-active", active);
    if (active) item.setAttribute("aria-current", "page");
    else item.removeAttribute("aria-current");
  });
  Object.keys(views).forEach((key) => { byId(`view-${key}`).hidden = key !== name; });
  byId("view-title").textContent = views[name].title;
  byId("view-kicker").textContent = views[name].kicker;
}

function analysisBody() {
  const body = state.analysis?.result?.analysis;
  return body && typeof body === "object" ? body : null;
}

function releaseTone(figure) {
  if (!figure || !figure.released) return "warning";
  return figure.release_state === "draft" ? "warning" : "info";
}

function releaseLabel(figure) {
  if (!figure || !figure.released) return "withheld";
  return figure.release_state || "released";
}

function figureDisplay(figure) {
  if (!figure) return "Unavailable";
  if (!figure.released || figure.release_state === "withheld") return "Withheld";
  return typeof figure.display === "string" ? figure.display : "Unavailable";
}

function kpi(label, value, help, statusBadge = null) {
  const card = element("article", "kpi");
  const header = element("div", "kpi-label", label);
  if (statusBadge) header.append(statusBadge);
  card.append(header, element("div", "kpi-value", value), element("div", "kpi-help", help));
  return card;
}

function sectionIntro(title, detail, when = null) {
  const intro = element("div", "section-intro");
  const copy = element("div");
  copy.append(element("h2", "", title), element("p", "", detail));
  intro.append(copy);
  if (when) intro.append(element("span", "timestamp", when));
  return intro;
}

function sourceAlerts() {
  const alerts = [];
  const source = currentSource();
  const health = state.health;
  const analysis = state.analysis;
  const sourceJobs = jobsForSource();

  if (source && !source.enabled) {
    alerts.push({ tone: "danger", title: "Source disabled", detail: "Collector authentication is blocked for this source.", view: "jobs" });
  }
  if ((health?.consecutive_failures || 0) > 0) {
    alerts.push({
      tone: "danger",
      title: `${exactNumber(health.consecutive_failures)} consecutive analysis failure${health.consecutive_failures === 1 ? "" : "s"}`,
      detail: `Latest normalized error: ${titleCase(health.last_error_type)}.`,
      view: "jobs",
    });
  }
  const dead = sourceJobs.filter((job) => job.state === "dead_letter").length;
  if (dead) {
    alerts.push({ tone: "danger", title: `${dead} dead-letter job${dead === 1 ? "" : "s"}`, detail: "Fix the cause before retrying the job.", view: "jobs" });
  }
  if (analysis && health?.last_received_at && new Date(health.last_received_at) > new Date(analysis.created_at)) {
    alerts.push({ tone: "warning", title: "Analysis is behind ingestion", detail: "New events arrived after this result was created.", view: "jobs" });
  }
  if ((health?.rejected_events || 0) > 0) {
    alerts.push({ tone: "warning", title: `${exactNumber(health.rejected_events)} rejected event${health.rejected_events === 1 ? "" : "s"} recorded`, detail: "Check collector schema compatibility and size limits.", view: "operations" });
  }
  if (!analysis && (health?.accepted_events || 0) > 0) {
    alerts.push({ tone: "info", title: "Analysis pending", detail: "Events are present, but no completed analysis is available yet.", view: "jobs" });
  }
  if (!alerts.length) {
    alerts.push({ tone: "good", title: "No current health warning", detail: "The available source and job signals do not show a fault.", view: "operations" });
  }
  return alerts;
}

function renderOverview(loading = false) {
  const root = byId("overview-content");
  root.replaceChildren();
  if (loading) {
    root.append(loadingPanel());
    return;
  }
  if (!state.me?.organizations?.length) {
    root.append(statePanel("○", "No organization membership", "An organization owner must add this signed-in user before data can be viewed."));
    return;
  }
  if (!state.sourceId) {
    root.append(statePanel("+", "No source yet", "An admin or owner can create a source through the API, then connect an explicit prompt-free collector."));
    return;
  }
  if (!state.health) {
    root.append(statePanel("!", "Source details unavailable", "Retry after the API connection or authorization problem is resolved."));
    return;
  }

  const body = analysisBody();
  const spend = body?.spend?.monthly_input_usd;
  const coverage = body?.coverage || {};
  const ratios = body?.ratios || {};
  root.append(sectionIntro(
    currentSource()?.name || "Selected source",
    "See cache use, estimated spend, and source health for the latest completed analysis.",
    state.analysis ? `Analysis created ${timestamp(state.analysis.created_at)}` : "No completed analysis",
  ));

  const grid = element("div", "kpi-grid");
  grid.append(
    kpi(
      "Monthly input spend",
      body ? figureDisplay(spend) : "No analysis",
      body ? (spend?.withheld_because || "Display value returned by the analysis contract.") : "Run an analysis after events arrive.",
      body ? badge(releaseLabel(spend), releaseTone(spend)) : badge("empty", "neutral"),
    ),
    kpi("Input from cache", percent(ratios.input_from_cache), "Share of input tokens served from cache."),
    kpi("Analysable coverage", percent(coverage.fraction), coverage.total !== undefined ? `${exactNumber(coverage.analysed)} of ${exactNumber(coverage.total)} request rows.` : "No coverage record."),
    kpi("Accepted events", compactNumber(state.health.accepted_events), `Last received ${timestamp(state.health.last_received_at)}.`),
  );
  root.append(grid);

  const lower = element("div", "overview-grid");
  const healthCard = element("article", "card");
  const healthHead = element("div", "card-head");
  healthHead.append(element("h3", "", "Current status"), element("span", "", "Source and analysis health"));
  const healthBody = element("div", "card-body alert-list");
  sourceAlerts().forEach((alert) => {
    const row = element("div", `health-alert ${alert.tone}`);
    const copy = element("div");
    copy.append(element("strong", "", alert.title), element("span", "", alert.detail));
    const go = element("button", "link-button", "View →");
    go.type = "button";
    go.addEventListener("click", () => switchView(alert.view));
    row.append(element("i"), copy, go);
    healthBody.append(row);
  });
  healthCard.append(healthHead, healthBody);

  const factsCard = element("article", "card");
  const factsHead = element("div", "card-head");
  factsHead.append(element("h3", "", "Analysis details"), body ? badge(titleCase(body.tier), "info") : badge("empty", "neutral"));
  const factsBody = element("div", "card-body");
  const facts = element("dl", "source-facts");
  [
    ["Source kind", titleCase(currentSource()?.kind)],
    ["Prefix efficiency", percent(ratios.prefix_efficiency)],
    ["Observed requests", exactNumber(ratios.requests)],
    ["Last job duration", state.health.last_job_duration_ms === null ? "Not recorded" : milliseconds(state.health.last_job_duration_ms)],
    ["Last ingest lag", state.health.last_ingest_lag_seconds === null ? "Not recorded" : `${state.health.last_ingest_lag_seconds.toFixed(1)} s`],
    ["Analysis engine", state.analysis?.result?.engine?.version || "Not recorded"],
  ].forEach(([term, definition]) => {
    const row = element("div");
    row.append(element("dt", "", term), element("dd", "", definition));
    facts.append(row);
  });
  factsBody.append(facts);
  factsCard.append(factsHead, factsBody);
  lower.append(healthCard, factsCard);
  root.append(lower);
}

function renderRecommendations(loading = false) {
  const root = byId("recommendations-content");
  root.replaceChildren();
  if (loading) {
    root.append(loadingPanel());
    return;
  }
  if (!state.sourceId) {
    root.append(statePanel("↗", "No source selected", "Select a source to see its cache recommendations."));
    return;
  }
  const body = analysisBody();
  if (!body) {
    root.append(statePanel("↗", "No completed analysis", "Recommendations appear only after the worker completes an analysis.", canRunJobs() ? button("Run analysis", "button-primary", runAnalysis) : null));
    return;
  }
  const findings = Array.isArray(body.findings) ? body.findings : [];
  root.append(sectionIntro(
    findings.length ? `${findings.length} ranked recommendation${findings.length === 1 ? "" : "s"}` : "No recommendation returned",
    "Each recommendation explains the observed problem, suggested change, and response-quality risk.",
    `Registry ${state.analysis.result.registry?.sha256?.slice(0, 10) || "not recorded"}`,
  ));
  if (!findings.length) {
    root.append(statePanel("✓", "No cache issue found", "Check again after more traffic arrives or after the workload changes."));
  } else {
    const list = element("div", "recommendation-list");
    findings.forEach((finding, index) => {
      const row = element("article", "recommendation");
      row.append(element("div", "recommendation-rank", String(index + 1).padStart(2, "0")));
      const main = element("div", "recommendation-main");
      const meta = element("div", "recommendation-meta");
      const severity = String(finding.severity || "unknown").toLowerCase();
      meta.append(
        badge(severity, severity === "high" ? "danger" : severity === "medium" ? "warning" : "neutral"),
        badge(finding.evidence_class || "evidence not recorded", "info"),
        badge(finding.code || "uncoded", "neutral"),
      );
      main.append(meta, element("h3", "", finding.title || "Untitled finding"));
      if (finding.detail) main.append(element("p", "", finding.detail));
      if (finding.fix) {
        const action = element("div", "recommendation-action");
        action.append(element("strong", "", "Recommended action · "), document.createTextNode(finding.fix));
        main.append(action);
      }
      const impact = element("aside", "recommendation-impact");
      const figure = finding.avoidable_usd_month;
      const amount = element("div");
      amount.append(element("div", "impact-label", "Avoidable / month"), element("div", "impact-value", figureDisplay(figure)), badge(releaseLabel(figure), releaseTone(figure)));
      impact.append(amount);
      impact.append(element("div", "quality-risk", `Quality risk · ${finding.quality_risk || "Not recorded"}`));
      row.append(main, impact);
      list.append(row);
    });
    root.append(list);
  }
  const notes = element("div", "analysis-notes");
  (body.caveats || []).forEach((note) => notes.append(element("div", "analysis-note caveat", note)));
  (body.notes || []).forEach((note) => notes.append(element("div", "analysis-note", note)));
  if (notes.childElementCount) root.append(notes);
}

function volumeChart(operations) {
  const wrapper = element("div", "chart-wrap");
  const buckets = operations.buckets || [];
  if (!buckets.length) {
    wrapper.append(statePanel("⌁", "No request signals in this window", "The API returned no valid prompt-free events for the selected period."));
    return wrapper;
  }
  const svg = svgElement("svg", { viewBox: "0 0 760 245", class: "volume-chart", role: "img", "aria-label": "Request volume and error events over time" });
  const chartTitle = svgElement("title");
  chartTitle.textContent = "Request volume bars with normalized error markers";
  svg.append(chartTitle);
  [35, 90, 145, 200].forEach((y) => svg.append(svgElement("line", { x1: 20, y1: y, x2: 740, y2: y, stroke: "#dedbd3", "stroke-width": 1 })));
  const maximum = Math.max(1, ...buckets.map((bucket) => bucket.requests));
  const step = 720 / buckets.length;
  const width = Math.max(3, Math.min(25, step * 0.62));
  buckets.forEach((bucket, index) => {
    const x = 20 + (index * step) + ((step - width) / 2);
    const height = (bucket.requests / maximum) * 175;
    const rect = svgElement("rect", { x, y: 210 - height, width, height, rx: 3, fill: "#a9d7c8" });
    const rectTitle = svgElement("title");
    rectTitle.textContent = `${timestamp(bucket.started_at)}: ${bucket.requests} requests`;
    rect.append(rectTitle);
    svg.append(rect);
    if (bucket.errors > 0) {
      const circle = svgElement("circle", { cx: x + (width / 2), cy: Math.max(19, 202 - height), r: 4.5, fill: "#eb735d" });
      const errorTitle = svgElement("title");
      errorTitle.textContent = `${bucket.errors} error events`;
      circle.append(errorTitle);
      svg.append(circle);
    }
  });
  svg.append(svgElement("line", { x1: 20, y1: 210, x2: 740, y2: 210, stroke: "#9eaaa6", "stroke-width": 1 }));
  wrapper.append(svg);
  const labels = element("div", "chart-label");
  labels.append(element("span", "legend", "Requests"), element("span", "legend errors", "Errors"));
  labels.append(element("span", "", `${timestamp(buckets[0].started_at)} → ${timestamp(buckets[buckets.length - 1].started_at)}`));
  wrapper.append(labels);
  return wrapper;
}

function renderOperations(loading = false) {
  const root = byId("operations-content");
  root.replaceChildren();
  if (loading) {
    root.append(loadingPanel());
    return;
  }
  if (!state.sourceId) {
    root.append(statePanel("⌁", "No source selected", "Select a source to inspect latency, time-to-first-token, errors, and volume."));
    return;
  }
  const operations = state.operations;
  if (!operations) {
    root.append(statePanel("!", "Operations unavailable", "Retry after the API connection or authorization problem is resolved."));
    return;
  }
  if (operations.truncated || operations.invalid_events > 0) {
    const messages = [];
    if (operations.truncated) messages.push(`The API examined its configured limit of ${exactNumber(operations.events_examined)} recent events.`);
    if (operations.invalid_events > 0) messages.push(`${exactNumber(operations.invalid_events)} stored event rows did not pass current schema validation.`);
    messages.push("This is a partial view; no missing values were estimated.");
    root.append(element("p", "partial-note", messages.join(" ")));
  }
  const summary = element("div", "kpi-grid");
  summary.append(
    kpi("Valid events", compactNumber(operations.valid_events), `${exactNumber(operations.events_examined)} rows examined by the API.`),
    kpi("Successful", compactNumber(operations.status.success), "Events with a validated success outcome."),
    kpi("Latency p50 / p95", `${milliseconds(operations.latency.p50_ms)} / ${milliseconds(operations.latency.p95_ms)}`, `${exactNumber(operations.latency.count)} events included.`),
    kpi("TTFT p50 / p95", `${milliseconds(operations.time_to_first_token.p50_ms)} / ${milliseconds(operations.time_to_first_token.p95_ms)}`, `${exactNumber(operations.time_to_first_token.count)} events included.`),
  );
  root.append(summary);

  const layout = element("div", "ops-grid");
  const chartCard = element("article", "card");
  const chartHead = element("div", "card-head");
  chartHead.append(element("h3", "", "Volume by receipt bucket"), element("span", "", `${operations.bucket_minutes}-minute buckets`));
  chartCard.append(chartHead, volumeChart(operations));

  const errorsCard = element("article", "card");
  const errorsHead = element("div", "card-head");
  errorsHead.append(element("h3", "", "Normalized outcomes"), element("span", "", `${exactNumber(operations.status.error)} errors`));
  const errorsBody = element("div", "card-body");
  const errorList = element("div", "error-list");
  [
    ...operations.errors.map((item) => [titleCase(item.error_type), item.count]),
    ["Cancelled", operations.status.cancelled],
    ["Unknown", operations.status.unknown],
  ].filter(([, count]) => count > 0).forEach(([label, count]) => {
    const row = element("div", "error-row");
    row.append(element("span", "", label), element("strong", "", exactNumber(count)));
    errorList.append(row);
  });
  if (!errorList.childElementCount) errorList.append(element("p", "muted", "No non-success outcomes in this window."));
  errorsBody.append(errorList);
  errorsCard.append(errorsHead, errorsBody);
  layout.append(chartCard, errorsCard);
  root.append(layout);
}

async function performJobAction(job, action, trigger) {
  trigger.disabled = true;
  try {
    await api(`/v1/jobs/${encodeURIComponent(job.id)}/${action}`, { method: "POST" });
    state.jobs = await api("/v1/jobs");
    showToast(action === "retry" ? "Job queued for retry." : "Job cancelled.");
    setSourceStatus();
    renderAll();
  } catch (error) {
    showToast(error instanceof Error ? error.message : "Job action failed.", "error");
  } finally {
    trigger.disabled = false;
  }
}

async function runAnalysis(event) {
  if (!state.sourceId || !canRunJobs()) return;
  const trigger = event?.currentTarget || byId("run-analysis");
  trigger.disabled = true;
  try {
    await api(`/v1/sources/${encodeURIComponent(state.sourceId)}/jobs`, { method: "POST", body: {} });
    state.jobs = await api("/v1/jobs");
    showToast("Analysis job queued.");
    switchView("jobs");
    renderJobs();
  } catch (error) {
    showToast(error instanceof Error ? error.message : "Analysis could not be queued.", "error");
  } finally {
    trigger.disabled = false;
  }
}

function renderJobs(loading = false) {
  const root = byId("jobs-content");
  root.replaceChildren();
  if (loading) {
    root.append(loadingPanel());
    return;
  }
  if (!state.sourceId) {
    root.append(statePanel("◷", "No source selected", "Select a source to inspect its analysis jobs and ingestion health."));
    return;
  }
  const actions = canRunJobs() ? button("Run analysis", "button-primary", runAnalysis) : null;
  const intro = sectionIntro(
    `${currentSource()?.name || "Selected source"} jobs`,
    canRunJobs() ? "Your role can run, cancel, and retry analysis jobs. Every action is enforced and audited by the API." : "Your role can read job status. Analyst, admin, or owner access is required for job actions.",
    state.health ? `Last success ${timestamp(state.health.last_success_at)}` : null,
  );
  if (actions) intro.append(actions);
  root.append(intro);

  if (state.health) {
    const healthGrid = element("div", "kpi-grid");
    healthGrid.append(
      kpi("Consecutive failures", exactNumber(state.health.consecutive_failures), `Last normalized error: ${titleCase(state.health.last_error_type)}.`, state.health.consecutive_failures ? badge("attention", "danger") : badge("clear", "info")),
      kpi("Rejected events", compactNumber(state.health.rejected_events), "Cumulative source health counter."),
      kpi("Duplicate events", compactNumber(state.health.duplicate_events), "Idempotent replays recorded by the API."),
      kpi("Latest duration", state.health.last_job_duration_ms === null ? "—" : milliseconds(state.health.last_job_duration_ms), "Duration of the last completed job attempt."),
    );
    root.append(healthGrid);
  }

  const jobs = jobsForSource();
  if (!jobs.length) {
    root.append(statePanel("◷", "No analysis jobs", "Queue a job after the collector has delivered prompt-free events."));
    return;
  }
  const list = element("div", "job-list");
  jobs.forEach((job) => {
    const row = element("article", "job-row");
    const identity = element("div", "job-id");
    identity.append(element("strong", "", job.id), element("span", "", `Created ${timestamp(job.created_at)}`));
    const stateCell = element("div", "job-cell");
    const jobTone = job.state === "succeeded" ? "info" : ["failed", "dead_letter"].includes(job.state) ? "danger" : job.state === "running" ? "warning" : "neutral";
    stateCell.append(badge(titleCase(job.state), jobTone));
    const attempt = element("div", "job-cell");
    attempt.append(element("strong", "", `${job.attempt} / ${job.max_attempts}`), element("small", "", "attempts"));
    const schedule = element("div", "job-cell");
    schedule.append(element("strong", "", timestamp(job.scheduled_for)), element("small", "", job.error_type ? `Error: ${titleCase(job.error_type)}` : "Scheduled time"));
    const controls = element("div", "job-actions");
    if (canRunJobs() && ["failed", "dead_letter"].includes(job.state)) {
      const retry = button("Retry", "button-secondary", (event) => performJobAction(job, "retry", event.currentTarget));
      controls.append(retry);
    }
    if (canRunJobs() && ["queued", "running"].includes(job.state)) {
      const cancel = button("Cancel", "button-danger", (event) => performJobAction(job, "cancel", event.currentTarget));
      controls.append(cancel);
    }
    row.append(identity, stateCell, attempt, schedule, controls);
    list.append(row);
  });
  root.append(list);
}

function renderNavigationCounts() {
  const findings = analysisBody()?.findings?.length || 0;
  const findingNode = byId("finding-count");
  findingNode.textContent = String(findings);
  findingNode.hidden = findings === 0;

  const alerts = jobsForSource().filter((job) => ["failed", "dead_letter"].includes(job.state)).length;
  const jobNode = byId("job-alert-count");
  jobNode.textContent = String(alerts);
  jobNode.hidden = alerts === 0;
}

function renderAll(loading = false) {
  updateIdentityContext();
  renderOverview(loading);
  renderRecommendations(loading);
  renderOperations(loading);
  renderJobs(loading);
  renderNavigationCounts();
}

function signOut() {
  state.token = null;
  state.me = null;
  state.organizationId = null;
  state.sourceId = null;
  state.sources = [];
  state.health = null;
  state.analysis = null;
  state.operations = null;
  state.jobs = [];
  state.refreshing = false;
  window.sessionStorage.removeItem(OAUTH_STATE_KEY);
  window.sessionStorage.removeItem(PKCE_VERIFIER_KEY);
  byId("workspace").hidden = true;
  byId("auth-view").hidden = true;
  byId("landing-view").hidden = false;
  byId("auth-actions").hidden = false;
  byId("auth-error").hidden = true;
  window.history.replaceState({}, document.title, `${window.location.pathname}#top`);
}

function animateEstimate(node, nextValue, formatter) {
  const previousValue = Number(node.dataset.value ?? nextValue);
  node.dataset.value = String(nextValue);
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    node.textContent = formatter(nextValue);
    return;
  }

  const startedAt = performance.now();
  const duration = 360;
  const tick = (now) => {
    if (node.dataset.value !== String(nextValue)) return;
    const elapsed = Math.min((now - startedAt) / duration, 1);
    const eased = 1 - Math.pow(1 - elapsed, 3);
    node.textContent = formatter(previousValue + ((nextValue - previousValue) * eased));
    if (elapsed < 1 && node.dataset.value === String(nextValue)) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function setRangeProgress(input) {
  const minimum = Number(input.min);
  const maximum = Number(input.max);
  const progress = ((Number(input.value) - minimum) / (maximum - minimum)) * 100;
  input.style.setProperty("--range-progress", `${progress}%`);
}

function updateRoiCalculator() {
  const baselineInput = byId("roi-baseline");
  const repeatInput = byId("roi-repeat");
  const discountInput = byId("roi-discount");
  const baseline = Number(baselineInput.value);
  const repeatShare = Number(repeatInput.value) / 100;
  const discount = Number(discountInput.value) / 100;
  const monthlySavings = baseline * repeatShare * discount;
  const projected = baseline - monthlySavings;
  const reduction = baseline === 0 ? 0 : monthlySavings / baseline;

  [baselineInput, repeatInput, discountInput].forEach(setRangeProgress);
  byId("roi-baseline-display").textContent = currency(baseline);
  byId("roi-repeat-display").textContent = percent(repeatShare);
  byId("roi-discount-display").textContent = percent(discount);
  byId("roi-baseline-chart").textContent = currency(baseline);
  byId("roi-percent").textContent = `${percent(reduction)} of monthly input-token spend`;

  animateEstimate(byId("roi-monthly"), monthlySavings, currency);
  animateEstimate(byId("roi-projected"), projected, currency);
  animateEstimate(byId("roi-annual"), monthlySavings * 12, currency);
  animateEstimate(byId("roi-reduction"), reduction, percent);

  byId("roi-standard-bar").style.width = "100%";
  byId("roi-scenario-bar").style.width = `${Math.max(0, 100 - (reduction * 100))}%`;
}

function initializeLandingInteractions() {
  requestAnimationFrame(() => document.documentElement.classList.add("landing-ready"));
  const revealNodes = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.14 });
    revealNodes.forEach((node) => observer.observe(node));
  } else {
    revealNodes.forEach((node) => node.classList.add("is-visible"));
  }

  const calculator = byId("roi-calculator");
  calculator.addEventListener("input", updateRoiCalculator);
  calculator.addEventListener("submit", (event) => event.preventDefault());
  updateRoiCalculator();

  const film = byId("product-film");
  document.querySelectorAll(".film-chapter").forEach((chapter) => {
    chapter.addEventListener("click", async () => {
      document.querySelectorAll(".film-chapter").forEach((item) => {
        const selected = item === chapter;
        item.classList.toggle("is-active", selected);
        item.setAttribute("aria-selected", String(selected));
      });
      film.poster = chapter.dataset.poster;
      film.src = chapter.dataset.film;
      byId("film-duration").textContent = chapter.dataset.duration;
      film.load();
      try {
        await film.play();
      } catch (_error) {
        // Browser autoplay policy can leave the film paused with controls visible.
      }
    });
  });

  if ("IntersectionObserver" in window && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    const filmObserver = new IntersectionObserver((entries) => {
      entries.forEach(async (entry) => {
        if (entry.isIntersecting) {
          try { await film.play(); } catch (_error) { /* Controls remain available. */ }
        } else {
          film.pause();
        }
      });
    }, { threshold: 0.55 });
    filmObserver.observe(film);
  }
}

byId("oidc-sign-in").addEventListener("click", beginOidcSignIn);
byId("development-token-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = byId("development-token");
  state.token = input.value;
  event.currentTarget.reset();
  await openWorkspace();
});
byId("organization-select").addEventListener("change", async (event) => {
  state.organizationId = event.target.value || null;
  await loadOrganization();
});
byId("source-select").addEventListener("change", async (event) => {
  state.sourceId = event.target.value || null;
  updateIdentityContext();
  await loadSource();
});
byId("window-select").addEventListener("change", async (event) => {
  state.windowHours = Number(event.target.value);
  await loadOperations();
});
byId("run-analysis").addEventListener("click", runAnalysis);
byId("refresh-dashboard").addEventListener("click", () => refreshWorkspace(false));
byId("sign-out").addEventListener("click", signOut);
window.addEventListener("hashchange", syncPublicRoute);
document.querySelectorAll(".nav-item").forEach((item) => {
  item.addEventListener("click", () => switchView(item.dataset.view));
});

initializeLandingInteractions();
initializeAuthentication();

window.setInterval(() => {
  if (!document.hidden) refreshWorkspace(true);
}, 30000);
