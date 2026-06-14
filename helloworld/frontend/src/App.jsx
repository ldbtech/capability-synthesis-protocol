import { useEffect, useRef, useState } from "react";

const API = "/api";

export default function App() {
  const [dataset, setDataset] = useState(null);
  const [caps, setCaps] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scroller = useRef(null);

  useEffect(() => { refreshDataset(); refreshCaps(); }, []);
  useEffect(() => {
    scroller.current?.scrollTo(0, scroller.current.scrollHeight);
  }, [messages]);

  async function refreshDataset() {
    const r = await fetch(`${API}/dataset`).then((r) => r.json()).catch(() => null);
    if (r) setDataset(r);
  }
  async function refreshCaps() {
    const r = await fetch(`${API}/capabilities`).then((r) => r.json()).catch(() => null);
    if (r) setCaps(r.capabilities || []);
  }

  async function onUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(`${API}/upload`, { method: "POST", body: fd }).then((r) => r.json());
    setDataset(r);
    pushSystem(`Loaded ${r.filename}: ${r.row_count} rows · ${r.columns.length} columns`);
  }

  function pushSystem(text) {
    setMessages((m) => [...m, { role: "system", text }]);
  }

  async function describe() {
    // Borrows the existing describe_dataset capability directly (no planner).
    const r = await fetch(`${API}/describe`, { method: "POST" }).then((r) => r.json());
    const res = r.result || {};
    pushSystem(
      `🔗 Borrowed "${r.borrowed}" → ${res.row_count} rows, columns: ${(res.columns || []).join(", ")}`
    );
  }

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);

    setMessages((m) => [...m, { role: "user", text }]);
    const idx = messages.length + 1;
    const assistant = { role: "assistant", trace: [], steps: [], result: null, synthesized: [] };
    setMessages((m) => [...m, assistant]);

    const update = (patch) =>
      setMessages((m) => {
        const copy = [...m];
        copy[idx] = { ...copy[idx], ...patch(copy[idx]) };
        return copy;
      });

    try {
      const resp = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop();
        for (const part of parts) {
          const line = part.replace(/^data: /, "").trim();
          if (!line) continue;
          const ev = JSON.parse(line);
          handleEvent(ev, update);
        }
      }
    } catch (err) {
      update(() => ({ result: { status: "ERROR", summary: String(err) } }));
    } finally {
      setBusy(false);
      refreshCaps();
    }
  }

  function handleEvent(ev, update) {
    if (ev.type === "planning") {
      if (ev.steps) update((a) => ({ steps: ev.steps, trace: [...a.trace, `🧠 plan: ${ev.steps.join(", ")}`] }));
      else update((a) => ({ trace: [...a.trace, `🧠 ${ev.message}`] }));
    } else if (ev.type === "event") {
      const k = ev.kind;
      if (k === "LOG") {
        const synth = /Capability synthesized: (.+)/.exec(ev.message);
        update((a) => ({
          trace: [...a.trace, `   ${ev.message}`],
          synthesized: synth ? [...a.synthesized, synth[1]] : a.synthesized,
        }));
      } else if (k === "CAPABILITY") {
        update((a) => ({ trace: [...a.trace, `⚙️ ${ev.message}`] }));
      } else if (k === "CAPABILITY_END") {
        update((a) => ({ trace: [...a.trace, `✅ ${ev.message}`] }));
      }
    } else if (ev.type === "result") {
      update(() => ({ result: ev }));
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>CSP<span>·</span>CSV-RAG</h1>
        <p className="tag">RAG when it knows. Synthesizes & runs real code when it doesn't.</p>

        <section className="panel">
          <h3>Dataset</h3>
          <label className="upload">
            <input type="file" accept=".csv" onChange={onUpload} hidden />
            ⬆ Upload CSV
          </label>
          {dataset?.ready ? (
            <div className="meta">
              <strong>{dataset.filename}</strong>
              <span>{dataset.row_count} rows</span>
              <div className="chips">
                {dataset.columns.map((c) => <span key={c} className="chip">{c}</span>)}
              </div>
              <button className="borrow-btn" onClick={describe}>
                🔗 Describe (borrows capability)
              </button>
            </div>
          ) : <p className="muted">No CSV loaded yet.</p>}
        </section>

        <section className="panel">
          <h3>Capabilities</h3>
          {caps.map((c) => <CapabilityCard key={c.name} cap={c} />)}
        </section>
      </aside>

      <main className="chat">
        <div className="messages" ref={scroller}>
          {messages.length === 0 && (
            <div className="empty">
              <h2>Ask anything about your data</h2>
              <div className="examples">
                <span>"Who works in Data Science?"</span>
                <span>"Average salary per department"</span>
                <span>"Correlation between age and salary"</span>
              </div>
            </div>
          )}
          {messages.map((m, i) => <Message key={i} m={m} />)}
        </div>

        <div className="composer">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder={dataset?.ready ? "Ask about your data…" : "Upload a CSV, then ask…"}
            disabled={busy}
          />
          <button onClick={send} disabled={busy || !input.trim()}>
            {busy ? "…" : "Send"}
          </button>
        </div>
      </main>
    </div>
  );
}

function Message({ m }) {
  if (m.role === "user") return <div className="msg user"><div className="bubble">{m.text}</div></div>;
  if (m.role === "system") return <div className="msg system">{m.text}</div>;

  const answer = extractAnswer(m.result);
  const sources = extractSources(m.result);
  const images = extractImages(m.result);

  return (
    <div className="msg assistant">
      <div className="bubble">
        {m.synthesized?.length > 0 && (
          <div className="synthbar">⚡ Synthesized & ran new code: {m.synthesized.join(", ")}</div>
        )}
        {images.map((src, i) => <img key={i} className="plot" src={src} alt="generated plot" />)}
        {answer ? <div className="answer">{answer}</div> : <div className="thinking">working…</div>}

        {sources?.length > 0 && (
          <details className="sources">
            <summary>{sources.length} source rows</summary>
            {sources.map((s, i) => (
              <div key={i} className="source">
                <span className="score">{s.score}</span>
                <code>{JSON.stringify(s.row)}</code>
              </div>
            ))}
          </details>
        )}

        {m.trace?.length > 0 && (
          <details className="trace">
            <summary>execution trace</summary>
            <pre>{m.trace.join("\n")}</pre>
          </details>
        )}
      </div>
    </div>
  );
}

function CapabilityCard({ cap }) {
  return (
    <details className="cap">
      <summary>
        <span className={`dot ${cap.kind}`} />
        {cap.name}
        <em>{cap.kind}</em>
      </summary>
      <p className="capdesc">{cap.description}</p>
      {cap.code && <pre className="code">{cap.code}</pre>}
    </details>
  );
}

function extractAnswer(result) {
  if (!result) return null;
  if (result.status === "ERROR") return `⚠ ${result.summary || result.error}`;
  const out = result.output || {};
  for (const v of Object.values(out)) {
    if (v && typeof v === "object" && typeof v.answer === "string") return v.answer;
  }
  // Computational result: pretty-print the payload, fall back to summary
  const payload = Object.values(out)[0];
  if (payload && typeof payload === "object") return prettyResult(payload);
  return result.summary || null;
}

function prettyResult(obj) {
  // Drop base64 image blobs from the text dump — they're rendered as <img>.
  const cleaned = stripImageFields(obj);
  if (cleaned && typeof cleaned === "object" && Object.keys(cleaned).length === 0) return "";
  try { return JSON.stringify(cleaned, null, 2); } catch { return String(obj); }
}

const IMG_KEY = /(image|histogram|chart|plot|figure|png).*base64|base64.*(image|png)/i;

function isBase64Png(v) {
  return typeof v === "string" && v.length > 100 && /^[A-Za-z0-9+/=\s]+$/.test(v.slice(0, 64));
}

function extractImages(result) {
  const imgs = [];
  const out = result?.output || {};
  const walk = (o) => {
    if (!o || typeof o !== "object") return;
    for (const [k, v] of Object.entries(o)) {
      if (IMG_KEY.test(k) && isBase64Png(v)) imgs.push(`data:image/png;base64,${v}`);
      else if (typeof v === "string" && v.startsWith("data:image")) imgs.push(v);
      else if (v && typeof v === "object") walk(v);
    }
  };
  walk(out);
  return imgs;
}

function stripImageFields(o) {
  if (Array.isArray(o)) return o.map(stripImageFields);
  if (o && typeof o === "object") {
    const r = {};
    for (const [k, v] of Object.entries(o)) {
      if (IMG_KEY.test(k) && (isBase64Png(v) || (typeof v === "string" && v.startsWith("data:image")))) continue;
      if (v === null && IMG_KEY.test(k)) continue;
      r[k] = stripImageFields(v);
    }
    return r;
  }
  return o;
}

function extractSources(result) {
  if (!result?.output) return [];
  for (const v of Object.values(result.output)) {
    if (v && Array.isArray(v.sources)) return v.sources;
  }
  return [];
}
