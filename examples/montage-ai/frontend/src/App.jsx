import { useEffect, useRef, useState } from "react";
import "./App.css";

const API = "/api";

const EXAMPLES = [
  "Create a mobile app login screen",
  "Add a hero section with a big headline and subtitle",
  "Make a 3-column pricing card layout",
  "Add a navigation bar at the top",
  "Draw a pie chart with 4 segments",
  "Create a dashboard with stats cards",
  "Add a gradient banner",
  "Make a timeline with 4 steps",
];

export default function App() {
  const [canvas, setCanvas] = useState({ elements: [], canvas_width: 1200, canvas_height: 700 });
  const [caps, setCaps] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState(null);
  const [pendingCreds, setPendingCreds] = useState([]); // [{env_key, service, get_it_at}]
  const [pendingGoal, setPendingGoal] = useState(null);
  const [credValues, setCredValues] = useState({});
  const scroller = useRef(null);

  useEffect(() => { refreshCanvas(); refreshCaps(); }, []);
  useEffect(() => {
    scroller.current?.scrollTo(0, scroller.current.scrollHeight);
  }, [messages]);

  async function refreshCanvas() {
    const r = await fetch(`${API}/canvas`).then(r => r.json()).catch(() => null);
    if (r) setCanvas(r);
  }

  async function refreshCaps() {
    const r = await fetch(`${API}/capabilities`).then(r => r.json()).catch(() => null);
    if (r) setCaps(r.capabilities || []);
  }

  async function clearCanvas() {
    await fetch(`${API}/canvas/clear`, { method: "POST" });
    setCanvas(c => ({ ...c, elements: [] }));
    setSelected(null);
  }

  async function send(text) {
    text = (text || input).trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);

    const userMsg = { role: "user", text };
    const assistantMsg = { role: "assistant", trace: [], synthesized: [], result: null };
    setMessages(m => [...m, userMsg, assistantMsg]);

    const update = patch =>
      setMessages(m => {
        const copy = [...m];
        const i = copy.length - 1;
        copy[i] = { ...copy[i], ...patch(copy[i]) };
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
      refreshCanvas();
    }
  }

  function handleEvent(ev, update) {
    if (ev.type === "planning") {
      const msg = ev.steps ? `🧠 plan: ${ev.steps.join(" → ")}` : `🧠 ${ev.message}`;
      update(a => ({ trace: [...a.trace, msg] }));
    } else if (ev.type === "event") {
      const synth = /Capability synthesized: (.+)/.exec(ev.message || "");
      update(a => ({
        trace: [...a.trace, `   ${ev.message}`],
        synthesized: synth ? [...a.synthesized, synth[1]] : a.synthesized,
      }));
    } else if (ev.type === "credential_required") {
      setPendingCreds(prev => {
        if (prev.find(c => c.env_key === ev.env_key)) return prev;
        return [...prev, ev];
      });
    } else if (ev.type === "result") {
      if (ev.status === "PENDING_CREDENTIALS" && ev.pending_goal) {
        setPendingGoal(ev.pending_goal);
      }
      update(() => ({ result: ev }));
    }
  }

  async function submitCredentials() {
    for (const cred of pendingCreds) {
      const val = credValues[cred.env_key]?.trim();
      if (!val) continue;
      await fetch(`${API}/credential`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ env_key: cred.env_key, value: val }),
      });
    }
    const goal = pendingGoal;
    setPendingCreds([]);
    setPendingGoal(null);
    setCredValues({});
    if (goal) send(goal); // auto-retry with credentials now stored
  }

  const layers = [...canvas.elements].reverse();
  const synthCount = caps.filter(c => c.kind === "synthesized").length;

  return (
    <div className="app">
      {/* ── Top bar ── */}
      <header className="topbar">
        <div className="logo">montage<span>AI</span></div>
        <div className="topbar-pills">
          <span className="pill reg">{caps.filter(c => c.kind === "registered").length} registered</span>
          <span className="pill syn">{synthCount} synthesized</span>
        </div>
        <div className="topbar-right">
          <button className="clear-btn" onClick={clearCanvas}>Clear canvas</button>
        </div>
      </header>

      <div className="workspace">
        {/* ── Layers panel ── */}
        <aside className="layers-panel">
          <div className="panel-section">
            <div className="panel-head">Layers</div>
            {layers.length === 0
              ? <p className="empty-hint">Canvas is empty</p>
              : layers.map(el => (
                <div key={el.id}
                  className={`layer-row ${selected === el.id ? "sel" : ""}`}
                  onClick={() => setSelected(el.id === selected ? null : el.id)}>
                  <span className="layer-icon">{LAYER_ICONS[el.type] || "◈"}</span>
                  <span className="layer-label">{el.label || el.type}</span>
                </div>
              ))
            }
          </div>

          <div className="panel-section">
            <div className="panel-head">Capabilities</div>
            {caps.map(c => (
              <div key={c.name} className="cap-row">
                <span className={`cdot ${c.kind}`} />
                <span className="cap-name">{c.name}</span>
              </div>
            ))}
          </div>
        </aside>

        {/* ── Canvas ── */}
        <main className="canvas-area" onClick={() => setSelected(null)}>
          {canvas.elements.length === 0 && (
            <div className="canvas-empty">
              <div className="canvas-empty-inner">
                <div className="canvas-empty-icon">✦</div>
                <p>Describe a design in the chat →</p>
              </div>
            </div>
          )}
          <div className="canvas-frame">
            <svg
              className="canvas-svg"
              viewBox={`0 0 ${canvas.canvas_width} ${canvas.canvas_height}`}
              preserveAspectRatio="xMidYMid meet"
            >
              <rect width={canvas.canvas_width} height={canvas.canvas_height} fill="#f8f9fc" />
              {canvas.elements.map(el => (
                <CanvasElement
                  key={el.id}
                  el={el}
                  selected={selected === el.id}
                  onSelect={e => { e.stopPropagation(); setSelected(el.id); }}
                />
              ))}
            </svg>
          </div>
        </main>

        {/* ── Chat panel ── */}
        <aside className="chat-panel">
          <div className="panel-head" style={{ padding: "14px 14px 8px" }}>
            AI Design Assistant
            {busy && <span className="busy-dot" />}
          </div>

          <div className="chat-msgs" ref={scroller}>
            {messages.length === 0 && (
              <div className="chat-empty">
                <p className="chat-hint">Describe what to design and CSP will synthesize the capability to build it.</p>
                <div className="examples">
                  {EXAMPLES.map(e => (
                    <button key={e} className="example-chip" onClick={() => send(e)}>{e}</button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m, i) => <ChatMsg key={i} m={m} />)}
          </div>

          {pendingCreds.length > 0 && (
            <div className="cred-form">
              <div className="cred-form-head">
                🔑 API credentials needed
              </div>
              {pendingCreds.map(c => (
                <div key={c.env_key} className="cred-row">
                  <div className="cred-meta">
                    <span className="cred-service">{c.service || c.env_key}</span>
                    {c.get_it_at && (
                      <a className="cred-link" href={c.get_it_at} target="_blank" rel="noreferrer">
                        Get key ↗
                      </a>
                    )}
                  </div>
                  <input
                    className="cred-input"
                    type="password"
                    placeholder={c.env_key}
                    value={credValues[c.env_key] || ""}
                    onChange={e => setCredValues(v => ({ ...v, [c.env_key]: e.target.value }))}
                  />
                </div>
              ))}
              <button
                className="cred-submit"
                onClick={submitCredentials}
                disabled={pendingCreds.every(c => !credValues[c.env_key]?.trim())}
              >
                Save &amp; continue →
              </button>
            </div>
          )}

          <div className="chat-composer">
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && send()}
              placeholder="Describe a design change…"
              disabled={busy}
            />
            <button onClick={() => send()} disabled={busy || !input.trim()}>
              {busy ? "…" : "→"}
            </button>
          </div>
        </aside>
      </div>
    </div>
  );
}

// ── Canvas element renderer ───────────────────────────────────────────────────

function CanvasElement({ el, selected, onSelect }) {
  if (el.type === "rect") return (
    <g onClick={onSelect} style={{ cursor: "pointer" }}>
      <rect x={el.x} y={el.y} width={el.w} height={el.h}
        fill={el.fill || "#6c8cff"} stroke={el.stroke && el.stroke !== "none" ? el.stroke : "none"}
        strokeWidth={1} rx={el.rx || 0} opacity={el.opacity ?? 1} />
      {selected && (
        <rect x={el.x - 2} y={el.y - 2} width={el.w + 4} height={el.h + 4}
          fill="none" stroke="#6c8cff" strokeWidth="1.5"
          strokeDasharray="5 3" rx={(el.rx || 0) + 2} />
      )}
    </g>
  );

  if (el.type === "circle") return (
    <g onClick={onSelect} style={{ cursor: "pointer" }}>
      <ellipse
        cx={el.x + (el.w || 80) / 2} cy={el.y + (el.h || 80) / 2}
        rx={(el.w || 80) / 2} ry={(el.h || 80) / 2}
        fill={el.fill || "#36d399"} stroke={el.stroke && el.stroke !== "none" ? el.stroke : "none"}
        opacity={el.opacity ?? 1} />
      {selected && (
        <ellipse cx={el.x + (el.w || 80) / 2} cy={el.y + (el.h || 80) / 2}
          rx={(el.w || 80) / 2 + 3} ry={(el.h || 80) / 2 + 3}
          fill="none" stroke="#6c8cff" strokeWidth="1.5" strokeDasharray="5 3" />
      )}
    </g>
  );

  if (el.type === "text") return (
    <g onClick={onSelect} style={{ cursor: "pointer" }}>
      <text x={el.x} y={el.y + (el.fontSize || 16)}
        fill={el.fill || "#1b1f2a"} fontSize={el.fontSize || 16}
        fontFamily="ui-sans-serif, system-ui, sans-serif"
        fontWeight={el.fontWeight || "normal"}
        opacity={el.opacity ?? 1}>
        {el.text || el.label}
      </text>
      {selected && (
        <rect x={el.x - 4} y={el.y - 2}
          width={(el.w || 200) + 8} height={(el.fontSize || 16) * 1.4 + 6}
          fill="none" stroke="#6c8cff" strokeWidth="1.5" strokeDasharray="5 3" />
      )}
    </g>
  );

  if (el.type === "line") return (
    <line onClick={onSelect} style={{ cursor: "pointer" }}
      x1={el.x} y1={el.y}
      x2={el.x + (el.w || 200)} y2={el.y + (el.h || 0)}
      stroke={el.stroke || el.fill || "#8b94a7"} strokeWidth={el.strokeWidth || 2}
      opacity={el.opacity ?? 1} />
  );

  return null;
}

// ── Chat message ──────────────────────────────────────────────────────────────

function ChatMsg({ m }) {
  if (m.role === "user") return (
    <div className="cmsg user"><div className="bubble">{m.text}</div></div>
  );

  const summary = m.result?.output
    ? Object.values(m.result.output).find(v => v?.summary)?.summary
    : m.result?.summary;

  const isError = m.result?.status === "ERROR";

  return (
    <div className="cmsg assistant">
      <div className="bubble">
        {m.synthesized?.length > 0 && (
          <div className="synthbar">
            ⚡ Synthesized: {m.synthesized.join(", ")}
          </div>
        )}
        {summary
          ? <div className={`answer ${isError ? "err" : ""}`}>
              {isError ? `⚠ ${summary}` : `✓ ${summary}`}
            </div>
          : <div className="thinking">working…</div>
        }
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

const LAYER_ICONS = { rect: "▭", circle: "◯", text: "T", line: "╱" };
