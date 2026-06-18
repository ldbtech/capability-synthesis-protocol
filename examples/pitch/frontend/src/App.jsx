import { useEffect, useRef, useState } from "react";

const EXAMPLES = [
  "World Cup 2026 Group A standings",
  "Argentina squad with key players",
  "Who wins Argentina vs France? predict it",
  "Simulate the knockout bracket",
  "Top scorers leaderboard",
];

export default function App() {
  const [views, setViews] = useState([]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [caps, setCaps] = useState([]);
  const [pendingCreds, setPendingCreds] = useState([]);
  const [pendingGoal, setPendingGoal] = useState("");
  const [credValues, setCredValues] = useState({});
  const scrollRef = useRef(null);

  const refreshBoard = async () => {
    const r = await fetch("/api/board");
    const d = await r.json();
    setViews(d.views || []);
  };
  const refreshCaps = async () => {
    const r = await fetch("/api/capabilities");
    const d = await r.json();
    setCaps(d.capabilities || []);
  };

  useEffect(() => {
    refreshBoard();
    refreshCaps();
  }, []);
  useEffect(() => {
    scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight);
  }, [messages]);

  async function send(text) {
    const goal = (text ?? input).trim();
    if (!goal || busy) return;
    setInput("");
    setBusy(true);
    setPendingCreds([]);
    const mine = { role: "user", text: goal };
    const reply = { role: "assistant", trace: [], plan: [], synth: [], answer: "", err: false };
    setMessages((m) => [...m, mine, reply]);

    const update = (patch) =>
      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1] = { ...copy[copy.length - 1], ...patch };
        return copy;
      });

    const creds = [];
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: goal }),
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let cur = { ...reply };
      const apply = (patch) => {
        cur = { ...cur, ...patch };
        update(cur);
      };
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop();
        for (const part of parts) {
          const line = part.replace(/^data: /, "").trim();
          if (!line) continue;
          let ev;
          try { ev = JSON.parse(line); } catch { continue; }

          if (ev.type === "planning") {
            if (ev.steps) apply({ plan: ev.steps });
            apply({ trace: [...cur.trace, ev.message] });
          } else if (ev.type === "credential_required") {
            creds.push(ev);
          } else if (ev.type === "event") {
            const msg = ev.message || "";
            if (msg.startsWith("Synthesizing capability:") || msg.startsWith("Capability synthesized:")) {
              const name = msg.split(":")[1]?.trim();
              if (name && !cur.synth.includes(name)) apply({ synth: [...cur.synth, name] });
            }
            apply({ trace: [...cur.trace, msg] });
          } else if (ev.type === "result") {
            if (ev.status === "PENDING_CREDENTIALS") {
              apply({ answer: "I need an API key to fetch that — add it below and I'll continue." });
            } else {
              apply({ answer: ev.summary || "Done.", err: (ev.errors || []).length > 0 });
            }
          }
        }
      }
      if (creds.length) {
        setPendingCreds(creds);
        setPendingGoal(goal);
      }
    } catch (e) {
      update({ answer: "Request failed: " + e.message, err: true });
    } finally {
      setBusy(false);
      refreshBoard();
      refreshCaps();
    }
  }

  async function submitCredentials() {
    for (const c of pendingCreds) {
      const v = (credValues[c.env_key] || "").trim();
      if (!v) continue;
      await fetch("/api/credential", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ env_key: c.env_key, value: v }),
      });
    }
    const goal = pendingGoal;
    setPendingCreds([]);
    setCredValues({});
    send(goal);
  }

  const registered = caps.filter((c) => c.kind === "registered" || c.kind === "REGISTERED");
  const synthesized = caps.filter((c) => !(c.kind === "registered" || c.kind === "REGISTERED"));

  return (
    <div className="app">
      <header className="topbar">
        <div className="logo">⚽ Pitch <span>· World Cup copilot</span></div>
        <div className="pills">
          <span className="pill reg">{registered.length} registered</span>
          <span className="pill syn">{synthesized.length} synthesized</span>
        </div>
        <button className="clear" onClick={async () => { await fetch("/api/board/clear", { method: "POST" }); refreshBoard(); }}>
          Clear board
        </button>
      </header>

      <div className="workspace">
        {/* Board */}
        <main className="board">
          {views.length === 0 ? (
            <div className="board-empty">
              <div className="be-icon">🏟️</div>
              <p>Ask anything about the World Cup — live or predicted.</p>
              <p className="be-sub">CSP synthesizes the tool it needs, then reuses it.</p>
            </div>
          ) : (
            views.map((v, i) => <ViewCard key={i} v={v} />)
          )}
        </main>

        {/* Chat */}
        <aside className="chat">
          <div className="chat-head">
            Ask Pitch {busy && <span className="busy" />}
          </div>
          <div className="chat-msgs" ref={scrollRef}>
            {messages.length === 0 && (
              <div className="hint">
                <p>Try one:</p>
                {EXAMPLES.map((e) => (
                  <button key={e} className="chip" onClick={() => send(e)}>{e}</button>
                ))}
              </div>
            )}
            {messages.map((m, i) =>
              m.role === "user" ? (
                <div key={i} className="msg user"><div className="bubble">{m.text}</div></div>
              ) : (
                <div key={i} className="msg bot">
                  <div className="bubble">
                    {m.synth?.length > 0 && (
                      <div className="synthbar">⚡ Synthesized: {m.synth.join(", ")}</div>
                    )}
                    {m.plan?.length > 0 && (
                      <div className="planbar">🧠 {m.plan.join(" → ")}</div>
                    )}
                    {m.answer
                      ? <div className={"answer" + (m.err ? " err" : "")}>{m.answer}</div>
                      : <div className="thinking">thinking…</div>}
                    {m.trace?.length > 0 && (
                      <details className="trace">
                        <summary>trace</summary>
                        <pre>{m.trace.join("\n")}</pre>
                      </details>
                    )}
                  </div>
                </div>
              )
            )}
          </div>

          {pendingCreds.length > 0 && (
            <div className="cred-form">
              <div className="cred-head">🔑 API key needed</div>
              {pendingCreds.map((c) => (
                <div key={c.env_key} className="cred-row">
                  <div className="cred-meta">
                    <span className="cred-svc">{c.service || c.env_key}</span>
                    {c.get_it_at && <a className="cred-link" href={c.get_it_at} target="_blank" rel="noreferrer">get key ↗</a>}
                  </div>
                  <input
                    className="cred-input"
                    type="password"
                    placeholder={c.env_key}
                    value={credValues[c.env_key] || ""}
                    onChange={(e) => setCredValues((s) => ({ ...s, [c.env_key]: e.target.value }))}
                  />
                </div>
              ))}
              <button className="cred-submit" onClick={submitCredentials}>Save & continue</button>
            </div>
          )}

          <div className="composer">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send()}
              placeholder="Ask about scores, squads, or predictions…"
              disabled={busy}
            />
            <button onClick={() => send()} disabled={busy}>➤</button>
          </div>
        </aside>
      </div>
    </div>
  );
}

/* ── Typed view renderers ─────────────────────────────────────────────────── */
function ViewCard({ v }) {
  return (
    <section className="view">
      {v.title && <h2 className="view-title">{v.title}</h2>}
      {v.view === "table" && <TableView data={v.data} />}
      {v.view === "cards" && <CardsView data={v.data} />}
      {v.view === "bracket" && <BracketView data={v.data} />}
      {v.view === "chart" && <ChartView data={v.data} />}
      {v.view === "stat" && <StatView data={v.data} />}
      {v.summary && <p className="view-summary">{v.summary}</p>}
    </section>
  );
}

function TableView({ data }) {
  const cols = data?.columns || [];
  const rows = data?.rows || [];
  return (
    <table className="tbl">
      <thead><tr>{cols.map((c, i) => <th key={i}>{c}</th>)}</tr></thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>{(Array.isArray(r) ? r : Object.values(r)).map((c, j) => <td key={j}>{String(c)}</td>)}</tr>
        ))}
      </tbody>
    </table>
  );
}

function CardsView({ data }) {
  const cards = data?.cards || [];
  return (
    <div className="cards">
      {cards.map((c, i) => (
        <div key={i} className="card">
          {c.image && <img className="card-img" src={c.image} alt="" onError={(e) => (e.target.style.display = "none")} />}
          <div className="card-title">{c.title}</div>
          {c.subtitle && <div className="card-sub">{c.subtitle}</div>}
          <div className="card-stats">
            {(c.stats || []).map((s, j) => (
              <div key={j} className="stat-chip">
                <span className="sc-label">{s.label}</span>
                <span className="sc-value">{s.value}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function BracketView({ data }) {
  const rounds = data?.rounds || [];
  return (
    <div className="bracket">
      {rounds.map((rd, i) => (
        <div key={i} className="round">
          <div className="round-name">{rd.name}</div>
          {(rd.matches || []).map((m, j) => (
            <div key={j} className="match">
              <div className="team"><span>{m.home}</span><b>{m.homeScore ?? ""}</b></div>
              <div className="team"><span>{m.away}</span><b>{m.awayScore ?? ""}</b></div>
              {m.note && <div className="match-note">{m.note}</div>}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function ChartView({ data }) {
  const bars = data?.bars || [];
  const max = Math.max(1, ...bars.map((b) => Number(b.value) || 0));
  return (
    <div className="chart">
      {bars.map((b, i) => (
        <div key={i} className="bar-row">
          <span className="bar-label">{b.label}</span>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${(Number(b.value) / max) * 100}%` }} />
          </div>
          <span className="bar-val">{b.value}{data.unit ? ` ${data.unit}` : ""}</span>
        </div>
      ))}
    </div>
  );
}

function StatView({ data }) {
  return (
    <div className="bigstat">
      <div className="bs-value">{data?.value}</div>
      <div className="bs-label">{data?.label}</div>
      {data?.sub && <div className="bs-sub">{data.sub}</div>}
    </div>
  );
}
