import { useEffect, useRef, useState } from "react";

const API = "/api";
const NODES = ["understand", "build", "narrate"];
const EXAMPLES = [
  "visualize quicksort",
  "animate binary search",
  "show BFS on a graph",
  "merge sort",
  "selection sort",
  "Dijkstra shortest path",
];

export default function App() {
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [nodeStatus, setNodeStatus] = useState({});
  const [log, setLog] = useState([]);
  const [code, setCode] = useState(null);
  const [capability, setCapability] = useState(null);
  const [borrowed, setBorrowed] = useState(false);
  const [frames, setFrames] = useState([]);
  const [narration, setNarration] = useState("");
  const [error, setError] = useState(null);

  async function run(req) {
    const goal = (req ?? input).trim();
    if (!goal || busy) return;
    setBusy(true);
    setNodeStatus({}); setLog([]); setCode(null); setCapability(null); setBorrowed(false);
    setFrames([]); setNarration(""); setError(null);

    const pushLog = (line) => setLog((l) => [...l, line]);

    try {
      const resp = await fetch(`${API}/visualize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request: goal }),
      });
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop();
        for (const p of parts) {
          const line = p.replace(/^data: /, "").trim();
          if (!line) continue;
          handle(JSON.parse(line), pushLog);
        }
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  function handle(ev, pushLog) {
    switch (ev.type) {
      case "node":
        setNodeStatus((s) => ({ ...s, [ev.node]: ev.status }));
        if (ev.detail) pushLog(`▸ ${ev.node}: ${ev.detail}`);
        break;
      case "csp":
        if (ev.kind === "plan") pushLog(`🧠 plan → ${ev.steps.join(", ")}`);
        else if (ev.kind === "retry") pushLog(`♻️ ${ev.message}`);
        else if (ev.kind === "borrow") pushLog(`🔗 ${ev.message}`);
        else if (ev.kind === "log") pushLog(`   ${ev.message}`);
        break;
      case "code":
        setCapability(ev.capability);
        setCode(ev.code);
        setBorrowed(!!ev.borrowed);
        pushLog(ev.borrowed
          ? `🔗 borrowed ${ev.capability} (reused, no synthesis)`
          : `⚡ invented ${ev.capability} (${ev.code.length} chars of new code)`);
        break;
      case "narration":
        setNarration(ev.text);
        break;
      case "done":
        setFrames(ev.frames || []);
        if (!ev.frames?.length) setError("No frames were produced.");
        break;
      case "error":
        setError(ev.message);
        break;
      default:
        break;
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Algo<span>Viz</span></h1>
        <p>
          Type any algorithm. There's <b>no code for it</b> — CSP writes the
          visualizer on the fly, runs it as a node in a LangGraph workflow, and
          animates the result.
        </p>
      </header>

      <div className="composer">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
          placeholder="e.g. visualize quicksort"
          disabled={busy}
        />
        <button onClick={() => run()} disabled={busy || !input.trim()}>
          {busy ? "building…" : "Visualize"}
        </button>
      </div>
      <div className="chips">
        {EXAMPLES.map((e) => (
          <button key={e} className="chip" disabled={busy}
            onClick={() => { setInput(e); run(e); }}>{e}</button>
        ))}
      </div>

      {/* LangGraph workflow strip */}
      <div className="workflow">
        {NODES.map((n, i) => (
          <div key={n} className="wf-step">
            <div className={`wf-node ${nodeStatus[n] || ""}`}>
              <span className="wf-dot" />
              {n}
              {n === "build" && <em>CSP</em>}
            </div>
            {i < NODES.length - 1 && <div className="wf-edge" />}
          </div>
        ))}
      </div>

      <div className="grid">
        {/* Animation */}
        <section className="panel viz">
          <h3>Animation</h3>
          {frames.length > 0 ? (
            <Player frames={frames} />
          ) : (
            <div className="placeholder">
              {busy ? "synthesizing & rendering…" : "the animation will appear here"}
            </div>
          )}
          {narration && <p className="narration">{narration}</p>}
          {error && <p className="error">⚠ {error}</p>}
        </section>

        {/* The WOW: freshly-written code */}
        <section className="panel code">
          <h3>
            {capability
              ? (borrowed
                  ? <>🔗 Borrowed (reused): <code>{capability}</code></>
                  : <>⚡ Invented live: <code>{capability}</code></>)
              : "Generated code"}
          </h3>
          {code ? (
            <pre>{code}</pre>
          ) : (
            <div className="placeholder small">
              CSP's generated Python will appear here — this code did not exist
              before you asked.
            </div>
          )}
        </section>
      </div>

      {/* Live trace */}
      <section className="panel trace">
        <h3>Live workflow trace</h3>
        <pre>{log.join("\n") || "…"}</pre>
      </section>
    </div>
  );
}

function Player({ frames }) {
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [fps, setFps] = useState(4);
  const timer = useRef(null);

  useEffect(() => { setI(0); setPlaying(true); }, [frames]);

  useEffect(() => {
    if (!playing || frames.length === 0) return;
    timer.current = setInterval(() => {
      setI((x) => (x + 1) % frames.length);
    }, 1000 / fps);
    return () => clearInterval(timer.current);
  }, [playing, fps, frames]);

  return (
    <div className="player">
      <img src={`data:image/png;base64,${frames[i]}`} alt={`frame ${i}`} />
      <div className="controls">
        <button onClick={() => setPlaying((p) => !p)}>{playing ? "⏸" : "▶"}</button>
        <input type="range" min="0" max={frames.length - 1} value={i}
          onChange={(e) => { setPlaying(false); setI(+e.target.value); }} />
        <span className="counter">{i + 1}/{frames.length}</span>
        <label className="fps">
          {fps} fps
          <input type="range" min="1" max="12" value={fps}
            onChange={(e) => setFps(+e.target.value)} />
        </label>
      </div>
    </div>
  );
}
