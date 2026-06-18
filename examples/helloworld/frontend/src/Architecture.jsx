import { useEffect, useState } from "react";
import "./Architecture.css";

// ── Diagram layout (SVG coords) ───────────────────────────────────────────────
const NODES = {
  consumer:    { x: 30,  y: 215, w: 165, h: 94, label: "Consumer",    sub: "FastAPI · LangGraph · CLI" },
  planner:     { x: 295, y: 70,  w: 175, h: 78, label: "Planner",     sub: "reuse · evolve · synthesize" },
  registry:    { x: 295, y: 222, w: 175, h: 78, label: "Registry",    sub: "general reusable verbs" },
  executor:    { x: 295, y: 374, w: 175, h: 78, label: "Executor",    sub: "runs the verb w/ new args" },
  llm:         { x: 545, y: 70,  w: 175, h: 78, label: "LLM",         sub: "writes / patches code" },
  synthesizer: { x: 545, y: 222, w: 175, h: 78, label: "Synthesizer", sub: "real Python def run()" },
  sandbox:     { x: 545, y: 374, w: 175, h: 78, label: "Sandbox",     sub: "subprocess + timeout" },
  store:       { x: 770, y: 374, w: 110, h: 78, label: "planner/",    sub: "persists · reloads" },
};

const EDGES = {
  "c-p":  { from: [195, 250], to: [295, 109], label: "goal" },
  "p-r":  { from: [382, 148], to: [382, 222] },
  "r-x":  { from: [382, 300], to: [382, 374], label: "exists → reuse" },
  "r-s":  { from: [470, 261], to: [545, 261], label: "new / evolve" },
  "s-l":  { from: [632, 222], to: [632, 148], label: "asks LLM" },
  "s-sb": { from: [632, 300], to: [632, 374] },
  "sb-st":{ from: [720, 413], to: [770, 413], label: "persist" },
  "x-c":  { from: [295, 410], to: [150, 305], label: "result", result: true },
  "sb-c": { from: [545, 430], to: [180, 300], label: "result", result: true },
};

const STEPS = [
  { t: "Submit a goal",
    d: "A consumer sends a natural-language goal. Here it's the CSV-RAG web app, but it could just as easily be a LangGraph node, a script, or an MCP-style stdio host.",
    nodes: ["consumer"], edges: ["c-p"] },
  { t: "Plan — capabilities are general verbs",
    d: "The Planner reads the Registry, where each capability is a general, reusable VERB (plot_chart, aggregate_table) — not a one-off task. It decides each step: reuse a verb, evolve one, or synthesize a new one.",
    nodes: ["planner", "registry"], edges: ["p-r"] },
  { t: "Reuse — same verb, new args",
    d: "If a verb already fits — even if it was first made for a different request — the Executor just runs it with NEW args. Ask for a bar chart, then a histogram: both reuse one plot_chart. No LLM, no regeneration.",
    nodes: ["registry", "executor"], edges: ["r-x"] },
  { t: "Missing → synthesize a general verb",
    d: "If nothing fits, the Synthesizer asks the LLM to WRITE Python — a real def run(args) that reads every specific from args, so it handles the whole class, not just today's question. Compile-checked before it's trusted.",
    nodes: ["registry", "synthesizer", "llm"], edges: ["r-s", "s-l"] },
  { t: "Evolve — patch a verb in place",
    d: "Want to change an existing verb (\"add goal difference to the table\")? The Synthesizer hands the LLM the current code plus your instruction and patches only what's needed — no rebuild from scratch.",
    nodes: ["registry", "synthesizer", "llm"], edges: ["r-s", "s-l"] },
  { t: "Run the code over real data",
    d: "The verb's code runs in an isolated Python sandbox (subprocess + timeout) over your real CSV rows — not a mock, not a simulation.",
    nodes: ["synthesizer", "sandbox"], edges: ["s-sb"] },
  { t: "Persist & reuse forever",
    d: "The verb (its spec + generated .py) is saved to planner/ and reloaded next run. Synthesized once, then reused across every similar goal — or borrowed safely while in use.",
    nodes: ["sandbox", "store"], edges: ["sb-st"] },
  { t: "Result streams back",
    d: "Outputs are collected and streamed back to the consumer as live events plus a final result.",
    nodes: ["executor", "sandbox", "consumer"], edges: ["x-c", "sb-c"] },
];

export default function Architecture() {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setStep((s) => (s + 1) % STEPS.length), 2800);
    return () => clearInterval(id);
  }, [playing]);

  const cur = STEPS[step];
  const activeNodes = new Set(cur.nodes);
  const activeEdges = new Set(cur.edges);

  return (
    <div className="arch">
      <header className="arch-head">
        <h1>How CSP works</h1>
        <p>
          Same shape as MCP — a consumer talks to an orchestrator — but when a
          capability <b>doesn't exist, CSP writes the code for it and runs it.</b>{" "}
          And what it writes is a <b>general, reusable verb</b>, so the next
          similar goal reuses it instead of regenerating.
        </p>
      </header>

      <div className="arch-body">
        <svg className="diagram" viewBox="0 0 900 470" preserveAspectRatio="xMidYMid meet">
          <defs>
            <marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="3"
              orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L7,3 L0,6 Z" className="arrowhead" />
            </marker>
            <marker id="arrow-r" markerWidth="9" markerHeight="9" refX="7" refY="3"
              orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L7,3 L0,6 Z" className="arrowhead-r" />
            </marker>
          </defs>

          {Object.entries(EDGES).map(([id, e]) => {
            const active = activeEdges.has(id);
            const [x1, y1] = e.from, [x2, y2] = e.to;
            return (
              <g key={id} className={`edge ${active ? "active" : ""} ${e.result ? "result" : ""}`}>
                <line x1={x1} y1={y1} x2={x2} y2={y2}
                  markerEnd={`url(#arrow${e.result ? "-r" : ""})`} />
                {e.label && (
                  <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 7} textAnchor="middle" className="elabel">
                    {e.label}
                  </text>
                )}
              </g>
            );
          })}

          {Object.entries(NODES).map(([id, n]) => {
            const active = activeNodes.has(id);
            const cx = n.x + n.w / 2;
            return (
              <g key={id} className={`node ${active ? "active" : ""}`}
                 onClick={() => { setPlaying(false); jumpToNode(id, setStep); }}>
                <rect x={n.x} y={n.y} width={n.w} height={n.h} rx="14" />
                <text x={cx} y={n.y + n.h / 2 - 3} textAnchor="middle" className="nlabel">{n.label}</text>
                <text x={cx} y={n.y + n.h / 2 + 17} textAnchor="middle" className="nsub">{n.sub}</text>
              </g>
            );
          })}
        </svg>

        <aside className="arch-side">
          <div className="step-card">
            <div className="step-num">{step + 1} / {STEPS.length}</div>
            <h3>{cur.t}</h3>
            <p>{cur.d}</p>
          </div>

          <div className="steps">
            {STEPS.map((s, i) => (
              <button key={i}
                className={`steprow ${i === step ? "on" : ""}`}
                onClick={() => { setPlaying(false); setStep(i); }}>
                <span className="dot" />{s.t}
              </button>
            ))}
          </div>

          <div className="arch-controls">
            <button onClick={() => { setPlaying(false); setStep((s) => (s - 1 + STEPS.length) % STEPS.length); }}>‹ Prev</button>
            <button className="play" onClick={() => setPlaying((p) => !p)}>{playing ? "⏸ Pause" : "▶ Play"}</button>
            <button onClick={() => { setPlaying(false); setStep((s) => (s + 1) % STEPS.length); }}>Next ›</button>
          </div>
        </aside>
      </div>

      <div className="legend">
        <span><i className="sw reg" /> registered — a function you wrote</span>
        <span><i className="sw syn" /> synthesized — a general verb the LLM wrote, run in the sandbox</span>
        <span><i className="sw res" /> result path</span>
      </div>
    </div>
  );
}

// Clicking a node jumps to the first step that highlights it.
function jumpToNode(id, setStep) {
  const i = STEPS.findIndex((s) => s.nodes.includes(id));
  if (i >= 0) setStep(i);
}
