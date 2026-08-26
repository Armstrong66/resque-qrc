"""Create an editable, publication-quality architecture figure for ResQue.

The figure is intentionally model-centric: it depicts the actual input tensor,
PCA projection, recurrent nine-qubit PennyLane TFIM circuit, measurements, and
selected linear readout.  It is not a pipeline diagram.  Values are loaded
from the authoritative per-horizon result artifacts when available.

Examples
--------
python scripts/generate_architecture_diagram.py --horizon 6
python scripts/generate_architecture_diagram.py --horizon 24 --output-dir outputs/figures

Outputs
-------
* ``resque_architecture_h{h}.svg``: publication-ready vector artwork.
* ``resque_architecture_h{h}.drawio``: editable diagrams.net source.
* ``resque_architecture_h{h}_spec.md``: exact circuit and run configuration.
* ``resque_architecture_h{h}_metadata.json``: provenance used for the figure.

No third-party Python package is required.  Open the .drawio file in
https://app.diagrams.net and export PDF/SVG there if a journal requires it.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def config_value(name: str, default: Any) -> Any:
    """Read project config only when the module can be imported safely."""
    try:
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import config  # type: ignore
        return getattr(config, name, default)
    except Exception:
        return default


def model_metadata(results: Path, horizon: int) -> dict[str, Any]:
    arch_path = results / f"best_qrc_architecture_h{horizon}.json"
    arch = load_json(arch_path)
    source = "saved per-horizon architecture" if arch else "configuration defaults (no saved architecture found)"
    arch = {
        "J": arch.get("J", config_value("J_DEFAULT", 1.0)),
        "h": arch.get("h", config_value("H_DEFAULT", 0.5)),
        "topology": arch.get("topology", config_value("TOPOLOGY_PRIMARY", "chain")),
        "use_data_reuploading": arch.get(
            "use_data_reuploading", config_value("USE_DATA_REUPLOADING", True)
        ),
        "n_qubits": arch.get("n_qubits", config_value("QUBIT_PRIMARY", 9)),
    }
    n = int(arch["n_qubits"])
    warm = load_json(results / f"h{horizon}" / "warm_start_source_ablation.json")
    warm_config = load_json(results / f"h{horizon}" / "warm_start_qrc_config.json")
    active = bool(warm.get("transfer_active", warm_config.get("transfer_active", False)))
    selected_source = warm.get("selected_source", warm_config.get("warm_start_source"))
    alpha = warm_config.get("warm_prior_alpha")
    readout = (f"transfer-prior ridge ({selected_source})" if active and selected_source
               else "cold ridge readout")
    return {
        **arch,
        "n_qubits": n,
        "feature_dim": 2 * n,
        "window_size": int(config_value("WINDOW_SIZE", 20)),
        "n_variables": len(config_value("TARGETS", ["temperature", "humidity", "pressure", "wind_speed"])),
        "targets": config_value("TARGETS", ["temperature", "humidity", "pressure", "wind_speed"]),
        "trotter_steps": int(config_value("TROTTER_STEPS", 4)),
        "evolution_time": float(config_value("EVOLUTION_TIME", 1.0)),
        "feedback": bool(config_value("USE_FEEDBACK", True)),
        "noise_rate": 0.0,
        "shots": "exact",
        "horizon": horizon,
        "architecture_source": source,
        "readout": readout,
        "transfer_active": active,
        "transfer_source": selected_source,
        "warm_prior_alpha": alpha,
    }


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def svg_rect(x, y, w, h, fill, stroke="#20334c", radius=18, shadow=True, opacity=1.0):
    filt = ' filter="url(#shadow)"' if shadow else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="2"{filt}/>')


def svg_text(x, y, text, size=18, fill="#17263c", weight="400", anchor="start", family="Inter,Arial,sans-serif"):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(text)}</text>')


def svg_multiline(x, y, lines, size=16, leading=21, fill="#17263c", weight="400", anchor="start"):
    return "".join(svg_text(x, y + i * leading, line, size, fill, weight, anchor)
                   for i, line in enumerate(lines))


def svg_arrow(x1, y1, x2, y2, color="#334d72", dashed=False, width=3):
    dash = ' stroke-dasharray="8 7"' if dashed else ""
    return (f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" marker-end="url(#arrow)"{dash}/>')


def render_svg(meta: dict[str, Any]) -> str:
    n, L, d = meta["n_qubits"], meta["window_size"], meta["feature_dim"]
    reupload = meta["use_data_reuploading"]
    dt = meta["evolution_time"] / meta["trotter_steps"]
    title = f"ResQue | Horizon {meta['horizon']} h deployed QRC architecture"
    encoding = "data reuploading" if reupload else "standard single injection"
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="1420" viewBox="0 0 2400 1420">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#f8fbff"/><stop offset="1" stop-color="#edf4fb"/></linearGradient>
  <linearGradient id="quantum" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#5c6cff"/><stop offset="1" stop-color="#25356f"/></linearGradient>
  <linearGradient id="data" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#e6faf4"/><stop offset="1" stop-color="#a9e7d4"/></linearGradient>
  <linearGradient id="readout" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#ffe6b0"/><stop offset="1" stop-color="#f6b73c"/></linearGradient>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="5" stdDeviation="6" flood-color="#20334c" flood-opacity=".17"/></filter>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#334d72"/></marker>
</defs>
<rect width="2400" height="1420" fill="url(#bg)"/>
<rect x="30" y="24" width="2340" height="88" rx="22" fill="#10223c"/>
{svg_text(70, 70, title, 31, '#ffffff', '700')}
{svg_text(70, 96, f"Actual values: n={n} | J={meta['J']} | h={meta['h']} | {meta['topology']} | {encoding} | p=0 | exact expectations", 17, '#b9d5f5', '400')}
{svg_text(2325, 71, 'MODEL ARCHITECTURE', 15, '#7cd6ff', '700', 'end')}
<text x="2325" y="95" font-family="Inter,Arial,sans-serif" font-size="13" fill="#b9d5f5" text-anchor="end">{esc(meta['architecture_source'])}</text>
''']

    # Main lane section headers
    sections = [(55, 145, 425, 655, "A", "Temporal weather tensor"),
                (505, 145, 340, 655, "B", "Train-only compression"),
                (870, 145, 790, 655, "C", "Recurrent quantum reservoir"),
                (1685, 145, 650, 655, "D", "Measured state and forecast")]
    for x, y, w, h, letter, label in sections:
        parts += [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="#ffffff" fill-opacity=".74" stroke="#c9d9eb" stroke-width="1.5"/>',
                  f'<circle cx="{x+30}" cy="{y+33}" r="16" fill="#1d416a"/>',
                  svg_text(x+30, y+39, letter, 16, '#ffffff', '700', 'middle'),
                  svg_text(x+58, y+39, label, 19, '#1d416a', '700')]

    # Data tensor: four channel cards and window timeline
    parts += [svg_text(80, 225, f"Input window  Xₜ₋ₗ₊₁:ₜ  ∈ ℝ^{L}×4  (L = {L} six-hour steps)", 17, '#274a6f', '700')]
    colors = ["#ed6a5a", "#43aa8b", "#577590", "#f9c74f"]
    targets_short = ["T", "RH", "SLP", "WS"]
    for i, (label, color) in enumerate(zip(targets_short, colors)):
        y = 260 + i * 73
        parts += [svg_rect(82, y, 354, 54, '#ffffff', '#d5e1ec', 10, False),
                  f'<rect x="83" y="{y+1}" width="9" height="52" rx="4" fill="{color}"/>',
                  svg_text(108, y+33, label, 15, '#24435f', '700'),
                  f'<polyline points="{145},{y+37} {170},{y+22} {197},{y+31} {224},{y+17} {251},{y+28} {278},{y+20} {306},{y+36} {334},{y+24} {360},{y+29} {405},{y+16}" fill="none" stroke="{color}" stroke-width="3"/>',
                  svg_text(425, y+33, 't', 12, '#66819a', '400', 'end')]
    parts += [svg_text(95, 585, "20 × 4 history matrix", 15, '#274a6f', '700'),
              svg_text(95, 611, "temperature · humidity · pressure · wind speed", 14, '#5c7793'),
              svg_text(95, 658, "normalised using train split statistics", 13, '#5c7793')]

    # PCA block
    parts += [svg_arrow(447, 410, 537, 410), svg_rect(545, 275, 258, 245, 'url(#data)', '#218a74', 20),
              svg_text(674, 322, 'PCA', 35, '#105c4d', '700', 'middle'),
              svg_text(674, 352, 'fit on training windows only', 14, '#105c4d', '400', 'middle'),
              f'<path d="M590 394 L635 360 L681 407 L730 325" fill="none" stroke="#16876d" stroke-width="4"/>',
              f'<line x1="588" y1="423" x2="760" y2="423" stroke="#16876d" stroke-width="2"/>',
              svg_text(674, 468, f'80 flattened features  →  {n} components', 16, '#105c4d', '700', 'middle'),
              svg_text(674, 496, f'xₜ ∈ ℝ^{n}', 18, '#105c4d', '700', 'middle')]

    # Quantum cube and feedback
    parts += [svg_arrow(806, 410, 887, 410),
              '<path d="M930 266 L1465 266 L1535 324 L1535 553 L1000 553 L930 494 Z" fill="url(#quantum)" stroke="#23376d" stroke-width="3" filter="url(#shadow)"/>',
              '<path d="M930 266 L1000 324 L1535 324" fill="none" stroke="#a6c8ff" stroke-width="2"/>',
              '<path d="M1000 324 L1000 553" fill="none" stroke="#a6c8ff" stroke-width="2"/>',
              svg_text(1230, 310, 'TFIM QUANTUM RESERVOIR', 23, '#ffffff', '700', 'middle'),
              svg_text(1230, 340, f'{n}-qubit {meta["topology"]} • τ={meta["evolution_time"]} • {meta["trotter_steps"]} Trotter steps', 15, '#d4e5ff', '400', 'middle')]
    # chain within cube
    node_x = [1050 + i * (360 / max(1, n - 1)) for i in range(n)]
    for i in range(n - 1):
        parts.append(f'<line x1="{node_x[i]}" y1="415" x2="{node_x[i+1]}" y2="415" stroke="#a8e2ff" stroke-width="5"/>')
    for i, x in enumerate(node_x):
        parts += [f'<circle cx="{x}" cy="415" r="19" fill="#d9f4ff" stroke="#182f68" stroke-width="3"/>',
                  svg_text(x, 421, f'q{i}', 12, '#182f68', '700', 'middle')]
    parts += [svg_text(1230, 474, 'input encoding  →  coherent Ising evolution  →  observables', 16, '#ffffff', '700', 'middle'),
              svg_text(1230, 505, f"H = −J Σ ZᵢZᵢ₊₁ − h Σ Xᵢ     (J={meta['J']}, h={meta['h']})", 18, '#d4e5ff', '400', 'middle')]
    if meta["feedback"]:
        parts += [svg_arrow(1730, 710, 875, 710, '#8b4fc1', True, 3),
                  f'<path d="M875,710 L875,581 L930,581 L930,545" fill="none" stroke="#8b4fc1" stroke-width="3" marker-end="url(#arrow)" stroke-dasharray="8 7"/>',
                  svg_rect(1040, 684, 420, 55, '#f2e9ff', '#8b4fc1', 14, False),
                  svg_text(1250, 718, 'recurrent feedback:  θₜ = π · clip[½(xₜ + ⟨Z⟩ₜ₋₁)]', 15, '#63328e', '700', 'middle')]

    # Measured feature vector & readout
    parts += [svg_arrow(1545, 410, 1715, 410),
              svg_rect(1730, 258, 250, 310, '#eef4ff', '#5078c7', 20),
              svg_text(1855, 304, 'MEASUREMENT', 20, '#284d95', '700', 'middle'),
              svg_text(1855, 332, 'expectation values', 14, '#4568a8', '400', 'middle')]
    for i in range(5):
        yy = 365 + i * 32
        parts += [f'<rect x="1762" y="{yy}" width="45" height="22" rx="4" fill="#5d7fd0"/>',
                  f'<rect x="1811" y="{yy}" width="45" height="22" rx="4" fill="#67b8e3"/>',
                  svg_text(1785, yy+16, f'Z{i+1}', 10, '#fff', '700', 'middle'),
                  svg_text(1833, yy+16, f'X{i+1}', 10, '#fff', '700', 'middle')]
    parts += [svg_text(1855, 550, f'rₜ = [⟨Z₀…Zₙ₋₁⟩, ⟨X₀…Xₙ₋₁⟩] ∈ ℝ^{d}', 14, '#284d95', '700', 'middle'),
              svg_arrow(1990, 410, 2035, 410),
              svg_rect(2042, 260, 250, 308, 'url(#readout)', '#b6730e', 20),
              svg_text(2167, 307, 'LINEAR READOUT', 19, '#704000', '700', 'middle'),
              svg_text(2167, 335, meta['readout'], 14, '#704000', '400', 'middle'),
              f'<rect x="2083" y="365" width="87" height="125" fill="#fff9e8" stroke="#b6730e" stroke-width="2"/>',
              f'<rect x="2170" y="365" width="55" height="125" fill="#ffd777" stroke="#b6730e" stroke-width="2"/>',
              svg_text(2126, 430, f'W\n{d}×4', 14, '#704000', '700', 'middle'),
              svg_text(2198, 430, 'ŷ', 19, '#704000', '700', 'middle'),
              svg_text(2167, 526, 'ŷₜ₊ₕ = rₜW', 21, '#704000', '700', 'middle')]

    # Outputs
    for i, (target, color) in enumerate(zip(meta["targets"], colors)):
        y = 620 + i * 39
        parts += [svg_rect(2045, y, 245, 29, '#ffffff', color, 9, False),
                  f'<circle cx="2064" cy="{y+14}" r="7" fill="{color}"/>',
                  svg_text(2080, y+19, f'{target}  at t + {meta["horizon"]} h', 14, '#263f5b', '700')]

    # Expanded exact circuit panel
    parts += [f'<rect x="55" y="830" width="2280" height="535" rx="24" fill="#ffffff" stroke="#a8bdd5" stroke-width="2"/>',
              svg_text(85, 879, 'Expanded PennyLane circuit for one reservoir time step', 24, '#1d416a', '700'),
              svg_text(85, 907, f"θₜ is calculated before the QNode; Δt = τ / S = {meta['evolution_time']} / {meta['trotter_steps']} = {dt:g}", 15, '#5a738f')]
    # circuit wire rows, five representatives followed ellipsis
    wire_ys = [955, 1010, 1065, 1120, 1175]
    wire_labels = ["q₀", "q₁", "⋮", f"q{n-2}", f"q{n-1}"]
    for yy, label in zip(wire_ys, wire_labels):
        parts += [svg_text(108, yy+5, label, 16, '#314f70', '700', 'end'),
                  f'<line x1="128" y1="{yy}" x2="1770" y2="{yy}" stroke="#496887" stroke-width="2"/>']
    # encoding gates
    inject_x = 210
    for yy in wire_ys:
        parts.append(f'<rect x="{inject_x}" y="{yy-18}" width="70" height="36" rx="6" fill="#34a6da" stroke="#0b628b" stroke-width="2"/>')
        parts.append(svg_text(inject_x+35, yy+6, 'RY(θ)', 12, '#fff', '700', 'middle'))
    # evolution blocks
    if reupload:
        block_xs = [340, 665, 990, 1315]
        for k, bx in enumerate(block_xs, 1):
            parts += [f'<rect x="{bx}" y="930" width="275" height="275" rx="14" fill="#eef0ff" stroke="#5364c7" stroke-width="2"/>',
                      svg_text(bx+138, 955, f'Trotter block {k}', 15, '#34479d', '700', 'middle')]
            for yy in wire_ys:
                parts += [f'<rect x="{bx+18}" y="{yy-15}" width="53" height="30" rx="5" fill="#34a6da" stroke="#0b628b" stroke-width="1.5"/>',
                          svg_text(bx+44, yy+5, 'RY(θ)', 10, '#fff', '700', 'middle')]
            parts += [f'<rect x="{bx+95}" y="965" width="88" height="172" rx="10" fill="#5969d5" stroke="#25356f" stroke-width="2"/>',
                      svg_multiline(bx+139, 1014, ['ZZ chain', '−2JΔt'], 13, 23, '#fff', '700', 'middle'),
                      f'<rect x="{bx+198}" y="950" width="57" height="215" rx="10" fill="#7b87e8" stroke="#25356f" stroke-width="2"/>',
                      svg_multiline(bx+226, 1012, ['RX', '−2hΔt'], 12, 23, '#fff', '700', 'middle')]
    else:
        parts += [svg_text(247, 935, 'single injection', 12, '#0b628b', '700', 'middle'),
                  f'<rect x="335" y="930" width="1140" height="275" rx="14" fill="#eef0ff" stroke="#5364c7" stroke-width="2"/>']
        for k, bx in enumerate([385, 650, 915, 1180], 1):
            parts += [f'<rect x="{bx}" y="950" width="215" height="210" rx="12" fill="#5969d5" stroke="#25356f" stroke-width="2"/>',
                      svg_text(bx+108, 978, f'Trotter block {k}', 14, '#fff', '700', 'middle'),
                      svg_multiline(bx+108, 1040, ['IsingZZ(−2JΔt)', 'on each chain edge', 'then RX(−2hΔt)', 'on every qubit'], 13, 24, '#eef4ff', '400', 'middle')]
    # Measurements panel
    parts += [f'<rect x="1810" y="930" width="470" height="275" rx="14" fill="#eef7ff" stroke="#4c78ba" stroke-width="2"/>',
              svg_text(2045, 963, 'Two observable families', 18, '#254b8b', '700', 'middle')]
    for idx, (label, color) in enumerate([('⟨PauliZ(i)⟩', '#5d7fd0'), ('⟨PauliX(i)⟩', '#35a5cc')]):
        yy = 1008 + idx * 78
        parts += [f'<rect x="1860" y="{yy}" width="185" height="48" rx="9" fill="{color}"/>',
                  svg_text(1952, yy+31, label, 18, '#fff', '700', 'middle'),
                  svg_text(2080, yy+30, f'for i = 0 … {n-1}', 15, '#254b8b')]
    parts += [svg_text(2045, 1172, f'concatenate → {d}-dimensional reservoir feature rₜ', 16, '#254b8b', '700', 'middle'),
              svg_text(85, 1278, 'Exact implemented operations:  qml.RY(θᵢ)  •  qml.IsingZZ(−2JΔt, [i,i+1])  •  qml.RX(−2hΔt)  •  qml.expval(PauliZ/PauliX)', 16, '#365573', '700'),
              svg_text(85, 1311, 'Simulation primary: p=0 and exact expectation values. Finite-shot and noise configurations are separate robustness ablations.', 14, '#5a738f'),
              svg_text(2310, 1330, 'Generated from project artifacts • editable source included', 13, '#5a738f', '400', 'end'),
'</svg>']
    return "\n".join(parts)


class Drawio:
    def __init__(self):
        self.cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
        self.idx = 2

    def cell(self, value, x, y, w, h, style, parent="1"):
        ident = f"c{self.idx}"; self.idx += 1
        self.cells.append(f'<mxCell id="{ident}" value="{esc(value)}" style="{style}" vertex="1" parent="{parent}"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return ident

    def edge(self, source, target, style="endArrow=block;html=1;strokeWidth=2;strokeColor=#334d72;"):
        ident = f"e{self.idx}"; self.idx += 1
        self.cells.append(f'<mxCell id="{ident}" value="" style="{style}" edge="1" parent="1" source="{source}" target="{target}"><mxGeometry relative="1" as="geometry"/></mxCell>')

    def xml(self):
        return ('<mxfile host="app.diagrams.net" agent="ResQue architecture generator" version="24.0.0">'
                '<diagram id="resque_model_architecture" name="ResQue Model Architecture">'
                '<mxGraphModel dx="2400" dy="1420" grid="1" gridSize="10" page="1" pageScale="1" pageWidth="2400" pageHeight="1420" math="0" shadow="0"><root>'
                + ''.join(self.cells) + '</root></mxGraphModel></diagram></mxfile>')


def render_drawio(meta: dict[str, Any]) -> str:
    """Native diagrams.net approximation of the SVG: all elements stay editable."""
    d = Drawio()
    title = d.cell(f"ResQue | Horizon {meta['horizon']} h deployed QRC architecture", 30, 20, 2300, 50,
                   "rounded=1;whiteSpace=wrap;html=1;fillColor=#10223c;strokeColor=#10223c;fontColor=#ffffff;fontSize=28;fontStyle=1;align=center;")
    _ = title
    labels = [(60, 105, 395, 620, "A. TEMPORAL WEATHER INPUT", "#e8f7f2", "#218a74"),
              (485, 105, 315, 620, "B. TRAIN-ONLY PCA", "#e7f5f7", "#16876d"),
              (830, 105, 800, 620, "C. RECURRENT QUANTUM RESERVOIR", "#eef0ff", "#5364c7"),
              (1660, 105, 650, 620, "D. MEASURED STATE AND FORECAST", "#fff5de", "#b6730e")]
    for x, y, w, h, label, fill, stroke in labels:
        d.cell(label, x, y, w, h, f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};fillOpacity=45;strokeColor={stroke};dashed=1;fontStyle=1;fontSize=17;fontColor=#1d416a;verticalAlign=top;align=center;")
    input_cell = d.cell(f"20 × 4 history window\nT • RH • SLP • WS\nnormalised weather traces", 95, 275, 305, 180,
                        "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#218a74;fontSize=19;fontStyle=1;align=center;verticalAlign=middle;")
    pca = d.cell(f"PCA\n80 → {meta['n_qubits']} components\nxₜ ∈ R^{meta['n_qubits']}", 535, 295, 220, 135,
                 "shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fillColor=#b8eadb;strokeColor=#16876d;fontSize=18;fontStyle=1;align=center;verticalAlign=middle;")
    cube = d.cell(f"{meta['n_qubits']}-qubit TFIM reservoir\n{meta['topology']} topology | J={meta['J']} | h={meta['h']}\n{meta['trotter_steps']} Trotter steps | {'reupload' if meta['use_data_reuploading'] else 'single injection'}",
                  930, 270, 565, 225,
                  "shape=cube;whiteSpace=wrap;html=1;fillColor=#5969d5;strokeColor=#25356f;fontColor=#ffffff;fontSize=20;fontStyle=1;align=center;verticalAlign=middle;")
    measure = d.cell(f"EXPECTATION MEASUREMENTS\n[⟨Z₀…Zₙ₋₁⟩, ⟨X₀…Xₙ₋₁⟩]\nrₜ ∈ R^{meta['feature_dim']}", 1710, 280, 245, 170,
                     "rounded=1;whiteSpace=wrap;html=1;fillColor=#e9f1ff;strokeColor=#5078c7;fontSize=18;fontStyle=1;align=center;verticalAlign=middle;")
    readout = d.cell(f"{meta['readout']}\nW: {meta['feature_dim']} × 4\nŷₜ₊ₕ = rₜW", 2040, 280, 220, 170,
                     "shape=cube;whiteSpace=wrap;html=1;fillColor=#ffd777;strokeColor=#b6730e;fontSize=18;fontStyle=1;align=center;verticalAlign=middle;")
    outputs = d.cell(f"Four forecasts at t + {meta['horizon']} h\ntemperature • humidity\npressure • wind speed", 2025, 545, 260, 95,
                     "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#ed6a5a;fontSize=17;fontStyle=1;align=center;verticalAlign=middle;")
    d.edge(input_cell, pca); d.edge(pca, cube); d.edge(cube, measure); d.edge(measure, readout); d.edge(readout, outputs)
    feedback = d.cell("feedback: θₜ = π clip[½(xₜ + ⟨Z⟩ₜ₋₁)]", 1010, 635, 430, 42,
                      "rounded=1;whiteSpace=wrap;html=1;fillColor=#f2e9ff;strokeColor=#8b4fc1;fontColor=#63328e;fontStyle=1;fontSize=15;align=center;")
    if meta["feedback"]:
        d.edge(measure, feedback, "endArrow=block;html=1;strokeWidth=2;strokeColor=#8b4fc1;dashed=1;")
        d.edge(feedback, cube, "endArrow=block;html=1;strokeWidth=2;strokeColor=#8b4fc1;dashed=1;")
    d.cell("EXPANDED PENNYLANE QNODE — one reservoir time step", 60, 790, 2250, 48,
           "rounded=1;whiteSpace=wrap;html=1;fillColor=#1d416a;strokeColor=#1d416a;fontColor=#ffffff;fontSize=22;fontStyle=1;align=center;")
    d.cell(f"θᵢ = π·clip(xᵢ)  ({'re-encoded before each block' if meta['use_data_reuploading'] else 'encoded once before free evolution'})\nΔt = τ/S = {meta['evolution_time']}/{meta['trotter_steps']}", 90, 860, 470, 88,
           "rounded=1;whiteSpace=wrap;html=1;fillColor=#e6faf4;strokeColor=#218a74;fontSize=17;align=center;verticalAlign=middle;")
    start_x = 610
    for k in range(meta["trotter_steps"]):
        block = d.cell(f"Trotter block {k+1}\n{'RY(θᵢ) → ' if meta['use_data_reuploading'] else ''}IsingZZ(−2JΔt) on chain edges\nRX(−2hΔt) on all qubits", start_x + k * 275, 860, 240, 145,
                       "rounded=1;whiteSpace=wrap;html=1;fillColor=#eef0ff;strokeColor=#5364c7;fontSize=16;fontStyle=1;align=center;verticalAlign=middle;")
        if k:
            d.edge(prev, block)
        prev = block
    measurements = d.cell("qml.expval(PauliZ(i)) for all i\n+ qml.expval(PauliX(i)) for all i\nconcatenate → reservoir state", 1745, 860, 430, 145,
                          "rounded=1;whiteSpace=wrap;html=1;fillColor=#e9f1ff;strokeColor=#5078c7;fontSize=17;fontStyle=1;align=center;verticalAlign=middle;")
    d.edge(prev, measurements)
    d.cell("Exact primary simulator configuration: p=0; analytic expectation values. Hardware/noise/finite-shot runs are separate ablations.", 110, 1080, 2100, 50,
           "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8fbff;strokeColor=#9db2c9;fontColor=#365573;fontSize=16;align=center;verticalAlign=middle;")
    d.cell(f"Artifact source: {meta['architecture_source']}", 110, 1190, 2100, 36,
           "text;html=1;align=center;verticalAlign=middle;fontSize=13;fontColor=#5a738f;")
    return d.xml()


def render_spec(meta: dict[str, Any]) -> str:
    encoding_loop = ("for _ in range(trotter_steps):\n        _encode(angles)\n        _evolve(dt)"
                     if meta["use_data_reuploading"] else
                     "_encode(angles)\n    for _ in range(trotter_steps):\n        _evolve(dt)")
    return f'''# ResQue architecture specification — horizon {meta['horizon']} h

This companion file gives the exact gate-model circuit depicted in the figure.
The values below were read from: **{meta['architecture_source']}**.

| Parameter | Value |
|---|---:|
| Qubits | {meta['n_qubits']} |
| Topology | {meta['topology']} |
| J | {meta['J']} |
| h | {meta['h']} |
| Trotter steps | {meta['trotter_steps']} |
| Evolution time | {meta['evolution_time']} |
| Encoding | {"data reuploading" if meta['use_data_reuploading'] else "standard single injection"} |
| Feedback | {meta['feedback']} |
| Readout features | {meta['feature_dim']} |
| Readout | {meta['readout']} |

```python
def _encode(angles):
    for i in range(n):
        qml.RY(float(angles[i]), wires=i)

def _evolve(dt):
    for (i, j) in [(i, i + 1) for i in range(n - 1)]:
        qml.IsingZZ(-2 * J * dt, wires=[i, j])
    for i in range(n):
        qml.RX(-2 * h * dt, wires=i)

@qml.qnode(dev, diff_method=None)
def circuit(angles):
    dt = evolution_time / trotter_steps
    {encoding_loop}
    return ([qml.expval(qml.PauliZ(i)) for i in range(n)] +
            [qml.expval(qml.PauliX(i)) for i in range(n)])
```

With feedback enabled, the input angles at time *t* are
`theta_t = clip(pi * 0.5 * (x_t + z_{{t-1}}), -pi, pi)`, where `z_{{t-1}}` is the
previous all-qubit Z expectation vector. At the first step, no feedback is
available and the projected input is used directly.
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate editable ResQue architecture figure")
    parser.add_argument("--horizon", type=int, default=6, help="Forecast horizon whose saved architecture is shown")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs" / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "figures" / "architecture")
    args = parser.parse_args()
    meta = model_metadata(args.results_dir, args.horizon)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"resque_architecture_h{args.horizon}"
    (args.output_dir / f"{stem}.svg").write_text(render_svg(meta), encoding="utf-8")
    (args.output_dir / f"{stem}.drawio").write_text(render_drawio(meta), encoding="utf-8")
    (args.output_dir / f"{stem}_spec.md").write_text(render_spec(meta), encoding="utf-8")
    (args.output_dir / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Architecture figure written to {args.output_dir}")


if __name__ == "__main__":
    main()
