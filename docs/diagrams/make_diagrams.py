"""Generate the study's draw.io diagrams from Python specs (editable output, no hand XML)."""
import pathlib
import xml.etree.ElementTree as ET

OUT = pathlib.Path("/home/user/qc_native_preference_tuning/docs/diagrams")

BOX = "rounded=1;whiteSpace=wrap;html=1;fontSize=12;"
NOTE = "text;html=1;whiteSpace=wrap;align=left;verticalAlign=top;fontSize=11;"
LANE = "swimlane;whiteSpace=wrap;html=1;fontSize=13;fontStyle=1;startSize=28;"
EDGE = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;fontSize=11;"
DASH = EDGE + "dashed=1;"

FILL = {
    "config": "fillColor=#fff2cc;strokeColor=#d6b656;",
    "data": "fillColor=#dae8fc;strokeColor=#6c8ebf;",
    "gate": "fillColor=#f8cecc;strokeColor=#b85450;",
    "train": "fillColor=#d5e8d4;strokeColor=#82b366;",
    "eval": "fillColor=#e1d5e7;strokeColor=#9673a6;",
    "track": "fillColor=#f5f5f5;strokeColor=#666666;",
    "measure": "fillColor=#dae8fc;strokeColor=#6c8ebf;",
    "pinned": "fillColor=#d5e8d4;strokeColor=#82b366;",
    "unavoid": "fillColor=#f8cecc;strokeColor=#b85450;",
    "outcome": "fillColor=#ffe6cc;strokeColor=#d79b00;",
    "plain": "fillColor=#ffffff;strokeColor=#000000;",
}


class Diagram:
    def __init__(self, name):
        self.name = name
        self.cells = []
        self.n = 1

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def box(self, text, x, y, w=160, h=60, kind="plain", style=BOX, parent="1"):
        cid = self._id()
        self.cells.append(("v", cid, text, style + FILL.get(kind, ""), x, y, w, h, parent))
        return cid

    def note(self, text, x, y, w=220, h=80, parent="1"):
        cid = self._id()
        self.cells.append(("v", cid, text, NOTE, x, y, w, h, parent))
        return cid

    def lane(self, text, x, y, w, h, kind="track"):
        cid = self._id()
        self.cells.append(("v", cid, text, LANE + FILL.get(kind, ""), x, y, w, h, "1"))
        return cid

    def edge(self, src, dst, label="", style=EDGE, points=None, exit=None, entry=None):
        cid = self._id()
        extra = ""
        if exit:
            extra += f"exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;"
        if entry:
            extra += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        self.cells.append(("e", cid, label, style + extra, src, dst, points or []))
        return cid

    def to_xml(self):
        mxfile = ET.Element("mxfile", host="qc_native_preference_tuning")
        diagram = ET.SubElement(mxfile, "diagram", name=self.name, id=self.name)
        model = ET.SubElement(diagram, "mxGraphModel", dx="1200", dy="800", grid="1", gridSize="10",
                              guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1",
                              pageScale="1", pageWidth="1400", pageHeight="1000")
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", id="0")
        ET.SubElement(root, "mxCell", id="1", parent="0")
        for c in self.cells:
            if c[0] == "v":
                _, cid, text, style, x, y, w, h, parent = c
                cell = ET.SubElement(root, "mxCell", id=cid, value=text, style=style, vertex="1", parent=parent)
                ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h)).set("as", "geometry")
            else:
                _, cid, label, style, src, dst, points = c
                cell = ET.SubElement(root, "mxCell", id=cid, value=label, style=style, edge="1", parent="1",
                                     source=src, target=dst)
                geo = ET.SubElement(cell, "mxGeometry", relative="1")
                geo.set("as", "geometry")
                if points:
                    arr = ET.SubElement(geo, "Array")
                    arr.set("as", "points")
                    for (px, py) in points:
                        ET.SubElement(arr, "mxPoint", x=str(px), y=str(py))
        return ET.tostring(mxfile, encoding="unicode")


def pipeline_flow():
    d = Diagram("pipeline_flow")
    X = 80  # left margin reserved for cross-lane routing
    d.lane("Configs (experiments are configs)", X, 20, 1340, 120, "config")
    lang = d.box("configs/language/{code}.yaml<br>code, ISO 639-3, FLORES code, register", X + 20, 60, 260, 60, "config")
    bench = d.box("configs/benchmark/{name}.yaml<br>sources, fields, scorer, blueprint", X + 300, 60, 260, 60, "config")
    series = d.box("configs/series/{series}.yaml<br>base model, conditions, seeds, constants", X + 580, 60, 280, 60, "config")
    plan = d.box("pipeline/plan.py<br>expands the matrix, prints stage commands", X + 900, 60, 300, 60, "config")
    d.edge(lang, plan, exit=(1, 0.5), entry=(0, 0.5), points=[(X + 290, 90)])
    d.edge(bench, plan, exit=(1, 0.5), entry=(0, 0.5))
    d.edge(series, plan, exit=(1, 0.5), entry=(0, 0.5))

    d.lane("Stage 1: datagen (per benchmark x language)", X, 170, 1340, 200, "data")
    src = d.box("English source on HF Hub<br>(GSM8K, MedQA, ...)", X + 20, 220, 200, 60, "data")
    sel = d.box("Fixed item selection<br>same source ids for every language", X + 250, 220, 220, 60, "data")
    tr = d.box("Adaption Adaptive Data<br>translation blueprint + fixed register", X + 500, 220, 240, 60, "data")
    gate = d.box("Quality gate<br>LID, preservation checks, NLLB agreement,<br>QE, human ESA sample", X + 770, 210, 260, 80, "gate")
    hf = d.box("HF dataset sanggusti/{benchmark}-{language}<br>splits train and test, provenance columns", X + 1060, 220, 260, 60, "data")
    d.edge(src, sel); d.edge(sel, tr); d.edge(tr, gate); d.edge(gate, hf, "pass")
    d.note("fail: regenerate once, then NLLB-200 retranslation,<br>then drop the item id in every language", X + 770, 300, 320, 50)

    d.lane("Stage 1b: covariates (per language, before any paid run)", X, 400, 1340, 110, "track")
    fert = d.box("Tokenizer fertility<br>base-model tokenizer on the eval split", X + 20, 440, 260, 50, "track")
    bpb = d.box("Base-model bits per byte<br>FLORES-200 devtest", X + 310, 440, 240, 50, "track")
    tq = d.box("Translation quality<br>gate metrics and human error rate", X + 580, 440, 260, 50, "track")
    base = d.box("Untuned base evals<br>B(L) in every language", X + 870, 440, 220, 50, "track")
    gonogo = d.box("Go / no-go per language", X + 1120, 440, 200, 50, "gate")
    d.edge(base, gonogo)
    d.edge(fert, gonogo, exit=(1, 0.5), entry=(0, 0.5)); d.edge(bpb, gonogo, exit=(1, 0.5), entry=(0, 0.5)); d.edge(tq, gonogo, exit=(1, 0.5), entry=(0, 0.5))

    d.lane("Stage 2: finetune (per benchmark x base x train language x replicate)", X, 540, 1340, 150, "train")
    raw = d.box("Upload train split as raw rows<br>processing_mode=raw, explicit column mapping", X + 20, 590, 300, 60, "train")
    auto = d.box("Adaption AutoScientist<br>pinned model and hyperparameters,<br>augmentation 0, 3 replicates", X + 360, 580, 280, 80, "train")
    ckpt = d.box("Checkpoint on HF Hub<br>sanggusti/{benchmark}-{train}-{series}-{base}-r{n}", X + 680, 590, 320, 60, "train")
    wb1 = d.box("wandb run<br>resolved config, best_hyperparams, tokens", X + 1040, 590, 280, 60, "track")
    d.edge(raw, auto); d.edge(auto, ckpt); d.edge(ckpt, wb1, "", DASH)

    d.lane("Stage 3: evaluate and analyze (per cell)", X, 720, 1340, 200, "eval")
    task = d.box("Inspect task translated_benchmark<br>-T benchmark -T language", X + 20, 770, 260, 60, "eval")
    scorer = d.box("Language-agnostic scorer<br>numeric match or choice letter,<br>literal ANSWER marker", X + 310, 760, 260, 80, "eval")
    logs = d.box("Eval logs<br>source_id, language, condition, replicate,<br>output tokens, output-language id", X + 600, 760, 280, 80, "eval")
    ana = d.box("Analysis<br>paired item bootstrap, GLMM,<br>Holm on the primary family", X + 910, 760, 240, 80, "outcome")
    rep = d.box("Report + docs/experiments.md", X + 1180, 770, 150, 60, "outcome")
    d.edge(task, scorer); d.edge(scorer, logs); d.edge(logs, ana); d.edge(ana, rep)
    d.note("Model under eval: base model (condition base) or hf/{checkpoint}; temperature 0, generous max tokens", X + 20, 850, 600, 40)

    # cross-lane edges routed through the left margin at distinct x positions
    d.edge(plan, src, "commands", DASH, exit=(0.5, 1), entry=(0.5, 0), points=[(X + 1050, 155), (X + 120, 155)])
    d.edge(hf, raw, "train split", DASH, exit=(0.5, 1), entry=(0, 0.5), points=[(X + 1190, 385), (30, 385), (30, 620)])
    d.edge(hf, task, "test split", DASH, exit=(0.5, 1), entry=(0, 0.5), points=[(X + 1190, 385), (15, 385), (15, 800)])
    d.edge(ckpt, task, "checkpoint", DASH, exit=(0.5, 1), entry=(0, 0.35), points=[(X + 840, 705), (45, 705), (45, 791)])
    d.edge(base, task, "", DASH, exit=(0.5, 1), entry=(0, 0.65), points=[(X + 980, 525), (60, 525), (60, 809)])
    return d


def design():
    d = Diagram("language_medium_design")
    langs = ["en", "id", "jv", "su", "min", "ace"]
    d.note("<b>Experiment matrix, series S01.</b> Rows: what the evaluated model was finetuned on. Columns: language of the evaluation items. Every cell uses the same source items translated per language, the same pinned finetuning recipe and a language-agnostic scorer. Tier 0 rows are the minimum publishable unit; tier 1 repeats native and english_anchor on a size-matched contrast base model (gsm8k only); tier 2 adds the pooled and Indonesian-anchor arms.", 20, 20, 1000, 80)
    x0, y0, cw, ch = 300, 120, 130, 58
    d.box("finetune condition \\ eval language", 20, y0, 270, ch, "config")
    for j, code in enumerate(langs):
        d.box(code, x0 + j * cw, y0, cw - 10, ch, "config")
    rows = [
        ("tier 0  base: no finetuning", "B(L)", ["all"], "track"),
        ("tier 0  native: finetuned in L", "A(L)", ["diag"], "train"),
        ("tier 0  english_anchor: finetuned in en", "A_en(L)", ["all"], "eval"),
        ("tier 0  regression: finetuned in L, scored in en", "R(L)", ["en"], "gate"),
        ("tier 0  round_trip: en model on L back-translated to en", "RT(L)", ["jv", "su", "min", "ace", "id"], "gate"),
        ("tier 2  indonesian_anchor: finetuned in id", "A_id(L)", ["jv", "su", "min", "ace"], "eval"),
        ("tier 2  pooled: one model on all languages", "A_all(L)", ["all"], "outcome"),
        ("tier 1  native + english_anchor on the contrast base", "A^B(L)", ["all"], "measure"),
    ]
    for i, (name, est, cells, kind) in enumerate(rows):
        y = y0 + (i + 1) * ch
        d.box(name, 20, y, 270, ch - 5, kind)
        for j, code in enumerate(langs):
            filled = cells == ["all"] or cells == ["diag"] or code in cells
            label = est if filled else ""
            if cells == ["diag"]:
                label = f"A({code})"
            if cells == ["en"] and code == "en":
                label = "R(id..ace)"
            d.box(label, x0 + j * cw, y, cw - 10, ch - 5, kind if label else "plain")
    y = y0 + 9 * ch + 20
    d.note("<b>Estimands per language L</b><br>B(L) base level; A(L) native accuracy; G(L) = A(L) - B(L) gain from native tuning; Delta_en(L) = A_en(L) - A(L) anchor advantage;<br>Reg(L) English regression; A(en) - RT(L) translator-loss bound; tau(L, L') = A(L) - A(L') the descriptive cross-language contrast.", 20, y, 560, 90)
    d.note("<b>Replication and inference</b><br>3 replicate finetunes per model (count confirmed by a variance pilot); items paired across languages by source id;<br>paired item bootstrap and a mixed-effects logistic model; Holm on the primary family.", 600, y, 480, 90)
    d.note("<b>Covariates reported next to every cell</b><br>base-model bits per byte on FLORES-200, tokenizer fertility per item, translation quality columns per item, Indonesian leakage, output-language fidelity, output tokens and cap hits.", 20, y + 100, 560, 80)
    d.note("<b>Adding a language</b><br>one file configs/language/{code}.yaml plus the code in the series list; the matrix, names and commands follow from pipeline/plan.py.", 600, y + 100, 480, 80)
    return d


def confounds():
    d = Diagram("confound_structure")
    d.note("<b>What stands between 'the language' and the accuracy we measure.</b> Green: pinned by design. Blue: measured and reported as a covariate. Red: cannot be removed, only made visible. Arrows point from cause to effect.", 20, 20, 1100, 50)
    L = d.box("Language of the medium L<br>(treatment)", 40, 300, 220, 70, "outcome")
    acc = d.box("Measured accuracy A(L)", 1060, 420, 220, 70, "outcome")
    col = 420
    exp = d.box("Base-model pretraining exposure X(L)", col, 100, 320, 60, "unavoid")
    tq = d.box("Translator competence in L<br>-> training data quality and eval item validity", col, 190, 320, 60, "measure")
    fert = d.box("Tokenizer fertility<br>-> tokens per item, truncation, cap effects", col, 280, 320, 60, "measure")
    reg = d.box("Register and orthography choice<br>(ngoko, loma, dialect)", col, 370, 320, 60, "pinned")
    leak = d.box("Indonesian leakage in outputs<br>(Minangkabau, Banjar)", col, 460, 320, 60, "measure")
    cont = d.box("Contamination asymmetry<br>(English test items seen in pretraining)", col, 580, 320, 60, "measure")
    scorer = d.box("Scorer language dependence<br>(numeric match, choice letter only)", col, 670, 320, 60, "pinned")
    opt = d.box("AutoScientist optimization and judge<br>(pinned hyperparameters, no early stop)", col, 760, 320, 60, "pinned")
    seed = d.box("Run-to-run variance<br>(3 replicates, run random effect)", col, 850, 320, 60, "pinned")
    for src in (exp, tq, fert, reg, leak):
        d.edge(L, src, exit=(1, 0.5), entry=(0, 0.5), points=[(340, 335)])
    for src in (exp, tq, fert, reg, leak, cont, scorer, opt, seed):
        d.edge(src, acc, exit=(1, 0.5), entry=(0, 0.5), points=[(900, 455)])
    d.edge(exp, tq, "", DASH, exit=(0.15, 1), entry=(0.15, 0))
    d.note("X(L) cannot be removed by design, only measured. Residual language effect = variation in A(L) that survives adjustment for X(L), fertility and translation quality. With five or six languages this is a descriptive quantity; the paired item design supports claims about cells, not about languages in general.", 40, 940, 900, 70)
    return d


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for build in (pipeline_flow, design, confounds):
        dia = build()
        path = OUT / f"{dia.name}.drawio"
        path.write_text(dia.to_xml())
        print("wrote", path)
