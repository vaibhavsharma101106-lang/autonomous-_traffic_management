"""Build the editable and PDF IT 207 submission from project artifacts."""

from pathlib import Path
import csv
import math
import textwrap
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor as PptColor
from pptx.util import Inches as PptInches, Pt as PptPt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image as PdfImage, PageBreak,
                                KeepTogether)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission"
EXP = OUT / "experiment_results"
NAVY = (20, 45, 76)
TEAL = (0, 132, 145)
INK = (34, 43, 52)
MUTED = (93, 108, 122)
LIGHT = (237, 243, 247)
CONTROLLERS = ["fixed", "rule_based", "ai_assisted"]
CONTROLLER_LABELS = {"fixed": "Fixed-time", "rule_based": "Rule-based", "ai_assisted": "AI-assisted"}


REPORT = [
    ("Chapter 1 Introduction", [
        "Urban intersections experience uneven demand, and a fixed allocation can leave a busy approach waiting while a quiet approach receives green time. This project continues an existing Python prototype that uses video-based perception and software signal simulation to explore that problem. Its purpose is to connect selected human cognition ideas to measurable AI components, build a reproducible software demonstration, and assess adaptive control against a fixed-time baseline.",
        "The project scope is software-only. Recorded video is the intended input; no physical sensors, embedded devices, or live traffic-light control are used. The supplied workspace includes detection, tracking, analysis, prediction, optimization, a FastAPI backend, and a Streamlit dashboard. At audit time it did not include a traffic video, labeled data, model weights, trained predictors, or prior experiment results. Consequently, this report distinguishes implemented code from features that could not be executed in this environment.",
        "Objectives are to inspect and preserve the existing architecture; repair the fallback signal simulation and experiment comparison; retain the YOLO and ByteTrack video pipeline; implement lane metrics and short-term prediction interfaces; compare three signal controllers; and document ethical and technical limitations. Experimental results in this submission come only from the seeded software simulator, not from road observations or labeled video."
    ]),
    ("Chapter 2 Human Intelligence Theory", [
        "The design uses human cognition as an engineering analogy, not as a claim that the software reproduces human intelligence. Selective attention is represented by configurable lane and region-of-interest polygons. The detector processes vehicle classes of interest and the analyser aggregates vehicles by lane, focusing computation on relevant parts of a frame. This is a task-specific filter; it can miss vehicles outside incorrectly configured regions.",
        "Working memory is represented by a bounded history buffer and persistent SQLite observations. Recent lane counts, queues, and traffic states provide context for prediction and control. The buffer is an external software record, not a psychological measurement. Learning is represented by the Random Forest, XGBoost, and LSTM training code and lagged traffic features. No trained model or validated prediction result was supplied, so the learning component remains an implementation capability rather than an experimentally established result.",
        "Decision-making is represented by controller logic that assigns green duration from demand, queue, predicted count, waiting urgency, and fairness weights, subject to configured bounds. The fixed-time controller provides a baseline; the rule-based and weighted adaptive controllers provide alternatives. These mappings are inspired by selective attention, working memory, learning, and decision-making theories. They are computational proxies only and do not measure people or establish biological equivalence."
    ]),
    ("Chapter 3 Literature Review", [
        "Single-stage YOLO detection frames object localization and classification as one neural-network inference pass [4]. The project config selects Ultralytics YOLO11n and COCO vehicle classes (car, motorcycle, bus, truck). The model weights are not present in the workspace, and detector performance was therefore not measured. ByteTrack associates high- and lower-confidence detections with tracklets to preserve identities through partial occlusion [5]. The project wraps the Ultralytics tracking interface, but there is no annotated video or ground-truth track file for IDF1 or MOTA evaluation.",
        "Traffic signal research commonly compares fixed timing with demand-responsive control using simulation. SUMO is an open-source microscopic traffic simulator with interfaces for traffic-light control [6]. Although this project contains a SUMO integration module, SUMO and TraCI were not installed, and the generated network does not contain a verified signal phase plan. The included experiment therefore uses the local queue simulator and is labelled accordingly.",
        "The additional repository by Prit2341 describes density-based green allocation and coordinating consecutive signals, and its README claims a 35 percent throughput improvement [7]. The claim is reported by that repository and is not reproduced here. No code was copied. The repository tree did not expose a license file during inspection, so only high-level ideas were considered. A remaining research gap for this project is evaluating corridor coordination on a valid multi-intersection network with calibrated demand, turning ratios, startup delays, and measured travel time."
    ]),
    ("Chapter 4 Proposed Methodology", [
        "The intended video path is: recorded video frame; YOLO detection; ByteTrack association; ROI-based lane assignment; lane counts and class totals; density, queue, and state estimates; database history; optional forecasts; then signal-duration recommendations. The dashboard exposes video analysis, lane monitoring, prediction, signal optimization, and experiment views. SQLite persists observations and decisions. A detailed component diagram is included as architecture.png.",
        "The controller experiment uses four approaches and three controllers: fixed-time with 30 seconds of green per approach, rule-based demand scoring, and a weighted AI-assisted controller. Each scenario runs for 1,800 simulation seconds using seeded Poisson vehicle arrivals. The low, medium, and heavy scenarios split their configured total flow equally among approaches. The uneven scenario uses 55, 25, 15, and 5 percent shares. Each controller receives the same random seed and approach arrival rates within a scenario. Saturation flow is set to 0.5 vehicles per green second, and an assumed 5 metres per queued vehicle converts queue counts to an indicative distance.",
        "Reported outputs are mean simulated queue delay per completed departure, mean queue length in metres per approach, throughput in vehicles per hour, and fraction of lane-seconds above the configured moderate threshold. These are outputs of a simplified queue model. They are not travel-time measurements, physical queue observations, or SUMO measurements. Each scenario-controller pair is a single seeded run; uncertainty and multi-seed confidence intervals are not estimated."
    ]),
    ("Chapter 5 Implementation", [
        "The existing source tree was retained. Core modules are src/detection/detector.py, src/tracking/tracker.py, src/analysis/traffic_analyser.py, src/prediction/predictor.py, src/optimization/signal_controller.py, and src/simulation/sumo_runner.py. FastAPI endpoints are in backend/main.py and the seven-page Streamlit dashboard is in dashboard/app.py. Configuration and thresholds are in config.yaml. Detection uses the configured YOLO11n name, confidence threshold, and class IDs; tracking uses ByteTrack; the analyser estimates queue length as five metres per vehicle and stores recent history.",
        "The simulator previously treated total cycle count as equal experimental duration, accumulated waiting time repeatedly, and used the same fallback arrival rate for every scenario. It now advances each approach phase over seeded one-second arrivals, tracks queued vehicle-seconds, uses a common simulated time horizon for controller comparisons, respects the configured scenario rates, and writes an explicitly labelled software_simulation result table. The utility module now imports OpenCV only in video-specific functions, allowing non-video simulation modules to run when OpenCV is unavailable.",
        "The present runtime had Python 3.14 and lacked the project's primary dependencies, video data, model files, and SUMO. A bundled Python 3.12 runtime with NumPy, Pandas, PyYAML, and Loguru was used for the simulator and tests; OpenCV, PyTorch, Ultralytics, Streamlit, and FastAPI were unavailable. The project requirements file specifies Python 3.11. Full dashboard, video, API, prediction-model, and SUMO execution was not verified here."
    ]),
    ("Chapter 6 Results and Evaluation", [
        "Table 1 summarizes the measured software simulation results. Values are means from one deterministic seeded run per scenario and controller. Throughput is normalized to vehicles per hour. Queue length is averaged per approach and multiplied by the assumed 5 metres per vehicle. A lower queue delay is preferable; higher throughput is preferable, subject to the limitations of the queue model.",
        "In low traffic, the AI-assisted controller produced 50.22 seconds mean queue delay and 224 vehicles per hour, compared with 43.50 seconds and 218 vehicles per hour for fixed-time. In medium traffic, AI-assisted produced 64.12 seconds and 598 vehicles per hour, versus 42.86 seconds and 606 vehicles per hour for fixed-time. In heavy traffic, AI-assisted produced 103.87 seconds and 1,168 vehicles per hour, versus 45.01 seconds and 1,208 vehicles per hour for fixed-time. In uneven demand, AI-assisted produced 37.19 seconds and 724 vehicles per hour, versus 52.62 seconds and 686 vehicles per hour for fixed-time. Thus the proposed controller did better on both reported measures in the uneven scenario and worse on both measures in the medium and heavy scenarios. The results do not support a general congestion-reduction claim.",
        "Rule-based and AI-assisted scores coincide in balanced cases when their inputs and resulting allocations are equal or similar. Differences in the uneven scenario reflect adaptation to lane demand and waiting urgency. The software tests confirm controller bounds, repeatability for a fixed seed, duration accounting, and arrival/departure consistency. There are no labeled video results, so precision, recall, mAP, IDF1, MOTA, classification F1, forecast MAE, and forecast RMSE are not available."
    ]),
    ("Chapter 7 Ethics Conclusion and Future Work", [
        "Recorded traffic footage can contain identifiable people, plates, and travel patterns. A deployment should minimize retention, restrict access, blur identifying details where practical, and document consent and lawful purpose. Detection errors may vary with camera angle, weather, lighting, vehicle size, and local vehicle mix. Miscounts can shift green allocation unfairly; the controller should enforce minimum service and waiting-time limits and expose its recommendations for human review.",
        "The current prototype is not validated for public-road control. Queue length and speed are approximate without camera calibration. The simulator omits pedestrians, turning conflict, startup loss, lane changing, spillback, network geometry, and calibrated distributions. The SUMO route/network integration has not been validated. In particular, a safe signal phase plan and conflict-free movements must be confirmed before treating SUMO results as evidence. The project does not connect to or command physical equipment.",
        "The project demonstrates a coherent software architecture and a reproducible local controller experiment, but it does not establish that the AI-assisted policy is superior in general. The next steps are to install the supported Python runtime and dependencies, provide licensed video and annotations, validate model paths, measure detection and tracking metrics, train forecasting models on time-ordered real data, run repeated scenarios with confidence intervals, and validate multi-intersection coordination in SUMO."
    ])
]

REFERENCES = [
    "[1] IT 207 Human Intelligence Project Details. Supplied course authority PDF, 6 pages. Seven-chapter outline, assessment rubric, and recommended submission package.",
    "[2] Broadbent, D. E. (1958). Perception and Communication. Pergamon Press.",
    "[3] Baddeley, A. (2000). The episodic buffer: a new component of working memory? Trends in Cognitive Sciences, 4(11), 417-423. https://doi.org/10.1016/S1364-6613(00)01538-2",
    "[4] Redmon, J., Divvala, S., Girshick, R., and Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. CVPR, 779-788. https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html",
    "[5] Zhang, Y. et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV. https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf",
    "[6] Eclipse SUMO documentation, SUMO at a Glance. https://sumo.dlr.de/docs/SUMO_at_a_Glance.html",
    "[7] Prit2341. Traffic-Management-Using-AI, README and simulation files. https://github.com/Prit2341/Traffic-Management-Using-AI (accessed 2026-11-02). No source code reused; repository-reported 35% result treated as unverified."
]


def font(size=20, bold=False):
    candidates = ["arialbd.ttf" if bold else "arial.ttf",
                  r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_results():
    path = EXP / "experiment_results.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def create_architecture():
    width, height = 1500, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, box_font, small_font = font(34, True), font(23, True), font(19)
    draw.text((60, 35), "Software traffic management architecture", fill=NAVY, font=title_font)
    boxes = {
        "video": (65, 180, 330, 325, "Recorded video\nFrame sampling"),
        "detect": (420, 130, 760, 300, "YOLO vehicle detection\nClass + confidence"),
        "track": (420, 375, 760, 545, "ByteTrack identities\nTrajectory history"),
        "lane": (850, 130, 1185, 300, "ROI lane counts\nCars / buses / trucks"),
        "analysis": (850, 375, 1185, 545, "Density + queue proxy\nTraffic state"),
        "memory": (1250, 170, 1460, 350, "SQLite + rolling\ntraffic memory"),
        "prediction": (850, 650, 1185, 810, "Forecasting models\nRF / XGBoost / LSTM"),
        "control": (420, 650, 760, 810, "Signal controllers\nFixed / rule / AI"),
        "dashboard": (65, 650, 330, 810, "Streamlit dashboard\nFastAPI services"),
    }
    for key, (x1, y1, x2, y2, label) in boxes.items():
        fill = (230, 241, 245) if key not in ("control", "dashboard") else (220, 239, 237)
        draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=fill, outline=TEAL, width=3)
        lines = label.split("\n")
        total_h = len(lines) * 32
        y = (y1 + y2 - total_h) / 2
        for index, line in enumerate(lines):
            f = box_font if index == 0 else small_font
            bbox = draw.textbbox((0, 0), line, font=f)
            draw.text(((x1 + x2 - (bbox[2] - bbox[0])) / 2, y + index * 32),
                      line, fill=INK, font=f)

    def arrow(start, end, color=MUTED):
        draw.line((start, end), fill=color, width=4)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        head = 13
        for delta in (2.55, -2.55):
            point = (end[0] + head * math.cos(angle + delta),
                     end[1] + head * math.sin(angle + delta))
            draw.line((end, point), fill=color, width=4)

    arrow((330, 250), (420, 220))
    arrow((590, 300), (590, 375))
    arrow((760, 215), (850, 215))
    arrow((760, 450), (850, 450))
    arrow((1015, 300), (1015, 375))
    arrow((1185, 225), (1250, 225))
    arrow((1185, 455), (1250, 310))
    arrow((1015, 545), (1015, 650))
    arrow((1250, 350), (1185, 650))
    arrow((850, 730), (760, 730))
    arrow((420, 730), (330, 730))
    arrow((850, 500), (760, 650))
    draw.text((60, 850), "All signal outputs are software recommendations; no physical controller is connected.",
              fill=MUTED, font=small_font)
    image.save(OUT / "architecture.png")


def create_chart(rows):
    colors_map = {"fixed": (54, 107, 160), "rule_based": (0, 132, 145),
                  "ai_assisted": (218, 131, 45)}
    scenarios = list(dict.fromkeys(row["scenario"] for row in rows))
    image = Image.new("RGB", (1600, 900), "white")
    d = ImageDraw.Draw(image)
    d.text((65, 35), "Controller comparison in seeded software simulation", fill=NAVY, font=font(32, True))
    d.text((65, 80), "Mean queue delay per completed departure (seconds)", fill=MUTED, font=font(21))
    left, top, chart_w, chart_h = 150, 160, 1360, 570
    d.line((left, top, left, top + chart_h), fill=INK, width=3)
    d.line((left, top + chart_h, left + chart_w, top + chart_h), fill=INK, width=3)
    vals = [float(r["avg_waiting_time_s"]) for r in rows]
    maximum = max(120, math.ceil(max(vals) / 20) * 20)
    for tick in range(0, maximum + 1, 20):
        y = top + chart_h - tick / maximum * chart_h
        d.line((left, y, left + chart_w, y), fill=(225, 231, 235), width=1)
        d.text((left - 48, y - 10), str(tick), fill=MUTED, font=font(16))
    group_w = chart_w / len(scenarios)
    bar_w = 68
    for gi, scenario in enumerate(scenarios):
        group_rows = {r["controller"]: r for r in rows if r["scenario"] == scenario}
        center = left + group_w * (gi + 0.5)
        for ci, controller in enumerate(CONTROLLERS):
            value = float(group_rows[controller]["avg_waiting_time_s"])
            x = center + (ci - 1) * (bar_w + 12)
            y = top + chart_h - value / maximum * chart_h
            d.rectangle((x, y, x + bar_w, top + chart_h), fill=colors_map[controller])
            d.text((x + 2, y - 25), f"{value:.1f}", fill=INK, font=font(15))
        label = scenario.replace("_", " ")
        bbox = d.textbbox((0, 0), label, font=font(18))
        d.text((center - (bbox[2] - bbox[0]) / 2, top + chart_h + 18), label, fill=INK, font=font(18))
    legend_y = 810
    for index, controller in enumerate(CONTROLLERS):
        x = 410 + index * 310
        d.rectangle((x, legend_y, x + 25, legend_y + 22), fill=colors_map[controller])
        d.text((x + 36, legend_y - 2), CONTROLLER_LABELS[controller], fill=INK, font=font(18))
    d.text((65, 860), "One run per scenario and controller; synthetic arrivals; not road or SUMO data.",
           fill=MUTED, font=font(17))
    image.save(EXP / "experiment_summary.png")


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def add_page_field(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("IT 207  |  Page ")
    run.font.size = Pt(9)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)


def create_docx(rows):
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin, sec.bottom_margin = Inches(.7), Inches(.65)
    sec.left_margin, sec.right_margin = Inches(.8), Inches(.8)
    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(10.5)
    styles["Normal"].font.color.rgb = RGBColor(*INK)
    styles["Normal"].paragraph_format.space_after = Pt(7)
    for name in ("Title", "Heading 1", "Heading 2"):
        styles[name].font.name = "Arial"
        styles[name].font.color.rgb = RGBColor(0, 0, 0)
    styles["Title"].font.size = Pt(26)
    styles["Heading 1"].font.size = Pt(16)
    styles["Heading 2"].font.size = Pt(12)
    add_page_field(sec.footer.paragraphs[0])
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Human Inspired AI for Intelligent Traffic Management")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = subtitle.add_run("IT 207 Project Report\nSeven chapter submission")
    r.bold, r.font.size = True, Pt(15)
    doc.add_paragraph("Student name: To be completed by student\nInstitution and section: To be completed by student\nSubmission date: To be completed by student")
    doc.add_paragraph("Evidence note: this report separates implemented code, software simulation results, and unverified capabilities. No real traffic video, labels, trained weights, or SUMO run was available in the supplied workspace.")
    doc.add_page_break()
    doc.add_heading("Abstract", level=1)
    doc.add_paragraph("This project continues a Python traffic management prototype with video detection and tracking modules, lane-based analysis, predictor implementations, three signal controllers, a software simulator, a FastAPI backend, and a Streamlit dashboard. The fallback simulation and experiment comparison were repaired and evaluated under four synthetic demand scenarios. Results are mixed: AI-assisted control improves delay and throughput in the uneven-demand run but performs worse than fixed timing in the medium and heavy balanced runs. The evaluation is a single seeded simulation per condition and does not measure real-road performance. YOLO detection, tracking, model forecasting, the full dashboard, and SUMO could not be run because the workspace had no model weights or input data and the execution environment lacked their dependencies. The report documents the system, human-cognition mapping, evidence, and limitations.")
    doc.add_heading("Contents", level=1)
    for heading, _ in REPORT:
        doc.add_paragraph(heading)
    doc.add_paragraph("References and reproducibility notes")
    doc.add_page_break()
    for heading, paragraphs in REPORT:
        doc.add_heading(heading, level=1)
        for paragraph in paragraphs:
            doc.add_paragraph(paragraph)
        if heading.startswith("Chapter 4"):
            doc.add_picture(str(OUT / "architecture.png"), width=Inches(6.7))
            cap = doc.add_paragraph("Figure 1. Software architecture. Recommendations remain in software.")
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if heading.startswith("Chapter 6"):
            table = doc.add_table(rows=1, cols=5)
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            table.style = "Table Grid"
            headers = ["Scenario", "Controller", "Delay s", "Queue m/approach", "Throughput / h"]
            for i, text in enumerate(headers):
                cell = table.rows[0].cells[i]
                cell.text = text
                set_cell_shading(cell, "142D4C")
                for run in cell.paragraphs[0].runs:
                    run.font.color.rgb = RGBColor(255, 255, 255)
                    run.font.bold = True
                    run.font.size = Pt(8)
            for row in rows:
                cells = table.add_row().cells
                vals = [row["scenario"].replace("_", " "), CONTROLLER_LABELS[row["controller"]],
                        row["avg_waiting_time_s"], row["avg_queue_length_m_per_approach"],
                        row["throughput_vehicles_per_hour"]]
                for cell, value in zip(cells, vals):
                    cell.text = str(value)
                    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    for run in cell.paragraphs[0].runs:
                        run.font.size = Pt(8)
            doc.add_picture(str(EXP / "experiment_summary.png"), width=Inches(6.7))
        if heading.startswith("Chapter 7"):
            doc.add_heading("References", level=2)
            for reference in REFERENCES:
                doc.add_paragraph(reference, style="List Number")
            doc.add_heading("Reproducibility", level=2)
            doc.add_paragraph("Run `python -m unittest discover -s tests -v` for simulator checks. Run `python -c \"from src.simulation.sumo_runner import run_sumo_experiments; run_sumo_experiments('submission/experiment_results')\"` to regenerate the experiment table. The demand profiles, seed, model parameters, and output labels are in config.yaml and src/simulation/sumo_runner.py.")
    doc.save(OUT / "final_report.docx")


def pdf_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("T", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22,
                                leading=27, textColor=colors.HexColor("#142D4C"), alignment=TA_CENTER, spaceAfter=10),
        "subtitle": ParagraphStyle("S", parent=base["Normal"], fontSize=11, leading=16,
                                   textColor=colors.HexColor("#5D6C7A"), alignment=TA_CENTER, spaceAfter=14),
        "h1": ParagraphStyle("H1x", parent=base["Heading1"], fontSize=15, leading=19,
                             textColor=colors.HexColor("#142D4C"), spaceBefore=11, spaceAfter=7, keepWithNext=True),
        "h2": ParagraphStyle("H2x", parent=base["Heading2"], fontSize=11.5, leading=14,
                             textColor=colors.HexColor("#008491"), spaceBefore=8, spaceAfter=5, keepWithNext=True),
        "body": ParagraphStyle("Bx", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2,
                                leading=13.2, textColor=colors.HexColor("#222B34"), spaceAfter=7),
        "small": ParagraphStyle("Sm", parent=base["BodyText"], fontSize=8, leading=10.5,
                                textColor=colors.HexColor("#384957"), spaceAfter=4),
    }


def page_footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D6E0E8"))
    canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#5D6C7A"))
    canvas.drawString(18 * mm, 10 * mm, "IT 207 | Human Inspired AI Traffic Management")
    canvas.drawRightString(192 * mm, 10 * mm, str(document.page))
    canvas.restoreState()


def pdf_doc(path, story):
    SimpleDocTemplate(str(path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
                      topMargin=18 * mm, bottomMargin=21 * mm, title=path.stem.replace("_", " ")).build(
                          story, onFirstPage=page_footer, onLaterPages=page_footer)


def create_report_pdf(rows):
    s = pdf_styles()
    story = [Paragraph("Human Inspired AI for Intelligent Traffic Management", s["title"]),
             Paragraph("IT 207 Project Report | Seven chapter submission", s["subtitle"]),
             Paragraph("Student details: to be completed by student", s["body"]),
             Paragraph("Evidence note: measured results in this report come from the seeded software simulator. No real traffic video, labels, trained weights, or SUMO run was available in the supplied workspace.", s["body"]),
             Paragraph("Abstract", s["h1"]),
             Paragraph("This project continues an existing Python prototype with detection and tracking modules, lane analysis, predictor implementations, signal controllers, a software simulator, API, and dashboard. The simulator comparison was repaired and evaluated under four synthetic demand scenarios. AI-assisted control improved both delay and throughput in the uneven scenario but did not outperform fixed timing in medium or heavy balanced traffic. This single seeded run does not establish a general performance gain. Video models and SUMO could not be run because required model files, input data, and runtime dependencies were absent.", s["body"]),
             Paragraph("Chapter guide", s["h1"])]
    for h, _ in REPORT:
        story.append(Paragraph(h, s["body"]))
    story.append(PageBreak())
    for heading, paragraphs in REPORT:
        story.append(Paragraph(heading, s["h1"]))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph.replace("&", "&amp;"), s["body"]))
        if heading.startswith("Chapter 4"):
            story.append(PdfImage(str(OUT / "architecture.png"), width=170 * mm, height=102 * mm))
            story.append(Paragraph("Figure 1. Software architecture. Signal outputs are software recommendations only.", s["small"]))
        if heading.startswith("Chapter 6"):
            data = [["Scenario", "Controller", "Delay s", "Queue m", "Throughput/h"]]
            for row in rows:
                data.append([row["scenario"].replace("_", " "), CONTROLLER_LABELS[row["controller"]],
                             row["avg_waiting_time_s"], row["avg_queue_length_m_per_approach"],
                             row["throughput_vehicles_per_hour"]])
            table = Table(data, repeatRows=1, colWidths=[33*mm, 30*mm, 23*mm, 26*mm, 27*mm])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#142D4C")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#B9C7D2")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F7")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([Spacer(1, 4), table, Spacer(1, 9),
                          PdfImage(str(EXP / "experiment_summary.png"), width=170*mm, height=95.6*mm)])
        if heading.startswith("Chapter 7"):
            story.append(Paragraph("References", s["h2"]))
            for ref in REFERENCES:
                story.append(Paragraph(ref.replace("&", "&amp;"), s["small"]))
    pdf_doc(OUT / "final_report.pdf", story)


def create_short_pdfs():
    s = pdf_styles()
    proposal = [
        "## Problem and rationale|Intersections with changing approach demand can allocate green time inefficiently. This project continues an existing software prototype and evaluates whether simple adaptive controllers can respond to lane demand while preserving minimum service and fairness.",
        "## Aim and objectives|The aim is to build and evaluate a software-only traffic management system inspired by human attention, memory, learning, and decision-making. Objectives: retain the existing video pipeline; analyze lane-wise counts and queue proxies; expose recommendations and comparisons through an existing dashboard; and compare fixed-time, rule-based, and AI-assisted control under reproducible scenarios.",
        "## Scope and method|Recorded video is the intended perception source. YOLO vehicle detection and ByteTrack identities feed ROI counts, traffic analysis, and optional forecasts. A weighted controller combines current demand, waiting urgency, fairness, and predictions. The present experiment uses a seeded software queue simulator because SUMO and its dependencies are not available. No hardware or physical signal connection is in scope.",
        "## Evaluation and expected evidence|The prototype compares mean queue delay, mean queue length, throughput, and congestion rate. Detection precision, recall, mAP, tracking IDF1/MOTA, and forecasting MAE/RMSE require labeled data and trained models and will be reported only after those inputs are available. The current single-run simulation is preliminary, not evidence of real-world effectiveness.",
        "## Risks and ethics|Video may contain identifying information. A future study should minimize retention, limit access, state lawful purpose, and evaluate errors across lighting, camera, vehicle class, and traffic conditions. Recommendations remain advisory in software. The controller is not validated for public-road deployment.",
        "## Schedule|1. Audit and repair the supplied architecture. 2. Verify modules and the local simulator. 3. Collect or select licensed data and train/evaluate models. 4. Conduct repeated baseline experiments. 5. Finalize report, presentation, and viva material. Steps requiring absent datasets or system dependencies remain open.",
    ]
    story = [Paragraph("Project Proposal", s["title"]), Paragraph("IT 207 Human Intelligence", s["subtitle"])]
    for section in proposal:
        h, body = section.split("|", 1)
        story.extend([Paragraph(h.removeprefix("## "), s["h1"]), Paragraph(body, s["body"])])
    pdf_doc(OUT / "project_proposal.pdf", story)

    literature = [
        ("Human cognition mapping", "Selective attention motivates lane and region filters. Working memory motivates a short traffic-history buffer. Learning motivates supervised traffic-forecast models. Decision-making motivates a multi-criteria signal controller. These are implementation analogies, not psychological measures; the course authority cautions against overclaiming intelligence from proxies [1]. Broadbent [2] and Baddeley [3] provide foundational accounts of attention and working memory."),
        ("Vehicle detection", "YOLO formulates object detection as a single neural-network prediction over an image [4]. This project configures an Ultralytics YOLO11n model for four COCO vehicle classes. The weights are absent, so the configured detector was not run and no detection accuracy is claimed."),
        ("Vehicle tracking", "ByteTrack associates detection boxes with tracklets, including lower-confidence detections, to reduce identity fragmentation [5]. The project uses the Ultralytics ByteTrack interface. Track quality requires video annotations; none were supplied, so IDF1 and MOTA remain unmeasured."),
        ("Traffic control and simulation", "The code compares a fixed allocation, count/wait heuristic, and weighted adaptive allocation. SUMO is an open-source microscopic simulator suitable for traffic control experiments [6], but this workspace's network and phase plan were not validated and SUMO was unavailable. Therefore the present comparison uses only the in-project seeded queue simulator."),
        ("Reference project and research gap", "Prit2341's repository discusses density-based green allocation, startup lag, vehicle category, lane count, and consecutive-signal coordination [7]. Its README's reported throughput gain is unverified by this study. No source code was reused; no repository license file was found during inspection. The next research step is to test corridor coordination in a validated network with multiple intersections, realistic phase conflicts, calibrated vehicle flow, and multi-seed evaluation."),
        ("References", "\n".join(REFERENCES)),
    ]
    story = [Paragraph("Literature Review", s["title"]), Paragraph("IT 207 Human Intelligence", s["subtitle"])]
    for h, body in literature:
        story.extend([Paragraph(h, s["h1"])])
        for para in body.split("\n"):
            story.append(Paragraph(para.replace("&", "&amp;"), s["small"] if h == "References" else s["body"]))
    pdf_doc(OUT / "literature_review.pdf", story)

    questions = [
        ("1. How does the system connect human intelligence to AI?", "It uses selective-attention ideas for ROI-based lane perception, working-memory ideas for recent traffic history, learning for forecast model code, and decision-making for weighted signal allocation. These are computational analogies, not measurements of human intelligence."),
        ("2. What was already present in the workspace?", "Detection, tracking, analysis, prediction, optimization, SUMO hooks, API, dashboard, and YAML configuration were present. The README, tests, model weights, data, and submission documents were absent."),
        ("3. Which detector and tracker does the project configure?", "YOLO11n with COCO vehicle classes is configured; the tracking module uses Ultralytics ByteTrack. No weights or labeled video were supplied, so neither was evaluated."),
        ("4. What do the queue and speed values mean?", "They are proxies. Queue length assumes five metres per vehicle and speed is a count-based heuristic unless calibrated positional data exist. They are not physical measurements."),
        ("5. What controllers were compared?", "Fixed-time, rule-based, and AI-assisted weighted allocation. All used the same arrival rates and random seed per scenario."),
        ("6. What are the main experiment results?", "AI-assisted performed better than fixed-time on delay and throughput in the uneven-demand scenario, but worse on both in medium and heavy balanced scenarios. One seeded run is preliminary and does not establish general superiority."),
        ("7. Why are there no mAP or IDF1 scores?", "There was no video, ground-truth annotation, or YOLO weight file in the workspace, so calculating those metrics would be unsupported."),
        ("8. Why was SUMO not used for reported results?", "SUMO and TraCI were not installed, and the generated network did not have a verified traffic-signal phase plan. The results therefore come from the software queue simulator."),
        ("9. What is the key limitation of the controller evaluation?", "The simulator simplifies arrivals, discharge, and phases; it omits turning conflicts, pedestrians, startup loss, spillback, calibration, and uncertainty across repeated seeds."),
        ("10. What ethical concerns apply to traffic video?", "Footage can expose faces, plates, and movement patterns. Minimize retention, restrict access, document lawful purpose, and measure performance across camera, lighting, class, and traffic conditions."),
        ("11. Does the system control real traffic lights?", "No. It is software-only and returns recommendations; there is no hardware or physical-controller integration."),
        ("12. What would you do next?", "Install the supported Python 3.11 dependencies, add licensed annotated video, verify model paths, measure detection and tracking metrics, train forecasting on real time-ordered data, and run repeated experiments in a validated SUMO network."),
    ]
    story = [Paragraph("Viva Questions and Answers", s["title"]), Paragraph("IT 207 Human Intelligence", s["subtitle"])]
    viva_body = ParagraphStyle("VivaBody", parent=s["body"], fontSize=8.6, leading=11.2, spaceAfter=5)
    viva_heading = ParagraphStyle("VivaHeading", parent=s["h2"], fontSize=10.5, leading=12, spaceBefore=5, spaceAfter=3)
    for question, answer in questions:
        story.extend([Paragraph(question, viva_heading), Paragraph(answer, viva_body)])
    pdf_doc(OUT / "viva_questions.pdf", story)


def add_slide(prs, title, bullets, notes=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background.fill
    background.solid()
    background.fore_color.rgb = PptColor(248, 250, 252)
    title_box = slide.shapes.add_textbox(PptInches(.65), PptInches(.4), PptInches(12), PptInches(.7))
    p = title_box.text_frame.paragraphs[0]
    p.text = title
    p.font.name, p.font.size, p.font.bold = "Arial", PptPt(28), True
    p.font.color.rgb = PptColor(*NAVY)
    y = 1.55
    for bullet in bullets:
        box = slide.shapes.add_textbox(PptInches(.85), PptInches(y), PptInches(11.5), PptInches(.68))
        tf = box.text_frame
        tf.word_wrap = True
        para = tf.paragraphs[0]
        para.text = bullet
        para.font.name, para.font.size = "Arial", PptPt(19)
        para.font.color.rgb = PptColor(*INK)
        para.space_after = PptPt(7)
        y += .82 if len(bullet) < 120 else 1.05
    footer = slide.shapes.add_textbox(PptInches(.65), PptInches(7.12), PptInches(12), PptInches(.22))
    fp = footer.text_frame.paragraphs[0]
    fp.text = "IT 207  |  software-only prototype  |  simulation results are synthetic"
    fp.font.name, fp.font.size, fp.font.color.rgb = "Arial", PptPt(9), PptColor(*MUTED)
    return slide


def create_presentation(rows):
    prs = Presentation()
    prs.slide_width, prs.slide_height = PptInches(13.333), PptInches(7.5)
    slide = add_slide(prs, "Human Inspired AI for Intelligent Traffic Management", [
        "IT 207 project presentation",
        "Software prototype with video analytics and adaptive signal recommendations",
        "Student and institution details: complete before submission",
    ])
    add_slide(prs, "Problem and project scope", [
        "Approach demand changes over time; fixed green allocations may not reflect current queues.",
        "The project uses recorded video, software analysis, and simulated control.",
        "No IoT devices, physical sensors, or traffic-light hardware are connected.",
        "The supplied workspace had no video, labels, pretrained weights, or prior experiment output.",
    ])
    add_slide(prs, "Human cognition mapping", [
        "Selective attention: configurable lane and region-of-interest polygons.",
        "Working memory: bounded traffic history and SQLite observations.",
        "Learning: forecasting model code for Random Forest, XGBoost, and LSTM.",
        "Decision-making: weighted demand, queue, waiting, and fairness inputs for green allocation.",
        "These are computational analogies, not psychological measurements.",
    ])
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PptColor(248, 250, 252)
    tb = slide.shapes.add_textbox(PptInches(.65), PptInches(.4), PptInches(12), PptInches(.7))
    p = tb.text_frame.paragraphs[0]
    p.text, p.font.name, p.font.size, p.font.bold = "System architecture", "Arial", PptPt(28), True
    p.font.color.rgb = PptColor(*NAVY)
    slide.shapes.add_picture(str(OUT / "architecture.png"), PptInches(1.66), PptInches(1.0),
                             width=PptInches(10.0), height=PptInches(6.0))
    footer = slide.shapes.add_textbox(PptInches(.65), PptInches(7.12), PptInches(12), PptInches(.22))
    fp = footer.text_frame.paragraphs[0]
    fp.text = "Software recommendations only; no physical controller is connected."
    fp.font.name, fp.font.size, fp.font.color.rgb = "Arial", PptPt(9), PptColor(*MUTED)
    add_slide(prs, "Implementation status", [
        "Existing: detector, tracker, analyser, predictor classes, controllers, API, dashboard, and config.",
        "Repaired: seeded simulation, scenario-specific demand, same simulated horizon, and metric units.",
        "Not run: video inference, tracking metrics, forecast validation, full dashboard, and SUMO.",
        "The environment had Python 3.14 and lacked the project's video and web runtime packages.",
    ])
    add_slide(prs, "Experiment design", [
        "Four scenarios: low, medium, heavy, and uneven approach demand.",
        "Three controllers compared with the same seeded Poisson arrivals per scenario.",
        "1,800 simulated seconds; 5 m per queued vehicle; 0.5 vehicles per green second.",
        "Metrics: queue delay per completed departure, queue length, throughput per hour, and congestion fraction.",
        "One deterministic run per condition; this is a simplified software simulation.",
    ])
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PptColor(248, 250, 252)
    tb = slide.shapes.add_textbox(PptInches(.65), PptInches(.4), PptInches(12), PptInches(.7))
    p = tb.text_frame.paragraphs[0]
    p.text = "Measured simulated queue delay"
    p.font.name, p.font.size, p.font.bold, p.font.color.rgb = "Arial", PptPt(28), True, PptColor(*NAVY)
    chart_data = CategoryChartData()
    scenarios = list(dict.fromkeys(row["scenario"].replace("_", " ") for row in rows))
    chart_data.categories = scenarios
    for controller in CONTROLLERS:
        vals = [float(next(r["avg_waiting_time_s"] for r in rows
                           if r["scenario"].replace("_", " ") == scenario and r["controller"] == controller))
                for scenario in scenarios]
        chart_data.add_series(CONTROLLER_LABELS[controller], vals)
    chart = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, PptInches(.8), PptInches(1.35),
                                   PptInches(11.7), PptInches(5.35), chart_data).chart
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.value_axis.has_title = True
    chart.value_axis.axis_title.text_frame.text = "Seconds"
    chart.category_axis.tick_labels.font.size = PptPt(13)
    chart.value_axis.tick_labels.font.size = PptPt(12)
    chart.has_title = False
    footer = slide.shapes.add_textbox(PptInches(.65), PptInches(7.12), PptInches(12), PptInches(.22))
    fp = footer.text_frame.paragraphs[0]
    fp.text = "One seeded run per condition. Lower is better; synthetic software simulation, not SUMO or road data."
    fp.font.name, fp.font.size, fp.font.color.rgb = "Arial", PptPt(9), PptColor(*MUTED)
    add_slide(prs, "Results interpretation", [
        "Uneven demand: AI-assisted had 37.19 s delay and 724 vehicles/hour; fixed-time had 52.62 s and 686 vehicles/hour.",
        "Medium demand: AI-assisted delay was 64.12 s versus 42.86 s fixed; throughput was 598 versus 606 vehicles/hour.",
        "Heavy demand: AI-assisted delay was 103.87 s versus 45.01 s fixed; throughput was 1,168 versus 1,208 vehicles/hour.",
        "The evidence is mixed and does not support a general improvement claim.",
    ])
    add_slide(prs, "Ethics and limitations", [
        "Video may expose faces, license plates, and movement patterns; minimize retention and restrict access.",
        "Errors may vary by camera angle, lighting, weather, vehicle class, and local traffic mix.",
        "Queue and speed values are rough proxies without camera calibration.",
        "Simulation omits pedestrians, turning conflicts, startup loss, spillback, and repeated-seed uncertainty.",
        "No public-road control is permitted by this prototype; recommendations are software-only.",
    ])
    add_slide(prs, "Next steps and references", [
        "Install the supported Python 3.11 stack and add compatible YOLO weights.",
        "Evaluate on licensed video with detection and tracking annotations.",
        "Train predictors on time-ordered real data and report forecast errors.",
        "Validate a conflict-free SUMO signal plan and run repeated, multi-intersection scenarios.",
        "References: IT 207 course PDF; Redmon et al. CVPR 2016; Zhang et al. ECCV 2022; Eclipse SUMO docs.",
    ])
    prs.save(OUT / "presentation.pptx")
    # python-pptx can emit signed negative axis IDs, while the OOXML chart
    # schema and downstream renderers expect non-negative unsigned IDs.
    chart_path = OUT / "presentation.pptx"
    staging_path = OUT / "presentation.normalized.pptx"
    with ZipFile(chart_path, "r") as source, ZipFile(staging_path, "w", ZIP_DEFLATED) as target:
        for item in source.infolist():
            payload = source.read(item.filename)
            if item.filename.startswith("ppt/charts/chart") and item.filename.endswith(".xml"):
                xml = payload.decode("utf-8")
                ids = {}
                for match in __import__("re").findall(r'<c:axId val="(-\d+)"', xml):
                    ids.setdefault(match, str(100000000 + len(ids) + 1))
                for old, new in ids.items():
                    xml = xml.replace(f'val="{old}"', f'val="{new}"')
                payload = xml.encode("utf-8")
            target.writestr(item, payload)
    staging_path.replace(chart_path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    rows = load_results()
    create_architecture()
    create_chart(rows)
    create_docx(rows)
    create_report_pdf(rows)
    create_short_pdfs()
    create_presentation(rows)
    print(f"Built submission files under {OUT}")


if __name__ == "__main__":
    main()
