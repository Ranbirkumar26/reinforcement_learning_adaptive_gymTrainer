import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = path.resolve(new URL("..", import.meta.url).pathname);
const SKILL_DIR = "/Users/tarry/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations";
const TMP_DIR = path.join(ROOT, ".codex-build", "ppt");
const FINAL_PPTX = path.join(ROOT, "slides", "final_adaptive_rl_gym_coach_v3.pptx");
const RUNTIME_PYTHON = "/Users/tarry/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const RUNTIME_NODE_MODULES = "/Users/tarry/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
process.env.RUNTIME_NODE_MODULES = RUNTIME_NODE_MODULES;

const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const fontFamily = resolvePresentationFont({ fontFamily: "Arial" });
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

const colors = {
  ink: "#172033",
  muted: "#536173",
  blue: "#2563eb",
  pale: "#e8eef7",
  green: "#15803d",
  amber: "#b45309",
  red: "#b91c1c",
  line: "#b8c4d6",
};

function addText(slide, text, left, top, width, height, options = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    typeface: fontFamily,
    fontSize: options.size ?? 24,
    bold: options.bold ?? false,
    color: options.color ?? colors.ink,
    autoFit: "shrinkText",
  };
  return box;
}

function addHeader(slide, title, subtitle = "") {
  slide.background.fill = "#ffffff";
  addText(slide, title, 64, 48, 900, 58, { size: 34, bold: true });
  if (subtitle) addText(slide, subtitle, 64, 104, 980, 34, { size: 17, color: colors.muted });
  addText(slide, "Adaptive RL Gym Coach", 1000, 54, 210, 26, { size: 13, color: colors.muted });
}

function addBand(slide, left, top, width, height, fill = colors.pale) {
  const shape = slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill,
    line: { fill: colors.line, width: 1 },
  });
  return shape;
}

function addBullets(slide, items, left, top, width, rowHeight, color = colors.ink) {
  items.forEach((item, index) => {
    addText(slide, item, left + 22, top + index * rowHeight, width - 22, rowHeight - 4, { size: 18, color });
    addBand(slide, left, top + index * rowHeight + 8, 8, 8, colors.blue);
  });
}

function addFooter(slide, number) {
  addText(slide, `${number} / 12`, 1120, 664, 90, 24, { size: 13, color: colors.muted });
}

function slide1() {
  const slide = presentation.slides.add();
  slide.background.fill = "#f8fafc";
  addText(slide, "Adaptive RL Gym Coach", 72, 82, 880, 70, { size: 44, bold: true });
  addText(slide, "Submission-ready squat coaching MVP", 72, 160, 760, 34, { size: 24, color: colors.muted });
  addBand(slide, 72, 252, 1080, 180, "#ffffff");
  addText(slide, "Pose estimation, biomechanics, fatigue, injury risk, reinforcement learning, and explainable feedback in one local demo.", 108, 294, 960, 96, { size: 28 });
  addText(slide, "Free local stack: Python, Streamlit, MediaPipe, Gymnasium, Stable-Baselines3", 72, 610, 940, 30, { size: 18, color: colors.muted });
  addFooter(slide, 1);
}

function slide2() {
  const slide = presentation.slides.add();
  addHeader(slide, "MVP Scope", "One complete squat workflow from input to coaching explanation");
  addBullets(slide, [
    "Video upload or bundled synthetic squat session",
    "Per-frame pose landmarks saved to CSV",
    "Per-rep biomechanics, fatigue, and risk features",
    "RL or heuristic policy chooses coaching action",
    "Streamlit app displays explanation and charts",
  ], 96, 184, 1020, 56);
  addFooter(slide, 2);
}

function slide3() {
  const slide = presentation.slides.add();
  addHeader(slide, "System Architecture", "Each layer writes an auditable artifact");
  const labels = ["Video or landmarks", "Pose extraction", "Biomechanics", "RL policy", "Explanation"];
  labels.forEach((label, index) => {
    addBand(slide, 84 + index * 230, 260, 180, 92, "#eef6ff");
    addText(slide, label, 104 + index * 230, 286, 140, 34, { size: 19, bold: true });
  });
  addText(slide, "Outputs: landmarks.csv, rep_features.csv, coach_outputs.json, charts, report, PPT", 96, 476, 1040, 36, { size: 24 });
  addFooter(slide, 3);
}

function slide4() {
  const slide = presentation.slides.add();
  addHeader(slide, "Biomechanics Features", "Feature extraction works per squat repetition");
  const rows = [
    ["Feature", "Purpose"],
    ["Knee angle minimum", "Squat depth and shallow-squat detection"],
    ["Hip angle minimum", "Range-of-motion context"],
    ["Torso lean maximum", "Forward-lean risk signal"],
    ["Knee tracking proxy", "Knee alignment risk signal"],
    ["Tempo and smoothness", "Fatigue and stability signal"],
  ];
  rows.forEach((row, index) => {
    const y = 172 + index * 60;
    addBand(slide, 92, y, 1090, 48, index === 0 ? colors.pale : "#ffffff");
    addText(slide, row[0], 116, y + 10, 320, 24, { size: 18, bold: index === 0 });
    addText(slide, row[1], 460, y + 10, 660, 24, { size: 18, bold: index === 0 });
  });
  addFooter(slide, 4);
}

function slide5() {
  const slide = presentation.slides.add();
  addHeader(slide, "Personalized Coaching State", "The policy sees movement, fatigue, risk, and history");
  addBullets(slide, [
    "Normalized knee, hip, and torso values",
    "Rep progress and previous coaching action",
    "Fatigue score from tempo, range, and jitter",
    "Injury risk from torso lean, knee tracking, and asymmetry",
    "Repeated mistake count and skill level",
  ], 96, 176, 1030, 56);
  addFooter(slide, 5);
}

function slide6() {
  const slide = presentation.slides.add();
  addHeader(slide, "MDP Formulation", "Each repetition becomes one decision step");
  addBand(slide, 92, 190, 320, 330, "#f8fafc");
  addText(slide, "State", 116, 216, 220, 32, { size: 28, bold: true });
  addText(slide, "Pose features, fatigue, risk, history, previous action", 116, 270, 250, 140, { size: 22 });
  addBand(slide, 480, 190, 320, 330, "#f8fafc");
  addText(slide, "Action", 504, 216, 220, 32, { size: 28, bold: true });
  addText(slide, "Eight coaching actions including rest, joint highlight, tempo cue, and no feedback", 504, 270, 250, 170, { size: 22 });
  addBand(slide, 868, 190, 320, 330, "#f8fafc");
  addText(slide, "Reward", 892, 216, 220, 32, { size: 28, bold: true });
  addText(slide, "Improvement adds reward. Risk, fatigue, repeated errors, and excessive cues subtract reward.", 892, 270, 250, 170, { size: 22 });
  addFooter(slide, 6);
}

function slide7() {
  const slide = presentation.slides.add();
  addHeader(slide, "Coaching Actions", "The action is the coaching decision, not an exercise label");
  const actions = ["verbal cue", "joint highlight", "slow tempo", "adjust difficulty", "recommend rest", "no feedback", "demonstration", "breathing cue"];
  actions.forEach((action, index) => {
    const col = index % 2;
    const row = Math.floor(index / 2);
    addBand(slide, 120 + col * 500, 170 + row * 92, 420, 58, index === 4 ? "#fff7ed" : "#f8fafc");
    addText(slide, action, 146 + col * 500, 186 + row * 92, 360, 26, { size: 22, bold: true });
  });
  addFooter(slide, 7);
}

function slide8() {
  const slide = presentation.slides.add();
  addHeader(slide, "Explainable Output", "Each correction includes evidence and next action");
  addBand(slide, 110, 180, 1060, 300, "#f8fafc");
  addText(slide, "Problem", 150, 214, 180, 28, { size: 24, bold: true, color: colors.red });
  addText(slide, "Knee tracking issue", 150, 258, 260, 34, { size: 24 });
  addText(slide, "Reason", 480, 214, 180, 28, { size: 24, bold: true, color: colors.amber });
  addText(slide, "Knee tracking proxy raised injury risk", 480, 258, 280, 76, { size: 24 });
  addText(slide, "Correction", 810, 214, 180, 28, { size: 24, bold: true, color: colors.green });
  addText(slide, "Keep knees aligned with toes", 810, 258, 280, 72, { size: 24 });
  addText(slide, "Evidence frame and highlighted joint are shown in the Streamlit demo.", 142, 548, 980, 32, { size: 23 });
  addFooter(slide, 8);
}

function slide9() {
  const slide = presentation.slides.add();
  addHeader(slide, "Generated Evaluation Artifacts", "The project writes repeatable outputs for submission");
  const chart = slide.charts.add("bar", {
    position: { left: 150, top: 180, width: 980, height: 350 },
    categories: ["Landmarks", "Rep features", "Coach JSON", "Charts", "Report", "PPT"],
    series: [{ name: "Status", values: [1, 1, 1, 1, 1, 1], fill: colors.blue }],
    barOptions: { direction: "column", grouping: "clustered" },
    hasLegend: false,
    dataLabels: { showValue: false },
  });
  applyPresentationChartFont(chart, { fontFamily });
  addText(slide, "Submission outputs generated", 150, 148, 600, 28, { size: 22, bold: true });
  addText(slide, "All outputs are local files. No paid API or cloud service required.", 150, 558, 980, 32, { size: 22, color: colors.muted });
  addFooter(slide, 9);
}

function slide10() {
  const slide = presentation.slides.add();
  addHeader(slide, "QA Status", "Automated checks cover core behavior and submission artifacts");
  addBullets(slide, [
    "Pytest passed with 22 unit and integration tests",
    "Compile, sample generation, training, and evaluation passed",
    "Streamlit health check passed",
    "Real squat video processed with 17 detected reps",
    "Real-video overlay and evidence frames generated",
  ], 96, 176, 1030, 56);
  addFooter(slide, 10);
}

function slide11() {
  const slide = presentation.slides.add();
  addHeader(slide, "Limitations", "MVP choices keep the build credible within one week");
  addBullets(slide, [
    "Squat-only exercise support",
    "Injury risk uses heuristic biomechanical proxies",
    "RL training uses simulated trajectories without real long-term user feedback",
    "Camera angle and landmark confidence affect movement estimates",
    "Grad-CAM deferred because no CNN posture classifier is trained",
  ], 96, 176, 1030, 56);
  addFooter(slide, 11);
}

function slide12() {
  const slide = presentation.slides.add();
  slide.background.fill = "#f8fafc";
  addText(slide, "Project Contribution", 82, 86, 900, 56, { size: 40, bold: true });
  addBand(slide, 82, 210, 1080, 220, "#ffffff");
  addText(slide, "The prototype reframes AI fitness feedback as personalized sequential coaching. It observes the user, chooses an action, and explains the decision with movement evidence.", 122, 260, 990, 96, { size: 30 });
  addText(slide, "Submission package: code, README, tests, charts, report PDF, and final PPT", 82, 610, 960, 30, { size: 20, color: colors.muted });
  addFooter(slide, 12);
}

[slide1, slide2, slide3, slide4, slide5, slide6, slide7, slide8, slide9, slide10, slide11, slide12].forEach((fn) => fn());

const candidatePath = path.join(TMP_DIR, "candidate.pptx");
await PresentationFile.exportPptx(presentation).then((blob) => blob.save(candidatePath));

const requirements = {
  explicitTotalSlideCount: 12,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [9],
  materializeLiteralChartWorkbooks: true,
};
const fontPolicy = { basis: "design", families: [fontFamily] };
const stagingDir = path.join(ROOT, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });

await finalizePresentation({
  ...requirements,
  workspaceDir: ROOT,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [9],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "final_adaptive_rl_gym_coach_v3.validation.json"),
});

console.log(FINAL_PPTX);
