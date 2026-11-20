"use strict";
const examples = JSON.parse(document.getElementById("examples").textContent);
const scenes = document.getElementById("scenes");
const timeline = document.getElementById("timeline");
const pauseButton = document.getElementById("pause");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
let current = examples[0];
let paused = reducedMotion.matches;
let renderVersion = 0;
let ready = false;

function animations() {
  return [...document.querySelectorAll(".canvas iframe")]
    .map(frame => frame.contentDocument?.querySelector("svg")).filter(Boolean);
}
function sync(time, stop) {
  for (const svg of animations()) {
    svg.pauseAnimations();
    svg.setCurrentTime(time);
    if (!stop) svg.unpauseAnimations();
  }
  paused = stop;
  pauseButton.textContent = stop ? "Play" : "Pause";
  updateTime(time);
}
function updateTime(time) {
  const position = ((time % current.duration) + current.duration) % current.duration;
  timeline.value = String(Math.round(position / current.duration * 1000));
  timeline.setAttribute("aria-valuetext", `${position.toFixed(1)} seconds`);
  document.getElementById("time").textContent = `${position.toFixed(1)} s`;
}
function show(index) {
  current = examples[index];
  const version = ++renderVersion;
  ready = false;
  pauseButton.disabled = true;
  document.getElementById("replay").disabled = true;
  document.getElementById("direction").textContent = current.direction;
  document.getElementById("watch").textContent = current.watch;
  for (const [i, button] of [...scenes.children].entries()) {
    button.setAttribute("aria-pressed", String(i === index));
  }
  updateTime(0);
  let loaded = 0;
  const renderable = ["base", "finetuned"].filter(arm => current[arm].renderable).length;
  for (const arm of ["base", "finetuned"]) {
    const result = current[arm];
    const view = document.getElementById(`${arm}-view`);
    view.replaceChildren();
    if (!result.renderable) {
      const message = document.createElement("p");
      message.className = "render-error";
      message.textContent = "This take could not be safely displayed.";
      view.append(message);
      continue;
    }
    const frame = document.createElement("iframe");
    frame.title = `${current.title} — ${arm === "base" ? "original model" : "model taught motion"}`;
    frame.setAttribute("sandbox", "allow-same-origin");
    frame.srcdoc = `<!doctype html><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'none'; style-src 'unsafe-inline'"><style>html,body{margin:0;width:100%;height:100%;overflow:hidden}body{display:grid;place-items:center}svg{height:100%;width:100%;max-width:420px}</style>${result.render_source}`;
    frame.onload = () => {
      if (version !== renderVersion) return;
      const svg = frame.contentDocument.querySelector("svg");
      // The same camera crop in both arms enlarges the original artwork only for display.
      svg.setAttribute("viewBox", "40 32 176 176");
      svg.pauseAnimations();
      svg.setCurrentTime(0);
      loaded += 1;
      if (loaded === renderable) {
        ready = true;
        pauseButton.disabled = false;
        document.getElementById("replay").disabled = false;
        sync(0, reducedMotion.matches);
      }
    };
    view.append(frame);
  }
}
for (const [index, example] of examples.entries()) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "scene";
  button.setAttribute("aria-pressed", "false");
  const icon = document.createElement("span");
  icon.className = "scene-icon";
  icon.setAttribute("aria-hidden", "true");
  // input_svg is original, validated artwork embedded by the local build, never user input.
  icon.innerHTML = example.input_svg;
  icon.querySelector("svg").setAttribute("viewBox", "80 106 96 96");
  const label = document.createElement("span");
  label.className = "scene-label";
  label.textContent = example.title;
  button.append(icon, label);
  button.addEventListener("click", () => show(index));
  scenes.append(button);
}
document.getElementById("replay").addEventListener("click", () => sync(0, false));
pauseButton.addEventListener("click", () => sync(animations()[0]?.getCurrentTime() || 0, !paused));
timeline.addEventListener("input", () => sync(Number(timeline.value) / 1000 * current.duration, true));
reducedMotion.addEventListener("change", event => {
  if (event.matches && ready) sync(animations()[0]?.getCurrentTime() || 0, true);
});
function tick() {
  if (ready && !paused) updateTime(animations()[0]?.getCurrentTime() || 0);
  requestAnimationFrame(tick);
}
show(0);
requestAnimationFrame(tick);
