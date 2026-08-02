"""
HTML/JS canvas replay component for Room 4's drone flight trajectories.

Physics/training happen entirely in Python (drone_env.py / dqn_solver.py);
this module only turns an already-simulated trajectory into a self-contained
HTML document with a <canvas> animation. No live physics or keyboard control
here -- it just plays back a precomputed frame array, so it's a plain
one-way `st.iframe(html_string, ...)` embed rather than a full bidirectional
custom component. No external images/fonts (same "no external assets" rule
as Room 1's CSS theme) -- every visual is drawn with canvas 2D primitives.
"""

import json


def _layout_dict(cfg):
    return {
        "room": {"w": cfg.room_w, "h": cfg.room_h},
        "walls": [{"x": w.x, "y": w.y, "w": w.w, "h": w.h} for w in cfg.walls],
        "wind_zones": [
            {"x": z.x, "y": z.y, "w": z.w, "h": z.h, "fx": z.fx, "fy": z.fy}
            for z in cfg.wind_zones
        ],
        "accel_zones": [
            {"x": z.x, "y": z.y, "w": z.w, "h": z.h, "fx": z.fx, "fy": z.fy}
            for z in cfg.accel_zones
        ],
        "decel_zones": [
            {"x": z.x, "y": z.y, "w": z.w, "h": z.h} for z in cfg.decel_zones
        ],
        "pad": {"x": cfg.pad.x, "y": cfg.pad.y, "r": cfg.pad.radius},
    }


def _embed_json(obj):
    return json.dumps(obj).replace("</", "<\\/")


def render_flight_canvas(trajectory, cfg, title, element_id="panel", canvas_px=420):
    frames = trajectory["frames"]
    layout = _layout_dict(cfg)
    status = "landed" if trajectory["solved"] else ("crashed" if trajectory["crashed"] else "timeout")

    html = _TEMPLATE
    html = html.replace("__TITLE__", title.replace('"', "'"))
    html = html.replace("__ELEMENT_ID__", element_id)
    html = html.replace("__CANVAS_PX__", str(canvas_px))
    html = html.replace("__STATUS__", status)
    html = html.replace("__FRAMES_JSON__", _embed_json(frames))
    html = html.replace("__LAYOUT_JSON__", _embed_json(layout))
    return html


_TEMPLATE = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 6px;
    background: #0d141c;
    font-family: 'Consolas', 'Courier New', monospace;
    color: #cfe8ff;
  }
  .title { font-size: 13px; font-weight: bold; margin-bottom: 4px; color: #8fd1ff; }
  canvas { display: block; border: 1px solid #35424f; border-radius: 6px; background: #060a10; }
  .controls { display: flex; align-items: center; gap: 6px; margin-top: 6px; font-size: 12px; }
  .controls button, .controls select {
    background: #182531; color: #cfe8ff; border: 1px solid #3a4a5a;
    border-radius: 4px; padding: 3px 8px; cursor: pointer; font-family: inherit; font-size: 12px;
  }
  .controls button:hover { border-color: #6fc4ff; }
  input[type=range] { flex: 1; }
  .caption { font-size: 11px; color: #7f97ab; margin-top: 3px; }
</style>
</head>
<body>
  <div class="title">__TITLE__</div>
  <canvas id="cv___ELEMENT_ID__" width="__CANVAS_PX__" height="__CANVAS_PX__"></canvas>
  <div class="controls">
    <button id="play___ELEMENT_ID__">⏸ Pause</button>
    <input type="range" id="scrub___ELEMENT_ID__" min="0" max="0" value="0">
    <select id="speed___ELEMENT_ID__">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
    </select>
    <button id="reset___ELEMENT_ID__">⟲</button>
  </div>
  <div class="caption" id="caption___ELEMENT_ID__"></div>

<script>
(function () {
  const FRAMES = __FRAMES_JSON__;
  const LAYOUT = __LAYOUT_JSON__;
  const FINAL_STATUS = "__STATUS__";
  const DT = 0.02;
  const CANVAS_PX = __CANVAS_PX__;
  const ID = "__ELEMENT_ID__";

  const canvas = document.getElementById("cv_" + ID);
  const ctx = canvas.getContext("2d");
  const scrub = document.getElementById("scrub_" + ID);
  const playBtn = document.getElementById("play_" + ID);
  const resetBtn = document.getElementById("reset_" + ID);
  const speedSel = document.getElementById("speed_" + ID);
  const caption = document.getElementById("caption_" + ID);

  const nFrames = FRAMES.length;
  scrub.max = Math.max(0, nFrames - 1);

  const roomW = LAYOUT.room.w, roomH = LAYOUT.room.h;
  const MARGIN = 14;
  const SCALE = (CANVAS_PX - 2 * MARGIN) / Math.max(roomW, roomH);

  function toPx(x, y) {
    return [x * SCALE + MARGIN, CANVAS_PX - (y * SCALE + MARGIN)];
  }

  const ACTION_ARROWS = ["↙", "←", "↖", "↓", "•", "↑", "↘", "→", "↗"];

  let frameIdx = 0;
  let playing = nFrames > 1;
  let lastTs = null;
  let simTime = 0;

  function drawZone(z, fill, stroke) {
    const [x0, y0] = toPx(z.x, z.y + z.h);
    const w = z.w * SCALE, h = z.h * SCALE;
    ctx.fillStyle = fill;
    ctx.fillRect(x0, y0, w, h);
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x0, y0, w, h);
    }
  }

  function drawArrowGlyph(cx, cy, fx, fy, color) {
    const mag = Math.hypot(fx, fy);
    if (mag < 1e-6) return;
    const ang = Math.atan2(-fy, fx);
    const len = 14;
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + len * Math.cos(ang), cy + len * Math.sin(ang));
    ctx.stroke();
  }

  function draw(frame) {
    ctx.clearRect(0, 0, CANVAS_PX, CANVAS_PX);
    const bg = ctx.createLinearGradient(0, 0, 0, CANVAS_PX);
    bg.addColorStop(0, "#0a1018"); bg.addColorStop(1, "#0d1c16");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, CANVAS_PX, CANVAS_PX);

    // room boundary
    const [rx0, ry0] = toPx(0, roomH);
    ctx.strokeStyle = "#35424f"; ctx.lineWidth = 1.5;
    ctx.strokeRect(rx0, ry0, roomW * SCALE, roomH * SCALE);

    // zones
    LAYOUT.wind_zones.forEach(z => {
      drawZone(z, "rgba(58,143,214,0.22)");
      const [cx, cy] = toPx(z.x + z.w / 2, z.y + z.h / 2);
      drawArrowGlyph(cx, cy, z.fx, z.fy, "#6fc4ff");
    });
    LAYOUT.accel_zones.forEach(z => {
      drawZone(z, "rgba(224,138,47,0.22)");
      const [cx, cy] = toPx(z.x + z.w / 2, z.y + z.h / 2);
      drawArrowGlyph(cx, cy, z.fx, z.fy, "#ffb15c");
    });
    LAYOUT.decel_zones.forEach(z => drawZone(z, "rgba(138,79,214,0.22)"));

    // walls -- styled as a "danger gate" (red), not a gray factory wall
    LAYOUT.walls.forEach(w => drawZone(w, "rgba(224,80,80,0.65)", "#ff9a9a"));

    // landing pad (pulsing)
    const pulse = 0.5 + 0.5 * Math.sin(simTime * 2.2);
    const [px, py] = toPx(LAYOUT.pad.x, LAYOUT.pad.y);
    const pr = LAYOUT.pad.r * SCALE;
    ctx.beginPath();
    ctx.arc(px, py, pr, 0, 2 * Math.PI);
    ctx.fillStyle = `rgba(53,255,138,${0.15 + 0.15 * pulse})`;
    ctx.fill();
    ctx.strokeStyle = "#35ff8a"; ctx.lineWidth = 1.5; ctx.stroke();

    // progressive flight trail
    ctx.strokeStyle = "#ffb347"; ctx.lineWidth = 2; ctx.beginPath();
    for (let i = 0; i <= frameIdx; i++) {
      const [x, y] = toPx(FRAMES[i].x, FRAMES[i].y);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // wind indicator near drone
    if (frame.wind && (frame.wind[0] !== 0 || frame.wind[1] !== 0)) {
      const [dx, dy] = toPx(frame.x, frame.y);
      drawArrowGlyph(dx, dy - 22, frame.wind[0], frame.wind[1], "#9fe3ff");
    }

    // drone sprite (triangle, rotated by velocity heading)
    const [dx, dy] = toPx(frame.x, frame.y);
    const heading = Math.atan2(-frame.vy, frame.vx || 0.0001);
    ctx.save();
    ctx.translate(dx, dy);
    ctx.rotate(heading);
    ctx.beginPath();
    ctx.moveTo(9, 0); ctx.lineTo(-6, 5); ctx.lineTo(-6, -5); ctx.closePath();
    ctx.fillStyle = frame.crashed ? "#ff4040" : (frame.landed ? "#35ff8a" : "#e8f6ff");
    ctx.fill();
    ctx.restore();

    // HUD
    ctx.fillStyle = "rgba(6,10,16,0.65)";
    ctx.fillRect(4, 4, 132, 92);
    ctx.fillStyle = "#8fd1ff"; ctx.font = "11px Consolas, monospace";
    const lines = [
      `t     ${frame.t.toFixed(2)}s`,
      `X,Y   ${frame.x.toFixed(2)}, ${frame.y.toFixed(2)}`,
      `Vx,Vy ${frame.vx.toFixed(2)}, ${frame.vy.toFixed(2)}`,
      `Speed ${frame.speed.toFixed(2)}`,
      `Act   ${ACTION_ARROWS[frame.action] || "?"}`,
    ];
    lines.forEach((line, i) => ctx.fillText(line, 10, 18 + i * 15));

    // status badge once playback reaches the final frame
    if (frameIdx >= nFrames - 1) {
      const label = FINAL_STATUS === "landed" ? "LANDED" : (FINAL_STATUS === "crashed" ? "CRASHED" : "TIMED OUT");
      const color = FINAL_STATUS === "landed" ? "#35ff8a" : (FINAL_STATUS === "crashed" ? "#ff4040" : "#ffcc55");
      ctx.font = "bold 12px Consolas, monospace";
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(6,10,16,0.75)";
      ctx.fillRect(CANVAS_PX - tw - 20, CANVAS_PX - 28, tw + 16, 20);
      ctx.fillStyle = color;
      ctx.fillText(label, CANVAS_PX - tw - 12, CANVAS_PX - 13);
    }

    caption.textContent = `reward so far: ${frame.reward !== undefined ? frame.reward.toFixed(3) : "-"}  |  frame ${frameIdx + 1}/${nFrames}`;
  }

  function tick(now) {
    if (lastTs === null) lastTs = now;
    const dtReal = (now - lastTs) / 1000;
    lastTs = now;
    if (playing && nFrames > 1) {
      simTime += dtReal * parseFloat(speedSel.value);
      frameIdx = Math.min(nFrames - 1, Math.floor(simTime / DT));
      scrub.value = frameIdx;
      if (frameIdx >= nFrames - 1) playing = false, playBtn.textContent = "▶ Play";
    } else {
      simTime += dtReal * 0.3; // keep pulse animation alive while paused
    }
    draw(FRAMES[frameIdx] || FRAMES[0]);
    requestAnimationFrame(tick);
  }

  playBtn.onclick = () => {
    playing = !playing;
    playBtn.textContent = playing ? "⏸ Pause" : "▶ Play";
    if (playing && frameIdx >= nFrames - 1) { frameIdx = 0; simTime = 0; }
  };
  resetBtn.onclick = () => {
    frameIdx = 0; simTime = 0; scrub.value = 0; playing = nFrames > 1;
    playBtn.textContent = "⏸ Pause";
  };
  scrub.oninput = () => {
    frameIdx = parseInt(scrub.value, 10);
    simTime = frameIdx * DT;
    playing = false;
    playBtn.textContent = "▶ Play";
  };

  if (nFrames > 0) requestAnimationFrame(tick);
})();
</script>
</body>
</html>
"""
