"""lang2act live demo — Hugging Face Space.

Two live surfaces:
  1. Drive the robot — a real MuJoCo sim per session; visitors move the arm
     with the same primitives the agent uses and try to beat the task.
  2. Agent replay — step through real recorded episodes of the VLM agent
     (Qwen2.5-VL-3B, run locally): every frame, thought, tool call, verdict.

The live agent itself is not run here: a 3B VLM on the free 2-vCPU tier
would take ~20 minutes per episode. Recorded runs are the honest substitute.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# On Spaces there is no EGL device; use software rendering.
if os.environ.get("SPACE_ID"):
    os.environ["MUJOCO_GL"] = "osmesa"

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent)) if (ROOT.parent / "lang2act").is_dir() else None

import gradio as gr
import numpy as np

from lang2act.robot import Robot

ASSETS = ROOT / "assets"
TASK = "pick up the block and place it on the target marker"

REPO = "https://github.com/PentesterPriyanshu/lang2act"
HEADER = f"""
# lang2act — language-to-action robot agent

Natural-language goal in → robot motion out. A planner/executor/verifier agent
stack drives this MuJoCo Fetch arm using **Qwen2.5-VL-3B served locally with
llama.cpp** — open weights, no cloud APIs, runs on a 2-core laptop.
Full write-up and code: **[{REPO.removeprefix('https://')}]({REPO})**
"""


# --------------------------------------------------------------- replay tab

def load_episodes() -> dict[str, dict]:
    episodes = {}
    for ep_dir in sorted(ASSETS.glob("ep_*")):
        if not (ep_dir / "episode.json").is_file():
            continue  # incomplete / mid-recording bundle
        meta = json.loads((ep_dir / "episode.json").read_text())
        label = (f"episode {meta['seed']} — "
                 f"{'success' if meta['env_success'] else 'failure'} "
                 f"({len(meta['steps'])} actions, {meta['wall_time_s']:.0f}s)")
        meta["dir"] = str(ep_dir)
        episodes[label] = meta
    return episodes


EPISODES = load_episodes()


def frame_path(meta: dict, idx: int) -> str:
    idx = max(0, min(idx, meta["n_frames"] - 1))
    return str(Path(meta["dir"]) / f"f{idx:04d}.jpg")


def step_info(meta: dict, step_no: int) -> tuple[str, str]:
    """Markdown for the given executor step (1-indexed; 0 = before start)."""
    plan = "\n".join(f"{i+1}. {s}" for i, s in enumerate(meta["plan"]))
    header = (
        f"**Task:** {meta['task']}\n\n**Planner output:**\n{plan}\n\n---\n"
    )
    if step_no == 0:
        return header + "*Initial scene — move the slider to step through the agent.*", \
               frame_path(meta, 0)
    s = meta["steps"][step_no - 1]
    args = ", ".join(f"{k}={v}" for k, v in s["args"].items())
    body = (
        f"### step {s['step']}: `{s['tool']}({args})`\n"
        f"🧠 *{s['thought']}*\n\n"
        f"↩️ result: `{s['result']}`\n"
    )
    if step_no == len(meta["steps"]):
        body += (
            f"\n---\n**Environment ground truth:** "
            f"{'✅ success' if meta['env_success'] else '❌ failure'}\n\n"
            f"**Verifier verdict:** "
            f"{'✅' if meta['verifier_success'] else '❌'} "
            f"{meta['verifier_assessment']}\n\n"
            f"{meta['llm_calls']} LLM calls · "
            f"{meta['prompt_tokens']}+{meta['completion_tokens']} tokens · "
            f"{meta['wall_time_s']:.0f}s wall time on a 2-core CPU"
        )
    return header + body, frame_path(meta, s["frame_end"])


def replay_update(label: str, step_no: int):
    meta = EPISODES[label]
    md, img = step_info(meta, int(step_no))
    return md, img, gr.update(maximum=len(meta["steps"]))


# ---------------------------------------------------------------- drive tab

def new_robot():
    return Robot(seed=None, render_width=384, render_height=384)


def drive_view(robot: Robot, msg: str):
    badge = "🎉 **TASK SOLVED** — block is on the target!" if robot.is_success() else ""
    return robot.camera(), f"```\n{robot.state_text()}\n```\n{msg}\n\n{badge}"


def drive_reset(robot):
    robot = robot or new_robot()
    robot.reset()
    return robot, *drive_view(robot, "*New scene. Get the block onto the red target.*")


def drive_goto(robot, x, y, z):
    robot = robot or new_robot()
    msg = robot.go_to(float(x), float(y), float(z))
    return robot, *drive_view(robot, f"`go_to` → {msg}")


def drive_goto_block(robot):
    robot = robot or new_robot()
    o = robot.object_pos
    msg = robot.go_to(o[0], o[1], o[2])
    return robot, *drive_view(robot, f"`go_to(block)` → {msg}")


def drive_goto_target(robot):
    robot = robot or new_robot()
    t = robot.goal_pos
    msg = robot.go_to(t[0], t[1], t[2])
    return robot, *drive_view(robot, f"`go_to(target)` → {msg}")


def drive_grip(robot, close: bool):
    robot = robot or new_robot()
    msg = robot.close_gripper() if close else robot.open_gripper()
    return robot, *drive_view(robot, f"→ {msg}")


# ----------------------------------------------------------------------- ui

with gr.Blocks(title="lang2act — LLM robot agent") as demo:
    gr.Markdown(HEADER)

    with gr.Tab("🤖 Agent replay (real recorded runs)"):
        gr.Markdown(
            "These are **unedited recordings of the actual agent**: every "
            "thought, grammar-constrained tool call, and camera frame, as it "
            "ran against the local model server."
        )
        if EPISODES:
            first = next(iter(EPISODES))
            ep_dd = gr.Dropdown(list(EPISODES), value=first, label="episode")
            step_sl = gr.Slider(0, len(EPISODES[first]["steps"]), value=0, step=1,
                                label="agent step")
            with gr.Row():
                md0, img0 = step_info(EPISODES[first], 0)
                info_md = gr.Markdown(md0)
                frame_im = gr.Image(img0, label="robot camera", type="filepath")
            ep_dd.change(replay_update, [ep_dd, step_sl], [info_md, frame_im, step_sl])
            step_sl.change(replay_update, [ep_dd, step_sl], [info_md, frame_im, step_sl])
        else:
            gr.Markdown("*(no recorded episodes bundled)*")

    with gr.Tab("🕹️ Drive the robot yourself (live sim)"):
        gr.Markdown(
            f"A **live MuJoCo simulation** runs for your session. Use the same "
            f"primitives the agent uses and try the task yourself: *{TASK}*."
        )
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=1):
                reset_b = gr.Button("🔄 new scene", variant="secondary")
                blk_b = gr.Button("go to block")
                tgt_b = gr.Button("go to target")
                with gr.Row():
                    open_b = gr.Button("open gripper")
                    close_b = gr.Button("close gripper")
                gr.Markdown("**manual go_to (metres):**")
                x_in = gr.Number(value=1.34, label="x")
                y_in = gr.Number(value=0.75, label="y")
                z_in = gr.Number(value=0.50, label="z")
                goto_b = gr.Button("go_to(x, y, z)", variant="primary")
            with gr.Column(scale=2):
                live_im = gr.Image(label="robot camera", type="numpy")
                live_md = gr.Markdown("*press* **new scene** *to start*")

        outs = [state, live_im, live_md]
        reset_b.click(drive_reset, [state], outs)
        blk_b.click(drive_goto_block, [state], outs)
        tgt_b.click(drive_goto_target, [state], outs)
        open_b.click(lambda r: drive_grip(r, False), [state], outs)
        close_b.click(lambda r: drive_grip(r, True), [state], outs)
        goto_b.click(drive_goto, [state, x_in, y_in, z_in], outs)

    gr.Markdown(
        "---\n*Why no live LLM here? The agent's 3B vision-language model "
        "needs ~40 s per decision on this free 2-vCPU Space — a full episode "
        "would time out. The replay tab shows real runs recorded on the same "
        "class of hardware; the code to reproduce them is one command in the "
        f"[repo]({REPO}).*"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
