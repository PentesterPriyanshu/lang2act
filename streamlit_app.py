"""lang2act live demo — Streamlit Community Cloud.

Two live surfaces:
  1. Agent replay — step through real recorded episodes of the VLM agent
     (Qwen2.5-VL-3B run locally with llama.cpp): every camera frame, thought,
     grammar-constrained tool call, and the verifier's verdict.
  2. Drive the robot — a real MuJoCo simulation runs for your session; move
     the arm with the same primitives the agent composes and try the task.

The live agent is not run here: its 3B vision-language model needs ~40 s per
decision on a small cloud CPU, so a full episode would time out. The replay
tab shows unedited real runs instead; one command in the repo reproduces them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Headless rendering: EGL where a GPU device exists (local dev), software
# rendering (osmesa) on cloud containers.
os.environ.setdefault("MUJOCO_GL", "egl" if os.path.exists("/dev/dri") else "osmesa")

import streamlit as st

ROOT = Path(__file__).parent
ASSETS = ROOT / "space" / "assets"
TASK = "pick up the block and place it on the target marker"
REPO = "https://github.com/PentesterPriyanshu/lang2act"

st.set_page_config(page_title="lang2act — LLM robot agent", page_icon="🦾",
                   layout="wide")

st.title("🦾 lang2act — language-to-action robot agent")
st.markdown(
    f"Natural-language goal in → robot motion out. A **planner / executor / "
    f"verifier** agent stack drives a MuJoCo Fetch arm using **Qwen2.5-VL-3B "
    f"served locally with llama.cpp** — open weights, no cloud APIs, "
    f"reproducible on a 2-core laptop. Code & write-up: [{REPO.removeprefix('https://')}]({REPO})"
)


@st.cache_data
def load_episodes() -> dict[str, dict]:
    episodes = {}
    for ep_dir in sorted(ASSETS.glob("ep_*")):
        if not (ep_dir / "episode.json").is_file():
            continue
        meta = json.loads((ep_dir / "episode.json").read_text())
        label = (f"episode {meta['seed']} — "
                 f"{'✅ success' if meta['env_success'] else '❌ failure'} "
                 f"({len(meta['steps'])} actions, {meta['wall_time_s']:.0f}s)")
        meta["dir"] = str(ep_dir)
        episodes[label] = meta
    return episodes


def frame_path(meta: dict, idx: int) -> str:
    idx = max(0, min(idx, meta["n_frames"] - 1))
    return str(Path(meta["dir"]) / f"f{idx:04d}.jpg")


replay_tab, drive_tab = st.tabs(
    ["🤖 Agent replay (real recorded runs)", "🕹️ Drive the robot (live sim)"])


# --------------------------------------------------------------- replay tab
with replay_tab:
    st.markdown(
        "Unedited recordings of the **actual agent**: every thought, "
        "grammar-constrained tool call, and camera frame, exactly as it ran "
        "against the local model server."
    )
    episodes = load_episodes()
    if not episodes:
        st.info("no recorded episodes bundled")
    else:
        label = st.selectbox("episode", list(episodes))
        meta = episodes[label]
        n_steps = len(meta["steps"])
        step_no = st.slider("agent step", 0, n_steps, 0,
                            help="0 = initial scene; slide right to step through")

        left, right = st.columns([3, 2])
        with left:
            plan = "\n".join(f"{i+1}. {s}" for i, s in enumerate(meta["plan"]))
            st.markdown(f"**Task:** {meta['task']}")
            with st.expander("planner output", expanded=(step_no == 0)):
                st.markdown(plan)
            if step_no == 0:
                st.markdown("*Initial scene — move the slider to step "
                            "through the agent's decisions.*")
                img_idx = 0
            else:
                s = meta["steps"][step_no - 1]
                args = ", ".join(f"{k}={v}" for k, v in s["args"].items())
                st.markdown(f"### step {s['step']}: `{s['tool']}({args})`")
                st.markdown(f"🧠 *{s['thought']}*")
                st.code(s["result"], language=None)
                img_idx = s["frame_end"]
            if step_no == n_steps and n_steps:
                st.divider()
                env_ok = meta["env_success"]
                ver_ok = meta["verifier_success"]
                st.markdown(
                    f"**Environment ground truth:** {'✅ success' if env_ok else '❌ failure'}  \n"
                    f"**Verifier verdict:** {'✅' if ver_ok else '❌'} "
                    f"{meta['verifier_assessment']}"
                )
                if ver_ok and not env_ok:
                    st.caption("⚠️ verifier disagreed with ground truth — "
                               "self-grading bias, discussed in the README")
                st.caption(
                    f"{meta['llm_calls']} LLM calls · "
                    f"{meta['prompt_tokens']}+{meta['completion_tokens']} tokens · "
                    f"{meta['wall_time_s']:.0f}s wall time on a 2-core CPU"
                )
        with right:
            st.image(frame_path(meta, img_idx),
                     caption="robot camera (what the VLM sees)",
                     width='stretch')


# ---------------------------------------------------------------- drive tab
with drive_tab:
    st.markdown(
        f"A **live MuJoCo simulation** runs for your session. Use the same "
        f"primitives the agent uses and try the task yourself: *{TASK}*."
    )

    @st.cache_resource
    def robot_factory():
        # one sim per server process; reset per interaction session
        from lang2act.robot import Robot
        return Robot(seed=None, render_width=384, render_height=384)

    try:
        robot = robot_factory()
    except Exception as e:  # rendering backend missing etc.
        st.error(f"live sim unavailable on this host: {e}")
        robot = None

    if robot is not None:
        if "drive_msg" not in st.session_state:
            st.session_state.drive_msg = "*Press* **new scene** *to start.*"

        controls, view = st.columns([1, 2])
        with controls:
            if st.button("🔄 new scene", width='stretch'):
                robot.reset()
                st.session_state.drive_msg = "*New scene. Get the block onto the red target.*"
            if st.button("go to block", width='stretch'):
                o = robot.object_pos
                st.session_state.drive_msg = f"`go_to(block)` → {robot.go_to(o[0], o[1], o[2])}"
            if st.button("go to target", width='stretch'):
                t = robot.goal_pos
                st.session_state.drive_msg = f"`go_to(target)` → {robot.go_to(t[0], t[1], t[2])}"
            c1, c2 = st.columns(2)
            if c1.button("open grip", width='stretch'):
                st.session_state.drive_msg = f"→ {robot.open_gripper()}"
            if c2.button("close grip", width='stretch'):
                st.session_state.drive_msg = f"→ {robot.close_gripper()}"
            st.markdown("**manual go_to (metres):**")
            x = st.number_input("x", value=1.34, step=0.01, format="%.2f")
            y = st.number_input("y", value=0.75, step=0.01, format="%.2f")
            z = st.number_input("z", value=0.50, step=0.01, format="%.2f")
            if st.button("go_to(x, y, z)", type="primary", width='stretch'):
                st.session_state.drive_msg = f"`go_to` → {robot.go_to(float(x), float(y), float(z))}"
        with view:
            st.image(robot.camera(), caption="robot camera", width=384)
            st.markdown(f"```\n{robot.state_text()}\n```")
            st.markdown(st.session_state.drive_msg)
            if robot.is_success():
                st.success("🎉 TASK SOLVED — block is on the target!")

st.divider()
st.caption(
    f"Why no live LLM here? The agent's 3B vision-language model needs ~40 s "
    f"per decision on a small cloud CPU — a full episode would time out. The "
    f"replay tab shows real runs recorded on the same class of hardware; "
    f"reproducing them is one command in the [repo]({REPO})."
)
