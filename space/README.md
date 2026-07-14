---
title: lang2act — LLM robot agent
emoji: 🦾
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
short_description: Open-weights VLM drives a MuJoCo robot arm
---

# lang2act — live demo

Natural-language goal in → robot motion out. A planner/executor/verifier
agent stack drives a MuJoCo Fetch arm using **Qwen2.5-VL-3B served locally
with llama.cpp** — open weights, no cloud APIs, reproducible on a 2-core
laptop.

This Space hosts:

- **Agent replay** — unedited recordings of the real agent: every thought,
  grammar-constrained tool call, camera frame, and the verifier's verdict.
- **Drive the robot** — a live MuJoCo simulation you control with the same
  motion primitives the agent composes.

Code, evals, and the full write-up:
**[github.com/PentesterPriyanshu/lang2act](https://github.com/PentesterPriyanshu/lang2act)**
