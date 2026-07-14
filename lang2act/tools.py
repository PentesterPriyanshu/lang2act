"""Tool surface exposed to the executor agent.

Every action the model can take is a typed tool with a JSON-schema signature.
The executor's replies are grammar-constrained to ACTION_SCHEMA, so even a
3B model cannot emit an unparseable or unknown action.
"""

from __future__ import annotations

from dataclasses import dataclass

from .robot import Robot

TOOL_DOCS = """\
Available tools:
- look(): returns the current camera image and numeric scene state.
- go_to(x, y, z): move the gripper to coordinates in metres. The workspace is
  roughly x in [1.15, 1.50], y in [0.55, 0.95], z in [0.42, 0.70]; the table
  surface is at z=0.425. Approach the block at its own z before grasping.
- open_gripper(): open the fingers.
- close_gripper(): close the fingers (grasp whatever is between them).
- done(reason): finish the episode and report what you achieved.

Important: if the target z is above the table surface (z > 0.47), the block
must be HELD at the target — do not open the gripper or it will fall. Call
done() while holding it in place. Only release for table-level targets.
"""

# Discriminated union: picking a tool forces that tool's arguments, so the
# sampler cannot produce e.g. go_to without coordinates.
_THOUGHT = {"type": "string", "maxLength": 300}
ACTION_SCHEMA = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "thought": _THOUGHT,
                "tool": {"enum": ["look", "open_gripper", "close_gripper"]},
            },
            "required": ["thought", "tool"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "thought": _THOUGHT,
                "tool": {"enum": ["go_to"]},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
            "required": ["thought", "tool", "x", "y", "z"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "thought": _THOUGHT,
                "tool": {"enum": ["done"]},
                "reason": {"type": "string", "maxLength": 200},
            },
            "required": ["thought", "tool", "reason"],
            "additionalProperties": False,
        },
    ]
}


@dataclass
class ToolResult:
    text: str
    image: object | None = None  # numpy RGB array when the tool returns vision
    is_done: bool = False


def dispatch(robot: Robot, action: dict) -> ToolResult:
    """Execute one validated action dict against the robot."""
    tool = action["tool"]
    if tool == "look":
        return ToolResult(text=robot.state_text(), image=robot.camera())
    if tool == "go_to":
        missing = [k for k in ("x", "y", "z") if k not in action]
        if missing:
            return ToolResult(text=f"error: go_to needs x, y, z (missing {missing})")
        return ToolResult(text=robot.go_to(action["x"], action["y"], action["z"]))
    if tool == "open_gripper":
        return ToolResult(text=robot.open_gripper())
    if tool == "close_gripper":
        return ToolResult(text=robot.close_gripper())
    if tool == "done":
        return ToolResult(text=action.get("reason", "done"), is_done=True)
    return ToolResult(text=f"error: unknown tool {tool!r}")
