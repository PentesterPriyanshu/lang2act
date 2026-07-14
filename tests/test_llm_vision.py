"""Smoke test: sim camera frame -> local VLM -> grammar-constrained JSON.

Exercises the full perception pipeline: MuJoCo render -> PNG/base64 ->
llama-server multimodal -> json_schema-constrained reply.

Run: .venv/bin/python -m tests.test_llm_vision  (server must be up)
"""

from lang2act.llm import LLMClient, image_content, text_content
from lang2act.robot import Robot

SCHEMA = {
    "type": "object",
    "properties": {
        "sees_robot_arm": {"type": "boolean"},
        "sees_table": {"type": "boolean"},
        "description": {"type": "string", "maxLength": 200},
    },
    "required": ["sees_robot_arm", "sees_table", "description"],
    "additionalProperties": False,
}


def main() -> int:
    llm = LLMClient()
    assert llm.health(), "llama-server is not up — run scripts/serve.sh"

    robot = Robot(seed=0)
    frame = robot.camera()
    robot.close()

    result = llm.chat(
        [{"role": "user", "content": [
            image_content(frame),
            text_content("Describe this scene. Is there a robot arm? A table?"),
        ]}],
        json_schema=SCHEMA,
    )
    reply = result.json()
    print(f"reply: {reply}")
    print(f"latency: {result.latency_s:.1f}s | tokens: "
          f"{result.prompt_tokens}+{result.completion_tokens}")
    assert reply["sees_robot_arm"], "VLM failed to recognize the robot arm"
    print("VISION PIPELINE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
