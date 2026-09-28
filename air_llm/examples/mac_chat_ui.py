"""Local browser chat UI for AirLLM on MacOS (Apple silicon, MLX backend).

    pip install -e "./air_llm[mac]" gradio
    python air_llm/examples/mac_chat_ui.py            # then open http://127.0.0.1:7860

The model stays loaded between messages and replies stream token by token. Only Llama-architecture
chat models are supported on MacOS. Set HF_TOKEN in the environment for gated models.
"""

import argparse
import os
import threading

import gradio as gr

from airllm import AutoModel
from airllm.airllm_llama_mlx import to_mx_array

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Only one model fits comfortably in unified memory, so keep a single loaded instance and swap it
# when the user picks another one.
_loaded = {"name": None, "model": None}
_load_lock = threading.Lock()


def get_model(name):
    with _load_lock:
        if _loaded["name"] != name:
            _loaded["model"] = None
            _loaded["model"] = AutoModel.from_pretrained(name, hf_token=os.environ.get("HF_TOKEN"))
            _loaded["name"] = name
        return _loaded["model"]


def _text(content):
    """Gradio message content is a string or a list of parts; keep the text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def build_input_ids(tokenizer, messages, context_limit):
    """Apply the chat template, dropping the oldest turns until the prompt fits."""
    system = [m for m in messages if m["role"] == "system"]
    turns = [m for m in messages if m["role"] != "system"]
    while True:
        ids = tokenizer.apply_chat_template(system + turns, add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True)["input_ids"]
        if ids.shape[1] <= context_limit or len(turns) <= 1:
            return ids
        turns = turns[2:] if len(turns) > 2 else turns[-1:]


def chat(message, history, model_name, system_prompt, max_new_tokens, temperature):
    model_name = (model_name or DEFAULT_MODEL).strip()
    if _loaded["name"] != model_name:
        yield f"Loading `{model_name}`. The first run downloads and splits it, which can take a while..."
    model = get_model(model_name)
    tokenizer = model.tokenizer

    messages = [{"role": "system", "content": system_prompt}] if system_prompt.strip() else []
    messages += [{"role": m["role"], "content": _text(m["content"])} for m in history]
    messages.append({"role": "user", "content": _text(message)})

    max_new_tokens = int(max_new_tokens)
    context = getattr(model.config, "max_position_embeddings", 2048)
    input_ids = build_input_ids(tokenizer, messages, context - max_new_tokens)

    tokens = []
    yield "..."
    for token in model.model_generate(to_mx_array(input_ids), temperature=float(temperature)):
        token_id = token.item()
        if token_id == tokenizer.eos_token_id:
            break
        tokens.append(token_id)
        yield tokenizer.decode(tokens, skip_special_tokens=True)
        if len(tokens) >= max_new_tokens:
            break
    if not tokens:
        yield "(the model ended its turn without replying)"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or local path")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other devices on your network")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--preload", action="store_true", help="load the model before the UI starts")
    args = parser.parse_args()

    if args.preload:
        get_model(args.model)

    demo = gr.ChatInterface(
        chat,
        title="AirLLM chat (MacOS)",
        description="Layers stream from disk one at a time, so large models fit in little memory "
                    "but each token takes a while. Replies stream as they are generated.",
        additional_inputs=[
            gr.Dropdown([DEFAULT_MODEL], value=args.model, allow_custom_value=True,
                        label="Model (Llama-architecture Hugging Face id or local path)"),
            gr.Textbox("", label="System prompt", lines=2),
            gr.Slider(1, 1024, value=128, step=1, label="Max new tokens"),
            gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="Temperature (0 = deterministic)"),
        ],
        additional_inputs_accordion=gr.Accordion("Settings", open=False),
        examples=[["What is the capital of United States?"], ["Write a haiku about the ocean."]],
        run_examples_on_click=False,
        # One generation at a time: layers stream through shared memory.
        concurrency_limit=1,
        analytics_enabled=False,
    )
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
