"""One Qwen3 non-thinking prompt renderer for SFT, preference/RL and inference."""


def render_prompt(tokenizer, prompt):
    if prompt.startswith("<|im_start|>"):
        raise ValueError(
            "Prompt is already chat-rendered; double templating is forbidden"
        )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # The pinned Qwen3 template closes the empty thinking section in the prompt.
    if not rendered.rstrip().endswith("</think>"):
        raise ValueError(
            "Tokenizer did not apply the expected Qwen3 non-thinking template"
        )
    return rendered
