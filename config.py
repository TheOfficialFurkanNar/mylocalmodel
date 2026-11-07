from dataclasses import dataclass, field
from typing import List


@dataclass
class ChatConfig:
    """Configuration for the CLI chat application."""

    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
    gguf_file: str = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    local_model_path: str = r""  # If set, uses local model path instead of downloading from HF
    attention_implementation: str = "sdpa"  # PyTorch fused attention when supported by CUDA hardware
    system_prompt: str = (
        "You are a careful assistant. Answer the user's actual request directly and stay "
        "on topic. Do not introduce unrelated calculations. Use inline LaTeX for any "
        "equations you include. When you provide code, format it as a fenced Markdown "
        "code block and include a language label when appropriate."
    )
    calculator_prompt: str = (
        "For math questions that require calculation, identify the relevant values and "
        "units, then use the calculator before answering. It accepts math expressions "
        "using +, -, *, /, //, %, parentheses, and powers with ** or ^; constants pi, "
        "e, tau; numeric functions sqrt, log, pow, max, min, abs, sin, cos, tan, exp, "
        "ceil, floor, round, sum, prod, mean; and calculus functions diff, integrate, "
        "limit, approx. Trigonometric functions use radians. It does not execute general "
        "Python code. Put exactly one expression, with no prose or assignments, inside "
        "<calculator>...</calculator>. Wait for the result and use it as authoritative; "
        "answer concisely with units."
    )
    max_length: int = 8192
    max_new_tokens: int = 2048
    temperature: float = 0.7  # Generation temperature for sampling
    calibration_temperature: float = 1.143  # Temperature for confidence calibration (T_gen * T_calib = 0.8)
    num_return_sequences: int = 1
    do_sample: bool = True
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    best_of_n: int = 4  # Number of candidate responses to sample and score per turn
    show_candidate_scores: bool = True  # Display candidate scores and ranking in CLI output
    stop_sequences: List[str] = field(default_factory=lambda: [
        "\nInstruction:",
        "\nResponse:",
        "\nUser:",
        "\nQ:",
        "\nHuman:",
        "### Instruction:",
        "### Response:",
        "###",
        "user:",
        "assistant:",
        "User:",
        "Assistant:"
    ])


