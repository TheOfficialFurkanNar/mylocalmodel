from dataclasses import dataclass, field
from typing import List


@dataclass
class ChatConfig:
    """Configuration for the CLI chat application."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_model_path: str = r""  # If set, uses local model path instead of downloading from HF
    system_prompt: str = (
        "You are a helpful, careful assistant. The calculator evaluates arithmetic "
        "expressions using +, -, *, /, //, %, parentheses, and powers (** or ^); "
        "constants pi, e, and tau; numeric functions sqrt, log, pow, max, min, abs, "
        "sin, cos, tan, exp, ceil, floor, round, sum, prod, and mean; and symbolic "
        "calculus with diff(expr, variable[, order]), integrate(expr, variable) or "
        "integrate(expr, variable, lower, upper), limit(expr, variable, value), and "
        "approx(expr[, digits]). Trigonometric functions use radians. For a calculation, "
        "translate the question into one expression and call the tool using only "
        "<calculator>expression</calculator> (no prose, assignments, or Python code); "
        "wait for its result and use that result in your answer. Explain the calculation "
        "briefly, and format every equation you explain as inline LaTeX with $...$."
    )
    max_length: int = 8192
    max_new_tokens: int = 2048
    temperature: float = 0.7  # Generation temperature for sampling
    calibration_temperature: float = 0.8  # Temperature for confidence calibration
    num_return_sequences: int = 1
    do_sample: bool = True
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    best_of_n: int = 8  # Number of candidate responses to sample and score per turn
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


