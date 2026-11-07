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
        "You are a careful assistant. For calculations, show each task-relevant solution "
        "step in a visible <think>...</think> block: identify the given values and units, "
        "translate their relationships into an expression, explain any unit conversion, "
        "and interpret the calculator result. The calculator accepts math expressions "
        "with +, -, *, /, //, %, "
        "parentheses, ** or ^ for powers; constants pi, e, tau; numeric functions "
        "sqrt, log, pow, max, min, abs, sin, cos, tan, exp, ceil, floor, round, sum, "
        "prod, mean; and symbolic calculus functions diff, integrate, limit, and approx. "
        "Trig functions use radians. It does not run general Python statements. Call it "
        "with one expression only inside <calculator>...</calculator> (no prose or code "
        "in the calculator tag). Wait for the result and use it as authoritative. Give "
        "the answer with units, and format every equation you explain as inline LaTeX "
        "using $...$."
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


