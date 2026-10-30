from dataclasses import dataclass, field
from typing import List


@dataclass
class ChatConfig:
    """Configuration for the CLI chat application."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_model_path: str = r""  # If set, uses local model path instead of downloading from HF
    system_prompt: str = (
        "You are a helpful, careful assistant. Show your reasoning steps before giving "
        "your final answer. Be clear and accurate. If you are unsure or lack information, "
        "say so plainly rather than guessing. When writing mathematics, use inline LaTeX "
        "with $...$ or display LaTeX with $$...$$ for equations. For calculations, use "
        "the calculator by writing exactly one math expression inside "
        "<calculator>expression</calculator>. Do not put assignments, print calls, "
        "Python statements, or explanations inside the calculator tags. Wait for the "
        "calculator result before giving a numerical answer."
    )
    max_length: int = 4096
    max_new_tokens: int = 1024
    temperature: float = 0.7  # Generation temperature for sampling
    calibration_temperature: float = 0.8  # Temperature for confidence calibration
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


