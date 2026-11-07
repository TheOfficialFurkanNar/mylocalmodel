import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import ChatConfig
from calibration import SimpleTemperatureScaler
from tools.calculator import calculate


class CLIChat:
    CALCULATOR_EXAMPLES = [
        (
            "A jacket costs $80 and has a 25% discount. What is the sale price?",
            "<calculator>80 * (1 - 25 / 100)</calculator>",
            "Calculator result for 80 * (1 - 25 / 100): 60",
            "The discount is $80 * 0.25 = $20, so the sale price is $80 - $20 = **$60**.",
        ),
        (
            "There are 5 bags with 12 apples in each bag. Each apple weighs 50 grams. What is the total weight?",
            "<calculator>5 * 12 * 50</calculator>",
            "Calculator result for 5 * 12 * 50: 3000",
            "There are $5 * 12 = 60$ apples. At 50 grams each, their total mass is **3000 grams (3 kilograms)**.",
        ),
        (
            "What is 17% of 240?",
            "<calculator>240 * 17 / 100</calculator>",
            "Calculator result for 240 * 17 / 100: 40.8",
            "17 percent of 240 is $240 * 17 / 100 = 40.8$.",
        ),
    ]

    def __init__(self, config: ChatConfig):
        # Check for CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This application requires a CUDA-enabled GPU.")

        self.config = config
        self.calculator_log_path = Path(__file__).resolve().parent / "calculator_calls.jsonl"
        self.device = torch.device("cuda")
        print(f"Using device: {self.device}")

        # Use local path if provided, otherwise use model name from HF
        model_path = config.local_model_path if config.local_model_path else config.model_name
        print(f"Loading model from: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        self.scaler = SimpleTemperatureScaler(config)
        self.conversation_history = []

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model loaded successfully!\n")

    def build_prompt(self, user_input: str) -> str:
        """Construct multi-turn instruction prompt using Qwen chat template."""
        messages = []
        
        # Add system prompt if configured
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})

        # Show small-model examples of translating word problems into one
        # calculator expression, then grounding the answer in the tool result.
        for question, tool_call, tool_result, answer in self.CALCULATOR_EXAMPLES:
            messages.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": tool_call},
                {"role": "user", "content": f"{tool_result}\nAnswer the question using this result."},
                {"role": "assistant", "content": answer},
            ])
        
        # Append historical turns
        for user_msg, ai_msg in self.conversation_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})
        
        # Append current user prompt
        messages.append({"role": "user", "content": user_input})
        
        # Apply chat template
        full_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # Tokenize and trim context if prompt exceeds max_length
        tokens = self.tokenizer.encode(full_prompt)
        max_allowed = self.config.max_length - self.config.max_new_tokens
        if len(tokens) > max_allowed and len(self.conversation_history) > 0:
            # If history is too long, drop earlier turns
            self.conversation_history.pop(0)
            return self.build_prompt(user_input)

        return full_prompt

    def truncate_at_stop_sequences(self, text: str) -> str:
        """Truncate generated output at the first occurrence of any stop sequence."""
        earliest_idx = len(text)
        for stop_seq in self.config.stop_sequences:
            idx = text.find(stop_seq)
            if idx != -1 and idx < earliest_idx:
                earliest_idx = idx
        return text[:earliest_idx].strip()

    def log_calculator_call(self, question: str, expression: str, result=None, error=None, elapsed_seconds=0.0):
        """Append a local-only calculator audit record; never add it to model messages."""
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "expression": expression,
            "success": error is None,
            "result": result,
            "error": error,
            "elapsed_seconds": round(elapsed_seconds, 6),
        }
        try:
            with self.calculator_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"Warning: Could not write calculator audit log: {exc}")

    def generate_best_of_n_response(self, user_input: str):
        """
        Sample N candidate responses in parallel via GPU batching,
        score each candidate using length-normalized log-likelihood,
        and select the candidate with highest calibrated confidence.
        """
        full_prompt = self.build_prompt(user_input)

        inputs = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        input_length = inputs.shape[1]

        N = self.config.best_of_n
        batched_inputs = inputs.repeat(N, 1)  # Shape: [N, input_length]
        attention_mask = torch.ones_like(batched_inputs)

        with torch.no_grad():
            outputs = self.model.generate(
                batched_inputs,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                no_repeat_ngram_size=3,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )

        sequences = outputs.sequences  # Shape: [N, full_seq_len]

        # Stack logits along dim=1 -> Shape: [N, gen_len, vocab_size]
        stacked_logits = torch.stack(outputs.scores, dim=1)

        candidates = []
        best_index = 0
        best_score = -float('inf')

        for i in range(N):
            gen_tokens = sequences[i, input_length:]
            raw_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            clean_text = self.truncate_at_stop_sequences(raw_text)

            # Score candidate using calibration scaler
            cand_logits = stacked_logits[i]
            score_dict = self.scaler.score_candidate(
                logits=cand_logits,
                generated_tokens=gen_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )

            cand_info = {
                'index': i + 1,
                'text': clean_text,
                'score': score_dict['score'],
                'geom_mean_prob': score_dict['geom_mean_prob'],
                'token_count': score_dict['token_count']
            }
            candidates.append(cand_info)

            if score_dict['score'] > best_score:
                best_score = score_dict['score']
                best_index = i

        winner = candidates[best_index]

        return winner['text'], {
            'winner': winner,
            'all_candidates': candidates
        }

    def chat(self):
        print("=== CLI Chat with Best-of-N Calibrated GPT-2 Instruct Model ===")
        print(
            f"Sampling Best-of-{self.config.best_of_n} candidates per turn (T_gen={self.config.temperature}, T_calib={self.config.calibration_temperature})\n")
        print("Type 'quit' or 'exit' to end the conversation\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            response_text, info = self.generate_best_of_n_response(user_input)

            # Resolve calculator requests and give the result (or error) back to the model.
            last_calculator_error = None
            calculation_succeeded = False
            for _ in range(2):
                tool_call = re.search(r"<calculator>(.*?)</calculator>", response_text, re.DOTALL | re.IGNORECASE)
                if tool_call is None:
                    break

                expression = tool_call.group(1).strip()
                calculation_started = time.perf_counter()
                try:
                    tool_result = calculate(expression)
                    feedback = f"Calculator result for {expression}: {tool_result}"
                    calculation_succeeded = True
                    last_calculator_error = None
                    self.log_calculator_call(
                        user_input, expression, result=tool_result,
                        elapsed_seconds=time.perf_counter() - calculation_started,
                    )
                except Exception as exc:
                    last_calculator_error = str(exc)
                    calculation_succeeded = False
                    feedback = f"Calculator error for {expression}: {exc}"
                    self.log_calculator_call(
                        user_input, expression, error=str(exc),
                        elapsed_seconds=time.perf_counter() - calculation_started,
                    )

                if calculation_succeeded:
                    instruction = (
                        "Use the calculator result as authoritative. Do not recalculate it. "
                        "Answer the original question directly."
                    )
                else:
                    instruction = (
                        "Retry once using a single valid math expression inside "
                        "<calculator>expression</calculator>. Do not use assignments, "
                        "print calls, or other Python statements. Do not give a numeric "
                        "answer unless the calculator succeeds."
                    )
                follow_up = (
                    f"{user_input}\n\n{feedback}\n"
                    f"{instruction}"
                )
                response_text, info = self.generate_best_of_n_response(follow_up)

            if last_calculator_error is not None and not calculation_succeeded:
                response_text = (
                    "I couldn't verify a numerical answer because the calculator request "
                    f"failed: {last_calculator_error}"
                )

            response_text = re.sub(r"<calculator>.*?</calculator>", "[calculator request could not be completed]", response_text, flags=re.DOTALL | re.IGNORECASE)

            if self.config.show_candidate_scores:
                print(f"\n--- Best-of-{self.config.best_of_n} Candidate Scores ---")
                for cand in info['all_candidates']:
                    is_winner = "*" if cand['index'] == info['winner']['index'] else " "
                    print(
                        f" {is_winner} Candidate {cand['index']}: Log-Likelihood = {cand['score']:.4f} | Geom Mean Prob = {cand['geom_mean_prob'] * 100:.1f}% ({cand['token_count']} tokens)")
                print("-" * 55)

            print(f"AI: {response_text}\n")

            # Record turn in history as tuple (User, AI) - ONLY winner is saved
            self.conversation_history.append((user_input, response_text))


def main():
    config = ChatConfig()
    chat = CLIChat(config)
    chat.chat()


if __name__ == "__main__":
    main()
