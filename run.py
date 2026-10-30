import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import ChatConfig
from calibration import SimpleTemperatureScaler


class CLIChat:
    def __init__(self, config: ChatConfig):
        # Check for CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This application requires a CUDA-enabled GPU.")

        self.config = config
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
