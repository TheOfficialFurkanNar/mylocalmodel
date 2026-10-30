# mylocalmodel

A small CUDA-based command-line chat application for Qwen instruct models. Each turn samples multiple responses, ranks them by temperature-scaled, length-normalized token log-likelihood, and continues the conversation with the highest-scoring response.

## Features

- Uses the Qwen chat template for system prompts and multi-turn conversation history.
- Generates a configurable Best-of-N batch of candidate responses.
- Scores candidates using the mean log-probability of generated tokens after temperature scaling. The geometric mean token probability is also shown as a percentage.
- Optionally prints scores for all candidates and marks the selected response.
- Can load a model from Hugging Face or from a local model directory.
- Drops the oldest conversation turns when the prompt exceeds the configured context budget.

## Requirements

- Python 3.9 or newer.
- A CUDA-enabled NVIDIA GPU and a PyTorch build with CUDA support. The application exits with an error if CUDA is unavailable.
- Enough GPU memory for the selected model and the batched candidates.

Python dependencies are listed in `requirements.txt`:

- `torch`
- `transformers`
- `sentencepiece`
- `tiktoken`

Install a PyTorch build compatible with your CUDA setup by following the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining project dependencies.

## Setup and run

Run these commands from the `mylocalmodel` directory:

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Start the chat:

```bash
python run.py
```

The default model is `Qwen/Qwen2.5-1.5B-Instruct`. On the first run, Transformers downloads it from Hugging Face unless `local_model_path` is set in `config.py` to a local model directory. Type `quit` or `exit` to end the session.

## Configuration

Edit the defaults in `config.py` (`ChatConfig`) before launching. Common settings include:

| Setting | Default | Purpose |
| --- | --- | --- |
| `model_name` | `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face model identifier |
| `local_model_path` | empty | Local model directory; takes precedence over `model_name` |
| `best_of_n` | `4` | Number of candidates generated in one batch |
| `max_length` | `4096` | Prompt context budget, including generation allowance |
| `max_new_tokens` | `1024` | Maximum tokens generated per candidate |
| `temperature` | `0.7` | Sampling temperature used during generation |
| `calibration_temperature` | `0.8` | Temperature used when scoring candidate token probabilities |
| `top_k` / `top_p` | `40` / `0.9` | Sampling filters |
| `show_candidate_scores` | `True` | Whether to print candidate scores and token counts |
| `system_prompt` | reasoning-steps prompt | Asks the model to show its reasoning steps before the final answer and state uncertainty rather than guess |

Lower `best_of_n` or `max_new_tokens` to reduce GPU memory use and generation time. Increasing `best_of_n` samples more candidates, but does not guarantee a better answer; selection relies on the model's own token likelihoods. The system prompt asks the model to show reasoning steps before answering and format mathematical expressions with inline `$...$` or display `$$...$$` LaTeX.

## Project files

- `run.py` — model loading, prompt construction, batched generation, response selection, and interactive CLI.
- `config.py` — model, generation, scoring, and stop-sequence settings.
- `calibration.py` — temperature scaling and length-normalized candidate scoring.
- `requirements.txt` — Python dependencies.
- `LICENSE` — MIT License.

## How candidate selection works

For each sampled candidate, let the prompt be \(x\), and let its generated token sequence be \(y_1, \ldots, y_L\). At generation step \(t\), the model produces a vocabulary-sized score vector \(z_t \in \mathbb{R}^{|V|}\). In this implementation, \(z_t\) is the score vector returned by Transformers during generation, after generation-time processing such as sampling temperature, top-k/top-p filtering, and repetition penalty. The candidate scorer applies a second temperature scaling, using the configured calibration temperature \(T_{\mathrm{calib}} > 0\):

$$
p_t^{(T)}(v) = \operatorname{softmax}\!\left(\frac{z_t}{T_{\mathrm{calib}}}\right)_v
= \frac{\exp(z_{t,v}/T_{\mathrm{calib}})}
       {\sum_{u \in V}\exp(z_{t,u}/T_{\mathrm{calib}})},
\qquad v \in V.
$$

Here, \(p_t^{(T)}(y_t)\) is the probability assigned at step \(t\) to the token that the candidate actually generated. The code sums the log of those probabilities and divides by the number of unmasked generated tokens. If \(M\) is the set of generated positions whose token ID is not the tokenizer's pad token ID, the implemented score is:

$$
S(x,y) = \frac{1}{|M|}\sum_{t \in M}\log p_t^{(T)}(y_t)
       = \frac{1}{|M|}\sum_{t \in M}
         \log\!\left(
           \frac{\exp(z_{t,y_t}/T_{\mathrm{calib}})}
                {\sum_{v \in V}\exp(z_{t,v}/T_{\mathrm{calib}})}
         \right).
$$

Among the \(N\) candidates, the selected response is:

$$
y^* = \underset{y^{(i)},\; i=1,\ldots,N}{\operatorname{arg\,max}}\; S(x,y^{(i)}).
$$

The division by \(|M|\) makes this a mean log-probability per scored token rather than a total log-probability, reducing the automatic preference for shorter sequences that a sum would create. Exponentiating the mean gives the geometric mean of the selected-token probabilities:

$$
G(x,y) = \exp(S(x,y))
        = \left(\prod_{t \in M}p_t^{(T)}(y_t)\right)^{1/|M|}.
$$

This value is what the CLI displays as `geom_mean_prob` (as a percentage). Higher \(T_{\mathrm{calib}}\) softens the distribution and lower values make it peakier; it changes the scores and can change the ranking. Because \(z_t\) already reflects generation-time processing, this score is specifically the likelihood under the processed generation scores after calibration scaling, not an independent evaluation using the model's unmodified logits. It is a model-based ranking signal, not a calibrated measure of factual correctness or answer quality. If the tokenizer has no distinct pad token, the code sets its pad token to EOS; in that case EOS positions are also excluded by the pad-ID mask.

## Notes

- The implementation loads the full model onto the CUDA device; it does not currently configure quantization, CPU offloading, or automatic device mapping.
- EOS ends generation normally. Configured stop strings are removed from decoded output after generation, so generation may continue beyond a configured stop string until EOS or the token limit.
- Conversation history is held in memory for the current run and is not saved between sessions.

## License

This project is distributed under the MIT License. See `LICENSE` for details.
