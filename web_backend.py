import base64
import importlib.metadata
import json
import math
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from calibration import SimpleTemperatureScaler
from config import ChatConfig
from tools.calculator import calculate
from tools.rag import get_rag
from tools.tts import get_tts, list_voices as tts_list_voices


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ConversationStore:
    """Small local JSON store for conversations and turn metadata."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {"version": 1, "conversations": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded.get("conversations"), dict):
                    self.data = loaded
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Warning: Could not load conversation history: {exc}")

    def _save(self):
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(self.path)

    def create(self):
        now = utc_now()
        conversation = {
            "id": str(uuid.uuid4()),
            "title": "New chat",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        with self.lock:
            self.data["conversations"][conversation["id"]] = conversation
            self._save()
        return conversation

    def get(self, conversation_id: str):
        with self.lock:
            conversation = self.data["conversations"].get(conversation_id)
            return json.loads(json.dumps(conversation)) if conversation else None

    def delete(self, conversation_id: str):
        with self.lock:
            if self.data["conversations"].pop(conversation_id, None) is None:
                return False
            self._save()
            return True

    def list(self):
        with self.lock:
            conversations = list(self.data["conversations"].values())
            return [
                {
                    "id": item["id"],
                    "title": item["title"],
                    "created_at": item["created_at"],
                    "updated_at": item["updated_at"],
                    "message_count": len(item["messages"]),
                }
                for item in sorted(
                    conversations, key=lambda item: item["updated_at"], reverse=True
                )
            ]

    def append_turn(self, conversation_id: str, user_text: str, assistant_text: str, metadata: dict):
        now = utc_now()
        with self.lock:
            conversation = self.data["conversations"].get(conversation_id)
            if conversation is None:
                return None
            if not conversation["messages"]:
                title = re.sub(r"\s+", " ", user_text).strip()
                conversation["title"] = title[:58] + ("…" if len(title) > 58 else "")
            conversation["messages"].extend([
                {
                    "id": str(uuid.uuid4()),
                    "role": "user",
                    "content": user_text,
                    "timestamp_utc": now,
                },
                {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": assistant_text,
                    "timestamp_utc": now,
                    "metadata": metadata,
                },
            ])
            conversation["updated_at"] = now
            self._save()
            return json.loads(json.dumps(conversation))


class ChatModel:
    CALCULATOR_EXAMPLES = [
        (
            "A jacket costs $80, has a 25% discount, and then 8% sales tax is applied to the discounted price. What is the final price?",
            "<calculator>80 * (1 - 25 / 100) * (1 + 8 / 100)</calculator>",
            "Calculator result for 80 * (1 - 25 / 100) * (1 + 8 / 100): 64.8",
            "The 25% discount gives $80 * (1 - 25/100) = $60$. Tax then adds $60 * 8/100 = $4.80$, so the final price is **$64.80**.",
        ),
        (
            "What is 17% of 240?",
            "<calculator>240 * 17 / 100</calculator>",
            "Calculator result for 240 * 17 / 100: 40.8",
            "17 percent of 240 is $240 * 17/100 = 40.8$.",
        ),
    ]

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This application requires a CUDA-enabled GPU.")

        self.config = ChatConfig()
        self.device = torch.device("cuda")
        self.calculator_log_path = Path(__file__).resolve().parent / "calculator_calls.jsonl"
        self.conversation_store = ConversationStore(
            Path(__file__).resolve().parent / "conversation_history.json"
        )
        print(f"Using device: {self.device}")

        model_path = self.config.local_model_path or self.config.model_name
        print(f"Loading model from: {model_path}...")
        load_kwargs = {} if self.config.local_model_path else {"gguf_file": self.config.gguf_file}
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, **load_kwargs
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation=self.config.attention_implementation,
            **load_kwargs,
        ).to(self.device)
        self.scaler = SimpleTemperatureScaler(self.config)

        # Initialize RAG retriever if enabled
        if self.config.enable_rag:
            print("Initializing RAG retriever...")
            try:
                self.rag = get_rag(device=self.device)
            except Exception as exc:
                print(f"Warning: Could not initialize RAG retriever: {exc}")
                self.rag = None
        else:
            self.rag = None

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model loaded successfully!")

    def build_prompt(self, user_input: str, history: list = None, rag_context: str = None) -> str:
        messages = []
        include_calculator = self.question_requires_calculation(user_input)
        if self.config.system_prompt:
            system_content = self.config.system_prompt
            if include_calculator:
                system_content += " " + self.config.calculator_prompt
            messages.append({"role": "system", "content": system_content})
        if include_calculator:
            for question, tool_call, tool_result, answer in self.CALCULATOR_EXAMPLES:
                messages.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": tool_call},
                    {"role": "user", "content": f"{tool_result}\nUse this result to answer."},
                    {"role": "assistant", "content": answer},
                ])

        history = list(history or [])
        for user_message, assistant_message in history:
            messages.extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ])

        user_content = f"{rag_context}\n\n{user_input}" if rag_context else user_input
        messages.append({"role": "user", "content": user_content})

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if len(self.tokenizer.encode(prompt)) > self.config.max_length - self.config.max_new_tokens and history:
            return self.build_prompt(user_input, history[1:], rag_context=rag_context)
        return prompt

    def generate_best_of_n_response(self, user_input: str, history: list = None, require_calculator=False, rag_context: str = None):
        prompt = self.build_prompt(user_input, history, rag_context=rag_context)
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.shape[1]
        candidate_count = self.config.best_of_n
        batched_inputs = inputs.repeat(candidate_count, 1)
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
                output_scores=True,
            )

        sequences = outputs.sequences
        logits = torch.stack(outputs.scores, dim=1)
        candidates = []
        for index in range(candidate_count):
            generated_tokens = sequences[index, input_length:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            text = self.truncate_at_stop_sequences(text)
            score = self.scaler.score_candidate(
                logits=logits[index],
                generated_tokens=generated_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            candidates.append({
                "index": index + 1,
                "text": text,
                "score": score["score"],
                "geom_mean_prob": score["geom_mean_prob"],
                "token_count": score["token_count"],
            })

        tool_pattern = re.compile(r"<calculator>.*?</calculator>", re.DOTALL | re.IGNORECASE)
        eligible = [candidate for candidate in candidates if tool_pattern.search(candidate["text"])]
        winner_pool = eligible if require_calculator and eligible else candidates
        winner = max(winner_pool, key=lambda candidate: candidate["score"])
        candidate_run = {
            "accepted_candidate": winner["index"],
            "candidate_count": candidate_count,
            "prompt_tokens": input_length,
            "candidates": [
                {
                    "index": candidate["index"],
                    "score": candidate["score"] if math.isfinite(candidate["score"]) else None,
                    "geom_mean_prob": candidate["geom_mean_prob"] if math.isfinite(candidate["geom_mean_prob"]) else None,
                    "token_count": candidate["token_count"],
                    "selected": candidate["index"] == winner["index"],
                }
                for candidate in candidates
            ],
        }
        return winner["text"], candidate_run

    def truncate_at_stop_sequences(self, text: str) -> str:
        earliest = len(text)
        for stop in self.config.stop_sequences:
            index = text.find(stop)
            if index != -1 and index < earliest:
                earliest = index
        return text[:earliest].strip()

    @staticmethod
    def question_requires_calculation(question: str) -> bool:
        has_number = re.search(r"\d", question) is not None
        math_intent = re.search(
            r"\b(?:calculate|compute|how many|how much|total|sum|average|mean|"
            r"percent(?:age)?|discount|tax|cost|price|weigh(?:s|ed|ing)?|weight|"
            r"speed|distance|product|quotient|times|multiplied|divided|squared|"
            r"square root|plus|minus)\b|[-+*/^=]",
            question,
            re.IGNORECASE,
        ) is not None
        return has_number and math_intent

    def log_calculator_call(self, question: str, expression: str, result=None, error=None, elapsed=0.0):
        record = {
            "timestamp_utc": utc_now(),
            "question": question,
            "expression": expression,
            "success": error is None,
            "result": result,
            "error": error,
            "elapsed_seconds": round(elapsed, 6),
        }
        try:
            with self.calculator_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"Warning: Could not write calculator audit log: {exc}")

    def respond(self, question: str, history: list = None):
        started_at = utc_now()
        started = time.perf_counter()
        generation_runs = []

        # Automatic RAG Pre-Retrieval
        rag_matches = []
        rag_context = None
        if self.config.enable_rag and self.rag is not None:
            try:
                rag_matches = self.rag.retrieve(
                    question,
                    top_k=self.config.rag_top_k,
                    max_distance=self.config.rag_max_distance,
                )
                if rag_matches:
                    rag_context = self.rag.format_context(rag_matches)
            except Exception as rag_err:
                print(f"Warning: RAG retrieval failed: {rag_err}")

        response, candidate_run = self.generate_best_of_n_response(
            question, history, rag_context=rag_context
        )
        candidate_run["stage"] = "initial"
        generation_runs.append(candidate_run)
        needs_calculator = self.question_requires_calculation(question)
        tool_pattern = re.compile(r"<calculator>(.*?)</calculator>", re.DOTALL | re.IGNORECASE)

        if needs_calculator and not tool_pattern.search(response):
            retry_prompt = (
                f"{question}\n\nUse the calculator before answering. Output exactly one "
                "math expression inside <calculator>...</calculator>, with no prose or "
                "Python statements. Do not answer yet."
            )
            response, candidate_run = self.generate_best_of_n_response(
                retry_prompt, history, require_calculator=True
            )
            candidate_run["stage"] = "calculator_retry"
            generation_runs.append(candidate_run)

        tool_calls = []
        calculation_succeeded = False
        last_error = None
        for attempt in range(2 if needs_calculator else 0):
            match = tool_pattern.search(response)
            if match is None:
                break
            expression = match.group(1).strip()
            calc_started = time.perf_counter()
            try:
                result = calculate(expression)
                elapsed = time.perf_counter() - calc_started
                self.log_calculator_call(question, expression, result=result, elapsed=elapsed)
                tool_calls.append({
                    "tool": "calculator",
                    "expression": expression,
                    "result": str(result),
                    "success": True,
                    "elapsed_ms": round(elapsed * 1000),
                })
                calculation_succeeded = True
                last_error = None
                follow_up = (
                    f"{question}\n\nCalculator result for {expression}: {result}\n"
                    "Use this result as authoritative. Do not recalculate. Answer the original question."
                )
                stage = "final_answer"
            except Exception as exc:
                elapsed = time.perf_counter() - calc_started
                last_error = str(exc)
                self.log_calculator_call(question, expression, error=last_error, elapsed=elapsed)
                tool_calls.append({
                    "tool": "calculator",
                    "expression": expression,
                    "error": last_error,
                    "success": False,
                    "elapsed_ms": round(elapsed * 1000),
                })
                calculation_succeeded = False
                follow_up = (
                    f"{question}\n\nCalculator error for {expression}: {last_error}\n"
                    "Retry once with one valid math expression inside <calculator>...</calculator>. "
                    "Do not give a numerical answer unless the calculation succeeds."
                )
                stage = f"calculator_repair_{attempt + 1}"

            response, candidate_run = self.generate_best_of_n_response(follow_up, history)
            candidate_run["stage"] = stage
            generation_runs.append(candidate_run)

        if needs_calculator and not calculation_succeeded and last_error is None:
            last_error = "the model did not produce a calculator request"
        if last_error is not None and not calculation_succeeded:
            response = f"I couldn't verify a numerical answer because the calculator request failed: {last_error}"

        response = tool_pattern.sub("", response).strip()
        response = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r"<think\b[^>]*>.*$", "", response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r"</?think\s*>", "", response, flags=re.IGNORECASE).strip()

        accepted = generation_runs[-1] if generation_runs else None
        accepted_candidate = None
        if accepted:
            accepted_candidate = next(
                (candidate for candidate in accepted["candidates"] if candidate["selected"]), None
            )
            if accepted_candidate:
                accepted_candidate = {
                    "stage": accepted["stage"],
                    **{key: accepted_candidate[key] for key in ("index", "score", "geom_mean_prob", "token_count")},
                }

        metadata = {
            "timestamp_utc": started_at,
            "duration_seconds": round(time.perf_counter() - started, 6),
            "model_name": self.config.local_model_path or self.config.model_name,
            "gguf_file": None if self.config.local_model_path else self.config.gguf_file,
            "attention_implementation": self.config.attention_implementation,
            "tool_used": bool(tool_calls),
            "tool_calls": tool_calls,
            "rag_used": bool(rag_matches),
            "rag_matches": rag_matches,
            "max_length": self.config.max_length,
            "max_new_tokens": self.config.max_new_tokens,
            "best_of_n": self.config.best_of_n,
            "accepted_candidate": accepted_candidate,
            "candidate_runs": generation_runs,
            "generation": {
                "temperature": self.config.temperature,
                "calibration_temperature": self.config.calibration_temperature,
                "top_k": self.config.top_k,
                "top_p": self.config.top_p,
                "do_sample": self.config.do_sample,
                "repetition_penalty": self.config.repetition_penalty,
            },
        }
        return response, metadata


chat_model = None
conversation_store = ConversationStore(Path(__file__).resolve().parent / "conversation_history.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chat_model
    print("Loading model on startup...")
    try:
        kokoro_version = importlib.metadata.version("kokoro-onnx")
        print(f"kokoro-onnx version: {kokoro_version}")
    except importlib.metadata.PackageNotFoundError:
        print("kokoro-onnx version: not installed")
    chat_model = ChatModel()
    yield
    print("Shutting down...")


app = FastAPI(title="Local Qwen Chat API", lifespan=lifespan)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/conversations")
async def list_conversations():
    return {"conversations": conversation_store.list()}


@app.post("/conversations")
async def create_conversation():
    return conversation_store.create()


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    conversation = conversation_store.get(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    if not conversation_store.delete(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True, "conversation_id": conversation_id}


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        conversation = (
            conversation_store.get(request.conversation_id)
            if request.conversation_id else conversation_store.create()
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        history = []
        messages = conversation["messages"]
        for index in range(0, len(messages) - 1, 2):
            user_message, assistant_message = messages[index:index + 2]
            if user_message["role"] == "user" and assistant_message["role"] == "assistant":
                history.append([user_message["content"], assistant_message["content"]])

        response, metadata = chat_model.respond(request.message, history)
        conversation_store.append_turn(request.conversation_id or conversation["id"], request.message, response, metadata)
        return {
            "conversation_id": conversation["id"],
            "response": response,
            "metadata": metadata,
        }
    except HTTPException:
        raise
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": chat_model is not None}


@app.get("/tts/voices")
async def tts_voices():
    """Return the available Kokoro TTS voice catalogue."""
    return {"voices": tts_list_voices()}


@app.get("/tts/stream")
async def tts_stream(
    text: str = Query(..., description="Text to synthesise"),
    voice: str = Query("af_heart", description="Kokoro voice id"),
):
    """
    Stream synthesised speech as Server-Sent Events.

    Each SSE ``data:`` event carries a base64-encoded WAV blob for one sentence
    chunk (24 kHz, 16-bit PCM mono).  A final ``event: done`` signals completion.
    The client should decode each chunk and schedule it for sequential Web Audio
    API playback.
    """
    tts = get_tts()

    async def generate():
        try:
            async for samples, sample_rate in tts.synthesize_stream(text, voice=voice):
                wav_bytes = tts.chunk_to_wav_bytes(samples, sample_rate)
                b64 = base64.b64encode(wav_bytes).decode("ascii")
                yield f"data: {b64}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {exc}\n\n"
        finally:
            yield "event: done\ndata: end\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse(str(static_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
