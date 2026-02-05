# openrouter_client.py
from openai import OpenAI
import os
import json
import time
import traceback
import Utils


class OpenRouterClient:
    """
    Client wrapper for OpenRouter API.

    Tracks:
      - total_input_tokens   
      - total_output_tokens  
      - total_tokens
      - total_cost_usd (approximate, based on a simple price table)
    """

    def __init__(self, api_key=None, mode="test", log_file=None):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ["OPENROUTER_API_KEY"],
        )

        self.max_retries = 10
        self.mode = mode
        self.log_file = log_file

        # ---- Accumulators for the whole run ----
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0

        # ---- Price table: USD per 1M tokens ----
    
        self.model_price_table = {
            "openai/gpt-4o-mini": {
                "input": 0.15,     # $0.15 per 1M input tokens
                "output": 0.60     # $0.60 per 1M output tokens
            },
            "mistralai/mistral-7b-instruct": {"input": 0.028, "output": 0.054},
            "google/gemma-3-12b-it": {"input": 0.03, "output": 0.10},
            "qwen/qwen-2.5-7b-instruct": {"input": 0.04, "output": 0.10},
            "microsoft/phi-4-reasoning-plus": {"input": 0.07, "output": 0.35},
            "deepseek/deepseek-r1-0528-qwen3-8b": {"input": 0.02, "output": 0.10},
        }

    # -------------------------------------------------
    # Cost computation (per 1M tokens)
    # -------------------------------------------------
    def _estimate_cost(self, model, input_tokens, output_tokens):
        """
        Approximate cost in USD for a single call.
        input_tokens  = tokens you send (prompt)
        output_tokens = tokens model generates (completion)
        """
        prices = self.model_price_table.get(model)
        if not prices:
            return 0.0

        input_rate = prices.get("input", 0.0)     # $ per 1M input tokens
        output_rate = prices.get("output", 0.0)   # $ per 1M output tokens

        cost_in = input_tokens * input_rate / 1_000_000.0
        cost_out = output_tokens * output_rate / 1_000_000.0
        return cost_in + cost_out

    # -------------------------------------------------
    # Main LLM call
    # -------------------------------------------------
    def generate_response(self, model, system_role, prompt, examples=None, temperature=0.0):
        """
        Generates a response from the given model.

        Returns a dict:
        {
          "decision": int,              # 1 / 0 / -1
          "rationale": str,
          "raw_response": str,
          "input_tokens": int,
          "output_tokens": int,
          "total_tokens": int,
          "usd_cost": float
        }
        """

        def _build_messages():
            messages = []
            if system_role:
                messages.append({"role": "system", "content": system_role})

            if examples:
                # Support (input, output) tuples OR dicts with keys 'input','output'
                for ex in examples:
                    if isinstance(ex, (tuple, list)) and len(ex) == 2:
                        user_ex, asst_ex = ex
                    elif isinstance(ex, dict):
                        user_ex = ex.get("input", "")
                        asst_ex = ex.get("output", "")
                    else:
                        continue
                    messages.append({"role": "user", "content": str(user_ex)})
                    messages.append({"role": "assistant", "content": str(asst_ex)})

            messages.append({"role": "user", "content": prompt})
            return messages

        def _send_request():
            """
            Sends the request once and returns:
              raw_text, input_tokens, output_tokens, total_tokens, cost_usd
            """
            messages = _build_messages()

            print(f"[INFO] Sending request to OpenRouter model: {model}")
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )

            choice = completion.choices[0]
            raw_text = (choice.message.content or "").strip()

            usage = getattr(completion, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens) if usage else (input_tokens + output_tokens)

            cost_usd = self._estimate_cost(model, input_tokens, output_tokens)

            # Update accumulators
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_tokens += total_tokens
            self.total_cost_usd += cost_usd

            print(
                f"[COST] model={model} | "
                f"input={input_tokens} output={output_tokens} total={total_tokens} | "
                f"est_cost=${cost_usd:.6f}"
            )

            return raw_text, input_tokens, output_tokens, total_tokens, cost_usd

        # --------- Retry loop ----------
        response_text = ""
        in_tok = out_tok = tot_tok = 0
        cost_usd = 0.0
        parsed = {}

        for attempt in range(self.max_retries + 1):
            try:
                response_text, in_tok, out_tok, tot_tok, cost_usd = _send_request()
                parsed = json.loads(response_text)

                if not isinstance(parsed, dict) or "decision" not in parsed:
                    raise ValueError("Parsed JSON missing 'decision' key")

                break  # success

            except (json.JSONDecodeError, ValueError) as e:
                if attempt < self.max_retries:
                    print(f"[WARN] JSON parse failed (attempt {attempt+1}/{self.max_retries}) → retrying...")
                    time.sleep(1)
                    continue
                else:
                    print("[ERROR] Failed to obtain valid JSON after retries.")
                    parsed = {
                        "decision": "no",
                        "rationale": f"Invalid JSON after {self.max_retries} attempts: {e}"
                    }

            except Exception as e:
                if attempt < self.max_retries:
                    print(f"[WARN] Exception during model call (attempt {attempt+1}/{self.max_retries}): {e}")
                    traceback.print_exc()
                    time.sleep(1)
                    continue
                else:
                    print(f"[ERROR] Exception after {attempt} attempts: {e}")
                    traceback.print_exc()
                    return {
                        "decision": -1,
                        "rationale": f"Exception after {attempt} attempts: {e}",
                        "raw_response": response_text,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "total_tokens": tot_tok,
                        "usd_cost": cost_usd,
                    }

        # --------- Normalize decision ----------
        decision_text = str(parsed.get("decision", "")).strip().lower()
        if decision_text in ("1", "true", "yes"):
            decision = 1
        elif decision_text in ("0", "false", "no"):
            decision = 0
        else:
            decision = -1

        rationale = str(parsed.get("rationale", "")).strip() or "No rationale provided by the model."

        out = {
            "decision": decision,
            "rationale": rationale,
            "raw_response": response_text,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": tot_tok,
            "usd_cost": cost_usd,
        }

        # Optional conversation logging in dev mode
        if self.mode == "dev":
            try:
                print(f"[INFO] Logging conversation to file: {self.log_file}")
                Utils.log_conversation((system_role or "") + (prompt or ""), response_text, self.log_file)
            except Exception:
                pass

        return out