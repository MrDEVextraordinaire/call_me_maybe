"""Call Me Maybe – constrained-decoding LLM function caller.

Given a natural-language prompt and a set of function definitions,
uses an LLM with constrained decoding to select the correct function
and extract its arguments with the right types.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable
from typing import Any, NoReturn

from pydantic import BaseModel
from llm_sdk import Small_LLM_Model  # type: ignore[import]


# ── Pydantic models ────────────────────────────────────────────────────────

class PromptItem(BaseModel):
    """A single natural-language prompt from the test input file."""

    prompt: str


class Nested(BaseModel):
    """Describes the type of a single function parameter."""

    type: str


class FunctionDefinitionItem(BaseModel):
    """One entry from the functions definition file."""

    name: str
    description: str
    parameters: dict[str, Nested]


class Result(BaseModel):
    """One output record: the original prompt, selected function, and args."""

    prompt: str
    name: str
    parameters: dict[str, Any]


# ── Utilities ──────────────────────────────────────────────────────────────

def die(msg: str) -> NoReturn:
    """Print an error message to stderr and exit with code 1."""
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def load_json(path: str) -> Any:
    """Load and parse a JSON file, exiting on any error."""
    try:
        with open(path) as json_file:
            return json.load(json_file)
    except (OSError, json.JSONDecodeError) as error:
        die(str(error))


def parse_validate_json() -> tuple[
    list[PromptItem], list[FunctionDefinitionItem], str
]:
    """Parse CLI arguments, load both input files, and validate with Pydantic.

    Returns:
        A tuple of (prompts, function definitions, output path).
    """
    parser = argparse.ArgumentParser(description="LLM function calling")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
    )
    args = parser.parse_args()

    raw_prompts = load_json(args.input)
    raw_definitions = load_json(args.functions_definition)

    # Validate each entry against its Pydantic model.
    # This catches missing fields or wrong types early with a clear error.
    prompts_parsed = [PromptItem(**prompt) for prompt in raw_prompts]
    function_def_parsed = [
        FunctionDefinitionItem(**definition) for definition in raw_definitions
    ]

    return prompts_parsed, function_def_parsed, args.output


def encode(model: Small_LLM_Model, text: str) -> list[int]:
    """Convert a text string into a flat list of token IDs.

    The SDK's encode() returns a 2-D tensor of shape [1, seq_len].
    We index [0] to unwrap the batch dimension before converting to a list.
    """
    encoded_tensor = model.encode(text)
    first_sequence = encoded_tensor[0]
    token_ids = first_sequence.tolist()
    return token_ids


# ── Constrained greedy decode ──────────────────────────────────────────────

def greedy(
    model: Small_LLM_Model,
    prompt_ids: list[int],
    is_valid: Callable[[str, str], bool],
    is_done: Callable[[str], bool],
    max_tokens: int = 64,
) -> str:
    """Generate tokens greedily, constrained to tokens that pass valid().

    At every step:
      1. Get logits for the next token from the LLM.
      2. Skip any token whose logit doesn't beat the current best (pruning).
      3. Decode the remaining candidates and check the validity constraint.
      4. Append the best valid token to the output and repeat.

    The logit-based pruning means we only call model.decode() on tokens
    that could beat the current best, keeping the loop fast in practice.

    Args:
        model:       The LLM wrapper.
        prompt_ids:  Starting token ID sequence (the prompt).
        valid:       Returns True if appending token_str is still allowed.
        done:        Returns True when generation should stop.
        max_tokens:  Hard cap on generated tokens.

    Returns:
        The generated string (not including the prompt).
    """

    # ---------------------------------------------------------
    # 1. INITIALIZATION
    # ---------------------------------------------------------
    # Make an independent copy of prompt_ids so appending new tokens
    # during generation does not mutate the caller's original list.
	#COPY
	#INDEX = TOKEN = INDEX = TOKEN
    tokens = list(prompt_ids)

    # Accumulator string for the text generated so far.
    output = ""

    # ---------------------------------------------------------
    # 2. AUTOREGRESSIVE GENERATION LOOP
    # ---------------------------------------------------------
    # Step-by-step token generation capped at `max_tokens` iterations.
    for _ in range(max_tokens):

        # Check if the stopping condition is satisfied (e.g., exact match found).
        generation_finished = is_done(output)
        if generation_finished:
            break

        # -----------------------------------------------------
        # 3. GET NEXT-TOKEN LOGITS (PROBABILITIES)
        # -----------------------------------------------------
        # Forward pass: retrieve unnormalized log-probabilities for every
        # token in the model's vocabulary based on current sequence context.
        logits = model.get_logits_from_input_ids(tokens)

        # Get the total vocabulary size and prepare an index sequence (0 to V-1).
        total_vocab_size = len(logits)
        all_token_indices = range(total_vocab_size)

        # Helper function acting as a sort key to extract the logit value
        # corresponding to a specific token ID.
        def get_logit_score(token_index: int) -> float:
            score = logits[token_index]
            return score

        # -----------------------------------------------------
        # 4. RANK CANDIDATES BY MODEL CONFIDENCE
        # -----------------------------------------------------
        # Sort vocabulary indices in descending order so we evaluate the
        # model's most likely tokens first (greedy exploration).
        sorted_tokens = sorted(
            all_token_indices,
            key=get_logit_score,
            reverse=True,
        )

        # Flag to track whether any candidate passes the grammar constraint.
        found_valid_token = False

        # -----------------------------------------------------
        # 5. CONSTRAINED SEARCH & SELECTION
        # -----------------------------------------------------
        # Iterate through candidate tokens from highest logit to lowest.
        for token_id in sorted_tokens:
            # Wrap token ID in a list for SDK decoding.
            token_list = [token_id]

            # Convert token ID back into human-readable text.
            token_str = model.decode(token_list)

            # Validate whether appending this token keeps the text compliant
            # with allowed prefixes (e.g., valid function name prefixes).
            token_is_valid = is_valid(output, token_str)

            if token_is_valid:
                # Append the validated token string to the accumulated output text.
                output = output + token_str

                # Append the token ID to the context so future forward passes
                # condition on this newly selected token.
                tokens.append(token_id)

                # Mark as resolved and exit candidate search for this step.
                found_valid_token = True
                break

        # If every possible token in the vocabulary violated the constraint,
        # terminate generation early to prevent deadlock or infinite loops.
        if not found_valid_token:
            break

    # ---------------------------------------------------------
    # 6. RETURN RESULT
    # ---------------------------------------------------------
    # Return the accumulated string generated during the constrained search.
    return output


# ── Function-name selection ────────────────────────────────────────────────

def pick_fn(
    model: Small_LLM_Model,
    user_prompt: str,
    functions: list[FunctionDefinitionItem],
) -> FunctionDefinitionItem:
    """Choose which function to call using prefix-constrained decoding.

    The decoder only permits tokens that continue a valid prefix of one of
    the known function names. This guarantees the output is always an exact
    function name — the model cannot hallucinate an unknown one.
    """
    function_names = [function.name for function in functions]

    # Build a bullet list of available functions, one per line.
    # Each line looks like: "- fn_greet: Generate a greeting message..."
    function_lines = []
    for function in functions:
        line = f"- {function.name}: {function.description}"
        function_lines.append(line)
    function_list = "\n".join(function_lines)

    # Assemble the full prompt shown to the LLM.
    selection_prompt = (
        "Pick the function for this request."
        " Reply with the function name only.\n"
        + function_list
        + f"\nRequest: {user_prompt}\nFunction name: "
    )

def is_valid(current: str, token_str: str) -> bool:
# lstrip handles a leading space the tokeniser may add to token 1.
# Allow the token only if the candidate is still a prefix of some name.

	combined = current + token_str
	stripped_candidate = combined.lstrip()

	for name in allowed_function_names:
		starts_with_prefix = name.startswith(stripped_candidate)
		if starts_with_prefix:
			return True

	return False

def is_done(current: str) -> bool:
        stripped_text = current.strip()

        for name in allowed_function_names:
            is_match = stripped_text == name
            if is_match:
                return True

        return False


# ── Parameter extraction ───────────────────────────────────────────────────

# Characters that are valid anywhere inside a JSON number literal.
_NUM_CHARS = frozenset("0123456789.-+eE")


def extract_param(
    model: Small_LLM_Model,
    user_prompt: str,
    param_name: str,
    param_type: str,
) -> Any:
    """Extract a single parameter value using type-constrained decoding.

    For "number" parameters: only tokens whose every character is a valid
    numeric character are allowed. Generation stops when no such token
    remains (the LLM naturally moves on to a space or newline).

    For "string" parameters: all tokens are allowed; generation stops when
    the LLM emits a newline, which is its natural end-of-value signal.
    """
    extraction_prompt = (
        f"Extract the value of '{param_name}' ({param_type})"
        f" from: {user_prompt}\nValue: "
    )
    prompt_ids = encode(model, extraction_prompt)

    if param_type == "number":
        def number_valid(current: str, token_str: str) -> bool:
            # Reject empty tokens and any token containing a non-numeric char.
            return bool(token_str) and all(
                char in _NUM_CHARS for char in token_str
            )

        raw_number = greedy(
            model,
            prompt_ids,
            number_valid,
            lambda _current: False,  # no early stop; rely on constraint
            max_tokens=20,
        )
        try:
            return float(raw_number.strip())
        except ValueError:
            return 0.0

    # Default: treat the parameter as a string.
    def string_valid(current: str, token_str: str) -> bool:
        # Allow any token — the LLM decides the string content freely.
        return True

    def string_done(current: str) -> bool:
        # Stop as soon as the LLM emits a newline (end-of-value signal).
        return "\n" in current

    raw_string = greedy(
        model,
        prompt_ids,
        string_valid,
        string_done,
        max_tokens=64,
    )

    # Keep only the first line and strip surrounding whitespace and quotes.
    return raw_string.split("\n")[0].strip().strip("\"'")


# ── Pipeline ───────────────────────────────────────────────────────────────

def process(
    model: Small_LLM_Model,
    user_prompt: str,
    functions: list[FunctionDefinitionItem],
) -> Result:
    """Run the full pipeline for one prompt: select a function, extract args.

    Args:
        model:       The LLM wrapper.
        user_prompt: The natural-language request to process.
        functions:   All available function definitions.

    Returns:
        A Result containing the prompt, chosen function name, and arguments.
    """
    selected_function = pick_fn(model, user_prompt, functions)

    # Extract each parameter value according to its declared type.
    parameters: dict[str, Any] = {}
    for param_name, param_info in selected_function.parameters.items():
        parameters[param_name] = extract_param(
            model, user_prompt, param_name, param_info.type
        )

    return Result(
        prompt=user_prompt,
        name=selected_function.name,
        parameters=parameters,
    )


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    """Load inputs, run the pipeline for every prompt, and write results."""
    prompts, functions, output_path = parse_validate_json()

    model = Small_LLM_Model()

    results: list[dict[str, Any]] = []
    for prompt_item in prompts:
        result = process(model, prompt_item.prompt, functions)
        print(f"  {prompt_item.prompt[:55]!r} -> {result.name}")
        results.append(result.model_dump())

    # Create the output directory if it does not already exist.
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as output_file:
        json.dump(results, output_file, indent=2)

    print(f"Wrote {len(results)} result(s) to {output_path}")


if __name__ == "__main__":
    main()
