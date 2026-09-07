import math
from typing import Any
from llm_sdk import Small_LLM_Model
from .base_models import FunctionDefinitionItem


def _argmax(logits: list[float]) -> int:
	"""Return the index of the maximum logit score."""
	best_idx = 0
	best_val = logits[0]
	for i in range(1, len(logits)):
		if logits[i] > best_val:
			best_val = logits[i]
			best_idx = i
	return best_idx


def build_system_prompt(functions: list[FunctionDefinitionItem]) -> str:
	"""Construct the system prompt detailing available functions and extraction rules."""
	lines = [
		"System:",
		"You are a function-calling assistant. Choose the single best function and extract arguments.",
		"",
		"Rules:",
		"- Return ONLY one valid JSON object.",
		"- When a parameter expects a regular expression, write valid Python regex syntax (e.g. \\d+ for digits, \\bword\\b for whole-word matches, [aeiou] for character sets).",
		"- Use exact types: numbers without quotes, strings with quotes.",
		"",
		"Available functions:",
	]
	for f in functions:
		params_list = [f"{k}: {v.type}" for k, v in f.parameters.items()]
		params_str = ", ".join(params_list)
		lines.append(f"- {f.name}({params_str}): {f.description}")
	lines.append("")
	return "\n".join(lines)


def generate_function_name(
	model: Small_LLM_Model,
	input_ids: list[int],
	function_names: list[str],
) -> str:
	"""Select the exact function name using prefix-constrained token decoding."""
	all_f_names_token_ids = [model.encode(name)[0].tolist() for name in function_names]
	quotation_id = model.encode('"')[0].tolist()[0]
	max_tokens = max(len(token_ids_per_f) for token_ids_per_f in all_f_names_token_ids) + 1


	working_ids = list(input_ids)
	generated_tokens: list[int] = []

	for _ in range(max_tokens):
		logits = model.get_logits_from_input_ids(working_ids)

		allowed_tokens: set[int] = set()
		is_complete = False

		for token_ids_per_f in all_f_names_token_ids:
			if token_ids_per_f[: len(generated_tokens)] == generated_tokens:
				if len(token_ids_per_f) > len(generated_tokens):
					allowed_tokens.add(token_ids_per_f[len(generated_tokens)])
				else:
					is_complete = True

		if is_complete: 
			allowed_tokens.add(quotation_id)


		masked: list[float] = [-math.inf] * len(logits)
		for tok_id in allowed_tokens:
			masked[tok_id] = logits[tok_id]
			print('tok_id: ', tok_id)
			print('masked[tok_id]: ', masked[tok_id])

		next_token_id = _argmax(masked)
		if next_token_id == quotation_id:
			break

		generated_tokens.append(next_token_id)
		working_ids.append(next_token_id)

	return model.decode(generated_tokens).strip()


def generate_boolean_value(
	model: Small_LLM_Model,
	input_ids: list[int],
) -> bool:
	"""Determine boolean value (true/false) by comparing logits for true and false tokens."""
	logits = model.get_logits_from_input_ids(input_ids)

	true_ids = [model.encode("true")[0].tolist()[-1], model.encode(" true")[0].tolist()[-1]]
	false_ids = [model.encode("false")[0].tolist()[-1], model.encode(" false")[0].tolist()[-1]]

	true_score = max(logits[tid] for tid in true_ids)
	false_score = max(logits[tid] for tid in false_ids)

	return true_score > false_score


def generate_number_value(
	model: Small_LLM_Model,
	input_ids: list[int],
	max_digits: int = 20,
) -> str:
	"""Generate a number string constrained to numeric characters and stopped by ',' or '}'."""
	working_ids = list(input_ids)
	generated_tokens: list[int] = []
	allowed_ids: set[int] = set()
	stop_ids: set[int] = set()

	for char in "0123456789-.":
		allowed_ids.add(model.encode(char)[0].tolist()[-1])
	for char in "},":
		stop_ids.add(model.encode(char)[0].tolist()[-1])

	allowed_tokens = allowed_ids | stop_ids

	for _ in range(max_digits):
		logits = model.get_logits_from_input_ids(working_ids)
		masked: list[float] = [-math.inf] * len(logits)

		for tok_id in allowed_tokens:
			masked[tok_id] = logits[tok_id]

		next_id = _argmax(masked)
		if next_id in stop_ids:
			break

		generated_tokens.append(next_id)
		working_ids.append(next_id)

	return model.decode(generated_tokens).strip()


def _would_repeat(tokens: list[int], next_id: int, min_period: int = 1, max_period: int = 6) -> bool:
	"""Check if adding next_id produces a periodic repetition pattern."""
	hypothetical = tokens + [next_id]
	for period in range(min_period, max_period + 1):
		if len(hypothetical) < 2 * period:
			continue
		if hypothetical[-period:] == hypothetical[-2 * period : -period]:
			return True
	return False


def generate_string_value(
	model: Small_LLM_Model,
	input_ids: list[int],
	max_tokens: int = 40,
) -> str:
	"""Generate a string value token-by-token until closing quote or stop condition."""
	close_quote_id = model.encode('"')[0].tolist()[-1]
	banned_chars = {'"', "\n", "\r", "”", "“", "‘", "’"}

	working_ids = list(input_ids)
	generated_tokens: list[int] = []

	for _ in range(max_tokens):
		logits = model.get_logits_from_input_ids(working_ids)

		best_id = _argmax(logits)
		best_str = model.decode([best_id])

		if best_id == close_quote_id and len(generated_tokens) > 0:
			break
		if any(c in best_str for c in banned_chars) and len(generated_tokens) > 0:
			break
		if _would_repeat(generated_tokens, best_id):
			break

		generated_tokens.append(best_id)
		working_ids.append(best_id)

	decoded = model.decode(generated_tokens).strip()
	return decoded.replace("\\\\", "\\")
