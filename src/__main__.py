
from .parse_validate import parse_validate_json
from .load_llm import get_vocab
from llm_sdk import Small_LLM_Model
from .base_models import PromptItem, FunctionDefinitionItem, Result
from typing import Any

def greedy(
	model: Small_LLM_Model,
    prompt_ids: list[int],
    is_valid: callable,
	is_done: callable,
	max_tokens: int = 67
) -> str:
	prompt_ids_copy = prompt_ids.copy()
	output = ""
	for _ in range(max_tokens):
		if is_done(output):
			break
		logits = model.get_logits_from_input_ids(prompt_ids_copy)
		all_token_indexes = range(len(logits))

		def index_to_logit_score(token_index: int) -> float:
			score = logits[token_index]
			return score

		sorted_tokens = sorted(all_token_indexes, key=index_to_logit_score, reverse=True)

		found_valid_token = False
		for token_id in sorted_tokens:
			token_str = model.decode([token_id])
			if (is_valid(output, token_str)):
				output = output + token_str
				prompt_ids_copy.append(token_id)
				found_valid_token = True
				break

		if not found_valid_token:
			break
	return output


def my_encode(model: Small_LLM_Model, text: str) -> list[int]:
	return model.encode(text)[0].tolist()


def allowed_fns(model, prompt, function_defs_data):
	allowed_function_names = [function_def_data.name for function_def_data in function_defs_data]

	function_name_desc = [f"- {f.name}: {f.description}" for f in function_defs_data]
	quick_function_list = "\n".join(function_name_desc)

	func_select_prompt = (
		"Pick the function for this request."
    	" Reply with the function name only.\n"
		+ quick_function_list
		+ f"\nRequest: {prompt}\nFunction name: "
	)
	print(func_select_prompt)

	fn_prompt_ids = my_encode(model, func_select_prompt)

	def is_valid(output: str, token_str: str) -> bool:
		combined = (output + token_str)
		return any(name.startswith(combined) for name in allowed_function_names)

	def is_done(output_progress: str):
		return output_progress.strip() in allowed_function_names


	result = greedy(model, fn_prompt_ids, is_valid, is_done,  max_tokens=42).strip()
	print("result f name: ",result)

	for name in allowed_function_names:
		if result.startswith(name):
			for function in function_defs_data:
				if function.name == name:
					return function

	return functions[0]

def extract_param(
    model: Small_LLM_Model,
    prompt: str,
    param_name: str,
    param_type: str,
) -> Any:
	ALLOWED_NUMBER_CHARACTERS = "0123456789.-"

	extraction_prompt = (
        f"Extract the value of '{param_name}' ({param_type})"
        f" from: {prompt}\nValue: "
    )
	print("e_prompt: ",extraction_prompt)
	prompt_ids = my_encode(model, extraction_prompt)
	if param_type == "number":
		def number_valid(current: str, token_str: str) -> bool: 
			if len(token_str) == 0:
				return False
			for char in token_str:
				if char not in ALLOWED_NUMBER_CHARACTERS:
					return False
			combined = (current  + token_str).strip()
			if combined.count(".") > 1:
					return False
			if "-" in combined:
				if combined.count("-") > 1:
					return False
				if not combined.startswith("-"):
					return False

			return True
		raw_number = greedy(
            model,
            prompt_ids,
            number_valid,
            lambda is_done: False,  # no early stop; rely on constraint
            max_tokens=20,
        )
		try:
			return float(raw_number.strip())
		except ValueError:
			return 0.0

	raw_string = greedy(
		model,
		prompt_ids,
		valid=lambda _current, _token_str: True,
		done=lambda current: "\n" in current,
		max_tokens=64,
	)
	return raw_string.split("\n")[0].strip().strip("\"'")



def process(model, prompt, function_defs_data) -> Result:
	function_name = allowed_fns(model, prompt, function_defs_data,)

	print("param items:", function_name.parameters.items())

	parameters: dict[str, Any] = {}
	for param_name, param_info in function_name.parameters.items():
		print(f"param_name: {param_name} param_info: {param_info} ")
		parameters[param_name] = extract_param(
            model, prompt, param_name, param_info.type
        )
		print("paraaaaaaaaaaaaaaaaaaaaam",parameters)

def main():

	prompts_data, function_defs_data, output_path = parse_validate_json()

	model = Small_LLM_Model()

	for prompt_data in prompts_data:
		res = process(model, prompt_data.prompt, function_defs_data,)
		print("\n\nnext prompt:")



if __name__ == "__main__":
	main()


"""uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json"""
