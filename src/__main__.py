import json
from pathlib import Path
from typing import Any

from llm_sdk import Small_LLM_Model
from .base_models import FunctionDefinitionItem, PromptItem, Result
from .parse_validate import parse_validate_json
from .constrained_decoding import (
	build_system_prompt,
	generate_function_name,
	generate_number_value,
	generate_boolean_value,
	generate_string_value,
)


def process(
	model: Small_LLM_Model,
	prompt: str,
	function_defs: list[FunctionDefinitionItem],
	system_prompt: str,
) -> Result:
	"""Process a single natural language prompt using constrained decoding."""
	function_names = [f.name for f in function_defs]
	super_prompt = (
		f"{system_prompt}\n"
		f"User:\n{prompt}\n\n"
		f'Assistant:\n{{"name": "'
	)

	input_ids = model.encode(super_prompt)[0].tolist()
	func_name = generate_function_name(model, input_ids, function_names)
	func_def = next((f for f in function_defs if f.name == func_name), function_defs[0])

	super_prompt += f'{func_name}", "parameters": {{'
	parameters: dict[str, Any] = {}
	param_items = list(func_def.parameters.items())

	for i, (param_name, param_info) in enumerate(param_items):
		super_prompt += f'"{param_name}": '
		input_ids = model.encode(super_prompt)[0].tolist()

		if param_info.type == "boolean":
			val = generate_boolean_value(model, input_ids)
			parameters[param_name] = val
			super_prompt += "true" if val else "false"
		elif param_info.type in ("number", "integer"):
			raw_val = generate_number_value(model, input_ids)
			try:
				val = float(raw_val) if param_info.type == "number" else int(float(raw_val))
			except ValueError:
				val = 0.0 if param_info.type == "number" else 0
			parameters[param_name] = val
			super_prompt += raw_val
		elif param_info.type == "string":
			super_prompt += '"'
			input_ids = model.encode(super_prompt)[0].tolist()
			val = generate_string_value(model, input_ids)
			parameters[param_name] = val
			super_prompt += f'{val}"'
		else:
			super_prompt += '"'
			input_ids = model.encode(super_prompt)[0].tolist()
			val = generate_string_value(model, input_ids)
			parameters[param_name] = val
			super_prompt += f'{val}"'

		if i < len(param_items) - 1:
			super_prompt += ", "

	super_prompt += "}}"
	return Result(
		prompt=prompt,
		name=func_name,
		parameters=parameters,
	)


def main() -> None:
	prompts_data, function_defs_data, output_string = parse_validate_json()
	model = Small_LLM_Model()
	system_prompt = build_system_prompt(function_defs_data)

	results: list[dict[str, Any]] = []
	for prompt_data in prompts_data:
		result = process(model, prompt_data.prompt, function_defs_data, system_prompt)
		print(f"  {prompt_data.prompt} -> {result.name}({result.parameters})")
		results.append(result.model_dump())

	output_path = Path(output_string)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with output_path.open(mode="w", encoding="utf-8") as output_file:
		json.dump(results, output_file, indent=2)

	print(f"Wrote {len(results)} result(s) to {output_path}")


if __name__ == "__main__":
	main()
