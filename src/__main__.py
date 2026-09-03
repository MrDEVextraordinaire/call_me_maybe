
from .parse_validate import parse_validate_json
from .load_llm import get_vocab
from llm_sdk import Small_LLM_Model


def greedy(    model: Small_LLM_Model,
    prompt_ids: list[int],
    max_tokens: int = 64
) -> str:
	prompt_ids_copy = prompt_ids.copy()
	output = ""
	for _ in range(3):
		logits = model.get_logits_from_input_ids(prompt_ids_copy)
		all_token_indexes = range(len(logits))

 
	def index_to_logit_score(token_index: int) -> float:
		score = logits[token_index]
		return score

	sorted_tokens = sorted(all_token_indexes, key=index_to_logit_score, reverse=True)

	for token_id in sorted_tokens[:8]:
		token_str = model.decode([token_id])
		output = output + token_str
	print(output)

	found_valid_token = False


	pass

def my_encode(model: Small_LLM_Model, text: str) -> list[int]:
	return model.encode(text)[0].tolist()


def allowed_fns(model, prompt, function_defs_data):
	allowed_function_names = [function_def_data.name for function_def_data in function_defs_data]

	function_name_desc = [f"- {f.name}: {f.description}" for f in function_defs_data]
	quick_function_list = "\n".join(function_name_desc)
	print(quick_function_list)

	func_select_prompt = (
		"Pick the function for this request."
    	" Reply with the function name only.\n"
		+ quick_function_list
		+ f"\nRequest: {prompt}\nFunction name: "
	)

	print(func_select_prompt)
	fn_prompt_ids = my_encode(model, func_select_prompt)

	def is_valid(output: str, token_str: str) -> bool:
		combined = 


	selected_name = greedy(model, fn_prompt_ids, max_tokens=42)


def process(model, prompt, function_defs_data):
	function_name = allowed_fns(model, prompt, function_defs_data, )

def main():

	prompts_data, function_defs_data, output_path = parse_validate_json()

	model = Small_LLM_Model()

	for prompt_data in prompts_data:
		res = process(model, prompt_data.prompt, function_defs_data,)
		print()



if __name__ == "__main__":
	main()


"""uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json"""
