import argparse
from .base_models import PromptItem, FunctionDefinitionItem
import json

def load_json(file_path_string: str):
	try:
		with open(file_path_string, "r") as f:
			file_path_string = json.load(f)
	except (OSError, json.JSONDecodeError) as e:
		print(e)
	return file_path_string


def parse_validate_json() -> tuple[list[FunctionDefinitionItem], list[FunctionDefinitionItem], str]:
	parser = argparse.ArgumentParser()
	parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
	parser.add_argument("--input", default="data/input/function_calling_tests.json")
	parser.add_argument("--output", default="data/output/output_file.json")
	args = parser.parse_args()

	prompts = load_json(args.input)
	definitions = load_json(args.functions_definition)

	function_def_parsed = [FunctionDefinitionItem(**definition) for definition in definitions]
	prompts_parsed = [PromptItem(**prompt) for prompt in prompts]

	return prompts_parsed, function_def_parsed, args.output