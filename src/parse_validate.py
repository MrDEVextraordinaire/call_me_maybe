import argparse
from .base_models import PromptItem, FunctionDefinitionItem
import json
from typing import Any


def load_json(file_path_string: str) -> list[dict[Any, Any]]:
    try:
        with open(file_path_string, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data).__name__}")

        return data
        
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading {file_path_string}: {e}")
        return []


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