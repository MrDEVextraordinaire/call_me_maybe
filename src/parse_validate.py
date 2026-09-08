import argparse
import json
from typing import Any
from .base_models import FunctionDefinitionItem, PromptItem


def load_json(file_path_string: str) -> list[dict[Any, Any]]:
    """Load and parse a JSON file expected to contain a list of objects.

    Args:
        file_path_string: Path to the JSON file.

    Returns:
        A list of parsed JSON objects, or an empty list on failure.
    """
    try:
        with open(file_path_string, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data).__name__}")

        return data

    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading {file_path_string}: {e}")
        return []


def parse_validate_json() -> tuple[
    list[PromptItem], list[FunctionDefinitionItem], str
]:
    """Parse command-line arguments and validate input files with Pydantic.

    Returns:
        A tuple containing:
            - list of validated PromptItem objects
            - list of validated FunctionDefinitionItem objects
            - output file path string
    """
    parser = argparse.ArgumentParser(
        description="Call Me Maybe: LLM Function Calling"
    )
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
        default="data/output/function_calling_results.json",
    )
    args = parser.parse_args()

    prompts = load_json(args.input)
    definitions = load_json(args.functions_definition)

    function_def_parsed: list[FunctionDefinitionItem] = []
    for definition in definitions:
        try:
            function_def_parsed.append(FunctionDefinitionItem(**definition))
        except Exception as e:
            print(f"Warning: Skipping invalid function definition: {e}")

    prompts_parsed: list[PromptItem] = []
    for prompt in prompts:
        try:
            prompts_parsed.append(PromptItem(**prompt))
        except Exception as e:
            print(f"Warning: Skipping invalid prompt entry: {e}")

    return prompts_parsed, function_def_parsed, args.output
