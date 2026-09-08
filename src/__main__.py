import json
from pathlib import Path
import sys
from typing import Any

from llm_sdk import Small_LLM_Model
from .base_models import FunctionDefinitionItem, Result
from .constrained_decoding import (
    build_prompt,
    generate_boolean_value,
    generate_function_name,
    generate_number_value,
    generate_string_value,
)
from .parse_validate import parse_validate_json


def pipeline(
    model: Small_LLM_Model,
    prompt: str,
    function_defs: list[FunctionDefinitionItem],
) -> Result:
    """Process a single natural language prompt using constrained decoding."""
    function_names = [f.name for f in function_defs]
    super_prompt = build_prompt(prompt, function_defs) + '{"name": "'

    input_ids = model.encode(super_prompt)[0].tolist()
    func_name = generate_function_name(model, input_ids, function_names)
    func_def = next(
        (f for f in function_defs if f.name == func_name),
        function_defs[0],
    )

    super_prompt += f'{func_name}", "parameters": {{'
    parameters: dict[str, Any] = {}
    param_items = list(func_def.parameters.items())

    for i, (param_name, param_info) in enumerate(param_items):
        super_prompt += f'"{param_name}": '
        input_ids = model.encode(super_prompt)[0].tolist()
        val: Any
        if param_info.type == "boolean":
            val = generate_boolean_value(model, input_ids)
            parameters[param_name] = val
            super_prompt += "true" if val else "false"
        elif param_info.type in ("number", "integer"):
            raw_val = generate_number_value(model, input_ids)
            try:
                if param_info.type == "number":
                    val = float(raw_val)
                else:
                    val = int(float(raw_val))
            except ValueError:
                val = 0.0 if param_info.type == "number" else 0
            parameters[param_name] = val
            super_prompt += raw_val
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


def print_result_entry(item: dict[str, Any], is_last: bool) -> None:
    """Print a single result entry with correct format."""
    comma = "" if is_last else ","
    prompt = json.dumps(item["prompt"])
    name = json.dumps(item["name"])
    params = json.dumps(item["parameters"])
    print(
        f"""    {{
        "prompt": {prompt},
        "name": {name},
        "parameters": {params}
    }}{comma}""",
    )


def main() -> None:
    """Main execution entrypoint for processing function calling prompts."""
    prompts_data, function_defs_data, output_string = parse_validate_json()

    output_path = Path(output_string)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not function_defs_data:
        print(
            "Warning: No valid function definitions loaded. Exiting.",
            file=sys.stderr,
        )
        with output_path.open(mode="w", encoding="utf-8") as output_file:
            json.dump([], output_file, indent=2)
        print("[]", flush=True)
        return

    if not prompts_data:
        print("Warning: No prompts to process. Exiting.", file=sys.stderr)
        with output_path.open(mode="w", encoding="utf-8") as output_file:
            json.dump([], output_file, indent=2)
        print("[]", flush=True)
        return

    model = Small_LLM_Model()

    results: list[dict[str, Any]] = []
    print("[", flush=True)
    for idx, prompt_data in enumerate(prompts_data):
        result = pipeline(model, prompt_data.prompt, function_defs_data)
        item = result.model_dump()
        results.append(item)
        is_last = idx == len(prompts_data) - 1
        print_result_entry(item, is_last=is_last)
    print("]", flush=True)

    with output_path.open(mode="w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)


if __name__ == "__main__":
    main()
