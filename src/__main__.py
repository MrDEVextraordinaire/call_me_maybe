import argparse

def main():
    parser = argparse.ArgumentParser(description="Function calling tool" \
    " that translates natural language" \
    " prompts into structured function calls")
    parser.add_argument("--functions_definition", type=str, required=True, 
                        help="Path to the JSON file containing " \
                        "function definitions")
    parser.add_argument("--input", type=str, required=True, help="English " \
    "written prompt to be translated into machine-executable output")
    parser.add_argument("--output", type=str, help="Path to the" \
    " JSON structured output file")
    args = parser.parse_args()

    print("args:", args)

if __name__ == "__main__":
    main()


"""uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json"""
