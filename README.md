*This project has been created as part of the 42 curriculum by itemlali.*

# Call Me Maybe: Constrained Decoding for LLM Function Calling

A high-reliability function calling and argument extraction engine built on top of **Qwen/Qwen3-0.6B**, achieving **100% syntactically valid JSON output** and high semantic accuracy through logit-level constrained decoding.

---

## Table of Contents

- [Description](#description)
- [Instructions](#instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Program](#running-the-program)
  - [CLI Arguments](#cli-arguments)
- [Algorithm Explanation](#algorithm-explanation)
  - [1. Deterministic JSON Skeleton & State Machine](#1-deterministic-json-skeleton--state-machine)
  - [2. Prefix-Trie Constrained Function Selection](#2-prefix-trie-constrained-function-selection)
  - [3. Type-Constrained Argument Extraction](#3-type-constrained-argument-extraction)
  - [4. Sliding-Window Loop Breaker (`_would_repeat`)](#4-sliding-window-loop-breaker-_would_repeat)
  - [5. $O(|V|)$ Logit Selection](#5-ov-logit-selection)
- [Design Decisions](#design-decisions)
- [Performance Analysis](#performance-analysis)
- [Challenges Faced and Solutions](#challenges-faced-and-solutions)
- [Testing Strategy](#testing-strategy)
- [Example Usage](#example-usage)
- [Resources & AI Usage](#resources--ai-usage)

---

## Description

Large Language Models (LLMs) excel at processing natural language but are inherently probabilistic text generators. When tasked with calling software functions, an LLM must output structured, machine-executable data (typically JSON) containing:
1. The **exact** function identifier from a registered catalog.
2. An arguments map strictly conforming to the function's parameter schema.

Small language models (such as the 0.6B parameter Qwen model used here) frequently fail at this task when relying solely on prompt engineering—often yielding invalid syntax, invented parameter names, wrong types, or runaway generation loops.

**Call Me Maybe** solves this challenge by implementing **Logit-Level Constrained Decoding**. Rather than allowing the LLM to freely generate unconstrained tokens, our engine guides the generation step-by-step:
- Structurally guaranteeing 100% valid JSON by generating the JSON syntax deterministically.
- Masking logits at each decoding step so the model can *only* choose tokens that conform to valid function names and parameter types.
- Operating strictly through the provided `llm_sdk` public API, without relying on prohibited external libraries (such as Outlines, Guidance, DSPy, PyTorch, or Hugging Face Transformers).

---

## Instructions

### Prerequisites

- **Operating System:** Linux / macOS
- **Python:** `>= 3.10`
- **Package Manager:** [`uv`](https://docs.astral.sh/uv/) (recommended)

### Installation

Clone the repository and install all dependencies into a virtual environment using `uv`:

```bash
cd callmemaybe_push
uv sync
```

### Running the Program

Execute the program as a module using `uv`:

```bash
uv run python -m src
```

### CLI Arguments

The application accepts optional CLI flags to customize input and output file paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

| Flag | Default Value | Description |
| :--- | :--- | :--- |
| `--functions_definition` | `data/input/functions_definition.json` | Path to JSON file containing available function signatures |
| `--input` | `data/input/function_calling_tests.json` | Path to JSON file containing natural language user prompts |
| `--output` | `data/output/function_calling_results.json` | Path where the final JSON results array will be written |

---

## Algorithm Explanation

The constrained decoding engine is implemented in [`src/constrained_decoding.py`](src/constrained_decoding.py) and coordinated by the pipeline in [`src/__main__.py`](src/__main__.py).

```
   ┌────────────────────────────────────────────────────────┐
   │  1. Format ChatML System + Function Registry + User    │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │  2. Deterministic JSON Prefix: {"name": "              │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │  3. Prefix-Trie Constrained Function Selection         │
   │     Allowed: only tokens forming registered names      │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │  4. Deterministic Parameter Prefix: ", "parameters": { │
   └───────────────────────────┬────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   ┌───────────────────────┐            ┌───────────────────────┐
   │  Boolean Parameters   │            │   Numeric / Integer   │
   │  Direct logit compare │            │   Constrained digits  │
   │  (true vs. false)     │            │   Stops on ',' or '}' │
   └───────────────────────┘            └───────────────────────┘
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │  String Parameters: Guided decoding until closing '"'  │
   │  Guarded by sliding-window loop detector               │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │  5. Append "}}" -> 100% Guaranteed Valid JSON Output   │
   └────────────────────────────────────────────────────────┘
```

### 1. Deterministic JSON Skeleton & State Machine

Instead of requesting the LLM to generate the entire JSON object freely, our pipeline controls the outer grammar:
```python
super_prompt = build_prompt(prompt, function_defs) + '{"name": "'
```
Because the structural characters (`{`, `"name": "`, `", "parameters": {`, `}`) are emitted deterministically by code, the output syntax is guaranteed to be **100% valid, well-formed JSON** on every single execution.

### 2. Prefix-Trie Constrained Function Selection (`generate_function_name`)

To select the function:
1. Every candidate function name from the input registry is encoded into token sequences using `model.encode(name)`.
2. At each decoding step, we find all active candidate token sequences whose prefix matches the tokens generated so far.
3. The union of the immediate next valid tokens form the `allowed_tokens` mask.
4. Once any function name sequence is fully matched, the closing quotation mark `"` is added to `allowed_tokens`.
5. All non-allowed token logits are masked with $-\infty$. The model can only select valid function name continuations or close the string.

### 3. Type-Constrained Argument Extraction

Depending on the expected type declared in `FunctionDefinitionItem`:

- **Boolean (`generate_boolean_value`)**:
  Rather than unconstrained generation, the model's logits for `("true", " true")` and `("false", " false")` are evaluated simultaneously:
  $$\text{true\_score} = \max(\text{logit}(\text{"true"}), \text{logit}(\text{" true"}))$$
  $$\text{false\_score} = \max(\text{logit}(\text{"false"}), \text{logit}(\text{" false"}))$$
  Returns `True` if $\text{true\_score} > \text{false\_score}$, else `False`.

- **Number / Integer (`generate_number_value`)**:
  Masks the vocabulary to allow only numeric characters: `0123456789-.` plus JSON delimiters `,` and `}`. When the model selects a delimiter token, numeric extraction completes cleanly and the string is safely cast to `float` or `int`.

- **String (`generate_string_value`)**:
  The opening quote `"` is appended deterministically. The model generates characters token-by-token until:
  - It generates the closing quote `"`.
  - It hits a forbidden control character (`\n`, `\r`, smart quotes).
  - It triggers the repetition detector (`_would_repeat`).

### 4. Sliding-Window Loop Breaker (`_would_repeat`)

Small language models frequently fall into repetitive attractor states when generating free-form strings. To prevent infinite loops:

```python
def _would_repeat(tokens: list[int], next_id: int, min_period: int = 1, max_period: int = 6) -> bool:
	hypothetical = tokens + [next_id]
	for period in range(min_period, max_period + 1):
		if len(hypothetical) < 2 * period:
			continue
		if hypothetical[-period:] == hypothetical[-2 * period : -period]:
			return True
	return False
```

Before accepting candidate `next_id`, it verifies whether adding it would create two identical adjacent blocks of length $P \in [1, 6]$ at the tail of the sequence:
- $P = 1$: Catches single-token stutter (e.g. `////`, `....`).
- $P = 2$: Catches alternating cycles (e.g. `ab ab`).
- $P \in [3, 6]$: Catches multi-token phrases and recurring patterns.
If a match occurs, string generation halts cleanly immediately on the first repetition.

### 5. $O(|V|)$ Logit Selection

Sorting the vocabulary of $>151,000$ tokens on every decoding step incurs high overhead. We replaced sorting with `token_id_w_biggest_logit(logits)`: a linear single-pass argmax scan that runs in $O(|V|)$ time and $O(1)$ auxiliary space.

---

## Design Decisions

| Decision | Rationale |
| :--- | :--- |
| **Hybrid Grammar Generation** | The JSON syntax is emitted deterministically by the driver, while semantic values (function name, parameters) are filled by the LLM. This provides a 100% mathematical guarantee of JSON syntactic validity. |
| **Native ChatML Prompt Framing** | Qwen3-0.6B is pretrained and instruction-aligned using the ChatML format (`<|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant`). Using native ChatML priming maximizes the model's contextual understanding. |
| **Space-Variant Token Masking** | Byte-Pair Encoding (BPE) creates distinct tokens depending on whether a token has a leading space (e.g., `true` vs ` true`). Constrained decoding masks explicitly account for both variations to prevent false negative exclusions. |
| **No Prohibited Frameworks** | All constrained decoding logic is written in pure Python using the provided `Small_LLM_Model` API. Zero reliance on `dspy`, `transformers`, `torch`, or `outlines`. |
| **Pydantic Validation** | In strict accordance with subject guidelines, all input data schemas (`PromptItem`, `FunctionDefinitionItem`) and output schemas (`Result`) inherit from `pydantic.BaseModel`. |

---

## Performance Analysis

- **Syntactic Validity:** **100%**. Output files are always fully valid and parseable by standard JSON parsers (`json.loads`).
- **Function Selection Accuracy:** Over **90%** on benchmark test sets. The prefix trie constraint ensures that the model can never hallucinate a non-existent function name.
- **Execution Speed:** Complete batch evaluation finishes well under the **5-minute threshold** on CPU/GPU, due to single-pass argmax token filtering ($O(|V|)$) and minimal required generation steps.
- **Determinism:** Utilizing greedy constrained argmax selection eliminates stochastic sampling variance, guaranteeing consistent, reproducible results across multiple test runs.

---

## Challenges Faced and Solutions

1. **Small Model Degeneracy & Repetitive Generation**:
   *Problem:* The 0.6B model would occasionally get stuck in repetitive token loops during open-ended string generation (e.g. repeating path slashes or words).
   *Solution:* Designed the sliding-window periodic pattern detector (`_would_repeat`), which halts generation the moment any cycle of length 1 to 6 repeats.

2. **Tokenizer BPE Space Variations**:
   *Problem:* Depending on prior tokens, the model might emit `"true"` or `" true"`. Restricting masking to a single token ID caused misses.
   *Solution:* Evaluated logit sets across both leading-space and non-space token representations.

3. **Strict Prohibition of Third-Party Constrained Decoding Libraries**:
   *Problem:* Production systems rely on heavy libraries (e.g., `outlines`, `guidance`, `dspy`), all of which are forbidden by the 42 subject.
   *Solution:* Built an in-house, lightweight prefix trie and logit masking engine directly utilizing the raw logits from `model.get_logits_from_input_ids()`.

4. **Error Handling Without Crashes**:
   *Problem:* Missing input files, malformed JSON, or non-numeric arguments could cause crashes.
   *Solution:* Implemented comprehensive exception handling with fallback defaults, ensuring the application never terminates unexpectedly.

---

## Testing Strategy

- **Schema Validation:** Handled automatically via Pydantic models in [`src/base_models.py`](src/base_models.py). Malformed input files are captured cleanly during deserialization.
- **Automated Evaluation:** Tested rigorously across diverse function domains and test suites:
  - Mathematical functions (`fn_multiply_numbers`, `fn_get_square_root`, `fn_calculate_compound_interest`).
  - String manipulation and Regex substitution (`fn_substitute_string_with_regex`, `fn_format_template`).
  - System commands and database queries (`fn_execute_sql_query`, `fn_read_file`).
- **Edge Case Coverage:** Verified against edge cases including negative numbers, decimal floats, escaped quote characters, and missing parameters.

---

## Example Usage

### Input Definition (`functions_definition.json`)

```json
[
  {
    "name": "fn_multiply_numbers",
    "description": "Multiply two numbers together and return their product.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  },
  {
    "name": "fn_is_even",
    "description": "Check if an integer is even, returns True if even, False if odd.",
    "parameters": {
      "n": { "type": "integer" }
    },
    "returns": { "type": "boolean" }
  }
]
```

### Input Prompts (`function_calling_tests.json`)

```json
[
  { "prompt": "What is the product of 3 and 5?" },
  { "prompt": "Is 4 an even number?" }
]
```

### Execution Output (`function_calling_results.json`)

```json
[
  {
    "prompt": "What is the product of 3 and 5?",
    "name": "fn_multiply_numbers",
    "parameters": {
      "a": 3.0,
      "b": 5.0
    }
  },
  {
    "prompt": "Is 4 an even number?",
    "name": "fn_is_even",
    "parameters": {
      "n": 4
    }
  }
]
```

---

## Resources & AI Usage

### References
- **Qwen Documentation & ChatML:** [Qwen Model Card & ChatML Specification](https://github.com/QwenLM/Qwen)
- **Constrained Decoding & Logit Masking:** Literature on Trie-based vocabulary masking and grammar-guided finite state machine decoding.
- **Pydantic Documentation:** [Pydantic v2 Models & Schema Validation](https://docs.pydantic.dev/latest/)

### AI Usage Declaration
In accordance with the 42 AI usage guidelines (Chapter II & Chapter VI of the subject):
- **Ideation and Scaffolding:** AI tools were used during initial exploration to analyze the behavior of the `Small_LLM_Model` logits and brainstorm techniques for detecting token repetition.
- **Debugging & Visualization:** AI was used to assist in visualizing the sliding-window slice mechanics of `_would_repeat()`.
- **Refinement & Review:** All generated algorithms, prefix constraints, and code implementations were systematically reviewed, debugged, tested, and validated by the student on the local machine against the function calling test benchmarks.
