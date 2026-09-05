from pydantic import BaseModel
from typing import Any


class PromptItem(BaseModel):
	prompt: str


class Nested(BaseModel):
	type: str


class FunctionDefinitionItem(BaseModel):
	name: str
	description: str
	parameters: dict[str, Nested]

class Result(BaseModel):

    prompt: str
    name: str
    parameters: dict[str, Any]
