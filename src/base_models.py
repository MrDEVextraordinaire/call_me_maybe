from pydantic import BaseModel


class PromptItem(BaseModel):
	prompt: str


class Nested(BaseModel):
	type: str


class FunctionDefinitionItem(BaseModel):
	name: str
	description: str
	parameters: dict[str, Nested]

