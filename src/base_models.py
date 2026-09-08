from pydantic import BaseModel
from typing import Any


class PromptItem(BaseModel):
    """Schema for a single natural language input prompt."""

    prompt: str


class Nested(BaseModel):
    """Schema for parameter type specification in function definitions."""

    type: str


class FunctionDefinitionItem(BaseModel):
    """Schema for a registered function specification."""

    name: str
    description: str
    parameters: dict[str, Nested]


class Result(BaseModel):
    """Schema for a single structured function call output entry."""

    prompt: str
    name: str
    parameters: dict[str, Any]
