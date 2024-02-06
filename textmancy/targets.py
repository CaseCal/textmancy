from typing import List
from langchain.pydantic_v1 import BaseModel, Field


class Theme(BaseModel):
    """ "A recurring idea or concept in a work of literature"""

    name: str = Field(
        ..., description="The name of the theme. Ex: Death and Mortality, Regret, etc."
    )
    reasoning: str = Field(
        ...,
        description="The reasoning behind the theme occuring in a specific peice of text",
    )


class Character(BaseModel):
    """A person or being in a work of literature"""

    name: str = Field(..., description="The name of the character")
    description: str = Field(
        ..., description="A description of the character and their role in the story"
    )
    physical_description: str = Field(
        ..., description="A description of the character's physical appearance"
    )
    known_as: List[str] = Field(
        ..., description="A list of names the character is known as in the story"
    )
    summary_of_actions: str = Field(
        ..., description="A summary of the actions the character takes in the story"
    )
