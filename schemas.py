"""
Database Schemas for HLife

Each Pydantic model represents a collection in your MongoDB database.
The collection name is the lowercase of the class name.
"""

from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict
from datetime import datetime


class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Email address")
    password_hash: str = Field(..., description="Password hash (server-side)")
    is_premium: bool = Field(False, description="Premium subscriber")


class Profile(BaseModel):
    user_id: str = Field(..., description="Reference to user _id as string")
    allergies: List[str] = Field(default_factory=list, description="List of allergies")
    conditions: List[str] = Field(default_factory=list, description="Health conditions like diabetes, hypertension")
    preferences: Dict[str, List[str]] = Field(
        default_factory=lambda: {"liked": [], "disliked": [], "cuisines": []},
        description="Food preferences"
    )
    calorie_target: Optional[int] = Field(None, description="Daily calorie target")
    protein_target: Optional[int] = Field(None, description="Daily protein target (g)")


class Food(BaseModel):
    name: str
    description: Optional[str] = None
    nutrients: Dict[str, float] = Field(
        default_factory=lambda: {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    )
    ingredients: List[str] = Field(default_factory=list)
    recipe: List[str] = Field(default_factory=list, description="Step-by-step instructions")
    tags: List[str] = Field(default_factory=list, description="Keywords: breakfast, vegan, low-carb")
    suitable_for: List[str] = Field(default_factory=list, description="Conditions this food is good for")


class Consumption(BaseModel):
    user_id: str
    food_id: Optional[str] = None
    custom_name: Optional[str] = None
    portion_size: float = Field(1.0, description="Portion multiplier relative to base food serving")
    nutrients: Dict[str, float] = Field(
        default_factory=lambda: {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    )
    consumed_at: datetime = Field(default_factory=datetime.utcnow)


class Weight(BaseModel):
    user_id: str
    value: float = Field(..., gt=0)
    recorded_at: datetime = Field(default_factory=datetime.utcnow)


class Badge(BaseModel):
    user_id: str
    key: str
    label: str
    awarded_at: datetime = Field(default_factory=datetime.utcnow)


class Planner(BaseModel):
    user_id: str
    week_of: datetime  # Monday of the week
    plan: Dict[str, Dict[str, Optional[str]]] = Field(
        default_factory=dict, description="day -> {breakfast,lunch,dinner} -> food_id"
    )


class GroceryList(BaseModel):
    user_id: str
    name: str
    items: List[str] = Field(default_factory=list)
    generated_from: Optional[List[str]] = None  # list of food_ids
