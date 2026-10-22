"""API input models and the structured outputs of the two agents."""
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, model_validator

MAX_TRIP_DAYS = 14


# ---------- API input ----------

class TravelRequest(BaseModel):
    destination: str = Field(min_length=2, max_length=100, examples=["Lisbon, Portugal"])
    start_date: date
    end_date: date
    budget_min: float = Field(ge=0, description="Lower bound of the total budget for the whole group")
    budget_max: float = Field(
        gt=0,
        description="Upper bound. Covers lodging, food, activities and local transport, "
        "not travel to and from the destination.",
    )
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")
    travelers: int = Field(ge=1, le=20)
    interests: list[str] = Field(min_length=1, max_length=10, examples=[["food", "history", "viewpoints"]])
    notes: str = Field(default="", max_length=500, description="Pace, diet, mobility or other preferences")

    @model_validator(mode="after")
    def _check(self, info: ValidationInfo) -> "TravelRequest":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        if self.days > MAX_TRIP_DAYS:
            raise ValueError(f"trips are limited to {MAX_TRIP_DAYS} days")
        if self.budget_min > self.budget_max:
            raise ValueError("budget_min must not exceed budget_max")
        # Stored requests are re-loaded later, possibly after the start date has passed.
        if not (info.context or {}).get("stored") and self.start_date < date.today():
            raise ValueError("start_date is in the past")
        self.interests = list(dict.fromkeys(i.strip().lower() for i in self.interests if i.strip()))
        if not self.interests:
            raise ValueError("interests must contain at least one non-empty item")
        return self

    @property
    def days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def nights(self) -> int:
        return self.days - 1


def load_request(data: dict) -> TravelRequest:
    """Re-load a request that was already validated when the plan was created."""
    return TravelRequest.model_validate(data, context={"stored": True})


class Modification(BaseModel):
    day: int | None = Field(
        default=None, ge=1, description="Day to change. Omit for plan-wide changes such as the hotel."
    )
    change: str = Field(min_length=3, max_length=500, examples=["Swap the museum for a beach afternoon"])


class ReviewRequest(BaseModel):
    action: Literal["approve", "reject", "modify"]
    feedback: str = Field(default="", max_length=2000, description="Required for reject")
    revise: Literal["research", "itinerary"] | None = Field(
        default=None,
        description="Reject only: which agent should redo its work. Omitted = decided by an LLM router.",
    )
    modifications: list[Modification] = Field(default_factory=list, max_length=10, description="Required for modify")

    @model_validator(mode="after")
    def _check(self) -> "ReviewRequest":
        if self.action == "reject" and not self.feedback.strip():
            raise ValueError("reject requires feedback")
        if self.action == "modify" and not self.modifications:
            raise ValueError("modify requires at least one modification")
        return self


# ---------- Research agent output ----------

class Attraction(BaseModel):
    name: str = Field(description="Official name, as it would appear on a map")
    area: str = Field(description="Neighbourhood or district")
    why: str = Field(description="One line on why it suits these travellers")
    duration_hours: float = Field(description="Typical visit length")
    cost_per_person: float = Field(description="Entry price per person in the trip currency, 0 if free")
    opens: str = Field(default="", description="Opening time HH:MM (24h), empty if unknown")
    closes: str = Field(default="", description="Closing time HH:MM (24h), empty if unknown")


class TypicalCosts(BaseModel):
    hotel_per_night: float = Field(description="Mid-range room for the whole group, trip currency")
    meal_per_person: float = Field(description="Typical sit-down meal, trip currency")
    local_transport_per_person_per_day: float = Field(description="Trip currency")


class ResearchReport(BaseModel):
    summary: str
    attractions: list[Attraction] = Field(description="8-15 places, most relevant first")
    neighbourhoods_to_stay: list[str]
    local_tips: list[str]
    safety: list[str]
    weather: str = Field(description="Summary of the weather_outlook result")
    seasonal_notes: list[str] = Field(description="Events, holidays, closures or crowds on the trip dates")
    typical_costs: TypicalCosts
    sources: list[str] = Field(description="URLs actually used")


# ---------- Planner agent output ----------

class Activity(BaseModel):
    start: str = Field(description="HH:MM")
    end: str = Field(description="HH:MM")
    name: str
    description: str
    area: str
    cost_per_person: float = Field(description="Trip currency, 0 if free")
    travel_minutes_from_previous: int = 0


class Meal(BaseModel):
    meal: Literal["breakfast", "lunch", "dinner"]
    name: str = Field(description="A real restaurant from find_restaurants when available")
    cuisine: str
    near: str
    cost_per_person: float = Field(description="Trip currency")


class DayPlan(BaseModel):
    day: int
    date: date
    theme: str
    activities: list[Activity]
    meals: list[Meal]
    notes: str = ""


class Lodging(BaseModel):
    name: str = Field(description="Hotel name or type of stay")
    area: str
    cost_per_night: float = Field(description="Whole group, trip currency")
    why: str


class Itinerary(BaseModel):
    title: str
    summary: str
    lodging: Lodging
    days: list[DayPlan]
    tips: list[str] = Field(description="Packing, weather, transport and safety tips")
