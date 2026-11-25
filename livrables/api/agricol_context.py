from typing import Optional

from pydantic import BaseModel


class RecommendContextModel(BaseModel):
    area: str
    average_rain_fall_mm_per_year: Optional[float] = None
    pesticides_tonnes: Optional[float] = None
    pesticides_kg_ha: Optional[float] = None
    avg_temp: Optional[float] = None


class AgricolContextModel(RecommendContextModel):
    item: str
