from pydantic import BaseModel, Field


class RecommendationExplanation(BaseModel):
    pattern_id: int
    antecedent: list[int]
    consequent: list[int]
    support: int
    support_ratio: float
    pattern_score: float
    matched_cart_items: list[int]
    raw_pattern: str


class ProductRecommendation(BaseModel):
    product_id: int
    product_source_id: str
    product_name: str
    unit_price: float
    normalized_weight: float
    transaction_count: int
    score: float = Field(ge=0)
    is_fallback: bool
    explanations: list[RecommendationExplanation]
