from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, StrictInt, field_validator

from app.recommendation_service import RecommendationService
from app.repositories import SqliteRecommendationRepository
from app.web_queries import DataUnavailableError, RetailAnalyticsQueries

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

CART_ITEM_ID_MAX_COUNT = 50


class RecommendationRequest(BaseModel):
    cart_item_ids: list[StrictInt] = Field(
        ...,
        max_length=CART_ITEM_ID_MAX_COUNT,
    )
    top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("cart_item_ids")
    @classmethod
    def _cart_items_must_be_positive(cls, value: list[int]) -> list[int]:
        if any(item_id <= 0 for item_id in value):
            raise ValueError("each cart item ID must be a positive integer")
        return value


def create_web_router(database_path: Path) -> APIRouter:
    router = APIRouter()
    queries = RetailAnalyticsQueries(database_path)
    recommendations = RecommendationService(
        SqliteRecommendationRepository(database_path)
    )

    @router.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "stats": queries.dashboard_stats(),
                "patterns": queries.list_patterns(limit=5),
            },
        )

    @router.get("/products", response_class=HTMLResponse)
    def products_page(
        request: Request,
        q: str = Query(default="", max_length=100),
        min_price: str = Query(default="", max_length=20),
        max_price: str = Query(default="", max_length=20),
        min_popularity: int = Query(default=0, ge=0),
        sort: str = Query(default="popular", pattern="^(popular|price_asc|price_desc)$"),
    ) -> HTMLResponse:
        min_price_val = _parse_price(min_price)
        max_price_val = _parse_price(max_price)
        products = queries.list_products(
            query=q,
            min_price=min_price_val,
            max_price=max_price_val,
            min_popularity=min_popularity,
            sort=sort,
        )
        return templates.TemplateResponse(
            request=request,
            name="products.html",
            context={
                "products": products,
                "filters": {
                    "q": q,
                    "min_price": min_price_val,
                    "max_price": max_price_val,
                    "min_popularity": min_popularity,
                    "sort": sort,
                },
            },
        )

    @router.get("/cart", response_class=HTMLResponse)
    def cart_page(
        request: Request,
        items: str = Query(default="", max_length=500),
    ) -> HTMLResponse:
        cart_ids = parse_cart_items(items)
        products = queries.get_products(set(cart_ids))
        recommendations_ = recommendations.recommend(cart_ids, top_k=5) if cart_ids else []
        cart_total = sum(product.unit_price for product in products) if products else 0
        return templates.TemplateResponse(
            request=request,
            name="cart.html",
            context={
                "cart_products": products,
                "cart_item_ids": cart_ids,
                "recommendations": recommendations_,
                "cart_total": cart_total,
            },
        )

    @router.post("/recommend", response_class=HTMLResponse)
    def recommendation_fragment(
        request: Request,
        payload: RecommendationRequest,
    ) -> HTMLResponse:
        queries.dashboard_stats()
        return templates.TemplateResponse(
            request=request,
            name="fragments/recommendations.html",
            context={
                "recommendations": recommendations.recommend(
                    payload.cart_item_ids,
                    payload.top_k,
                )
            },
        )

    @router.get("/patterns", response_class=HTMLResponse)
    def patterns_page(
        request: Request,
        limit: int = Query(default=25, ge=1, le=100),
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="patterns.html",
            context={"patterns": queries.list_patterns(limit=limit), "limit": limit},
        )

    return router


def _parse_price(raw: str) -> float | None:
    if not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def parse_cart_items(value: str) -> list[int]:
    if not value.strip():
        return []
    item_ids: list[int] = []
    for raw_value in value.split(","):
        stripped = raw_value.strip()
        if not stripped.isdigit() or int(stripped) <= 0:
            continue
        item_id = int(stripped)
        if item_id not in item_ids:
            item_ids.append(item_id)
    return item_ids
