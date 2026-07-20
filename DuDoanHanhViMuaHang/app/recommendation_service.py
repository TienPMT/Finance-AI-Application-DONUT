from collections import defaultdict
from dataclasses import dataclass

from app.repositories import PatternRecord, ProductRecord, SqliteRecommendationRepository
from app.schemas import ProductRecommendation, RecommendationExplanation

MAX_TOP_K = 50


@dataclass(frozen=True)
class PatternContribution:
    pattern: PatternRecord
    contribution: float
    matched_cart_items: tuple[int, ...]


class RecommendationService:
    def __init__(self, repository: SqliteRecommendationRepository) -> None:
        self.repository = repository

    def recommend(
        self,
        cart_item_ids: list[int] | tuple[int, ...],
        top_k: int = 5,
    ) -> list[ProductRecommendation]:
        limit = self._normalize_top_k(top_k)
        cart = self._normalize_cart(cart_item_ids)
        recommendations = self._recommend_from_patterns(cart, limit)
        if recommendations:
            return recommendations
        return self._fallback_recommendations(cart, limit)

    def _recommend_from_patterns(
        self,
        cart: set[int],
        limit: int,
    ) -> list[ProductRecommendation]:
        if not cart:
            return []
        candidate_scores: dict[int, float] = defaultdict(float)
        candidate_contributions: dict[int, list[PatternContribution]] = defaultdict(list)
        for pattern in self.repository.list_patterns():
            match = self._match_pattern(pattern, cart)
            if match is None:
                continue
            match_ratio, matched_items = match
            contribution = pattern.pattern_score * match_ratio
            for item_id in pattern.consequent:
                if item_id in cart:
                    continue
                candidate_scores[item_id] += contribution
                candidate_contributions[item_id].append(
                    PatternContribution(pattern, contribution, matched_items)
                )
        if not candidate_scores:
            return []
        product_ids = set(candidate_scores)
        products = self.repository.get_products_by_ids(product_ids)
        product_ids_with_rows = set(products)
        ranked_item_ids = sorted(
            product_ids_with_rows,
            key=lambda item_id: (-candidate_scores[item_id], item_id),
        )[:limit]
        return [
            self._build_recommendation(
                product=products[item_id],
                score=candidate_scores[item_id],
                contributions=candidate_contributions[item_id],
                is_fallback=False,
            )
            for item_id in ranked_item_ids
        ]

    def _fallback_recommendations(
        self,
        cart: set[int],
        limit: int,
    ) -> list[ProductRecommendation]:
        products = self.repository.list_fallback_products(cart, limit)
        return [
            self._build_fallback_recommendation(product)
            for product in products
        ]

    def _match_pattern(
        self,
        pattern: PatternRecord,
        cart: set[int],
    ) -> tuple[float, tuple[int, ...]] | None:
        antecedent = set(pattern.antecedent)
        if not antecedent or not antecedent.issubset(cart):
            return None
        matched_items = tuple(sorted(antecedent & cart))
        return len(matched_items) / len(antecedent), matched_items

    def _build_recommendation(
        self,
        product: ProductRecord,
        score: float,
        contributions: list[PatternContribution],
        is_fallback: bool,
    ) -> ProductRecommendation:
        ranked_contributions = sorted(
            contributions,
            key=lambda item: (-item.contribution, item.pattern.id),
        )
        return ProductRecommendation(
            product_id=product.item_id,
            product_source_id=product.product_source_id,
            product_name=product.product_name,
            unit_price=product.unit_price,
            normalized_weight=product.normalized_weight,
            transaction_count=product.transaction_count,
            score=round(score, 6),
            is_fallback=is_fallback,
            explanations=[
                self._build_explanation(contribution)
                for contribution in ranked_contributions
            ],
        )

    def _build_fallback_recommendation(
        self,
        product: ProductRecord,
    ) -> ProductRecommendation:
        return ProductRecommendation(
            product_id=product.item_id,
            product_source_id=product.product_source_id,
            product_name=product.product_name,
            unit_price=product.unit_price,
            normalized_weight=product.normalized_weight,
            transaction_count=product.transaction_count,
            score=round(product.fallback_score, 6),
            is_fallback=True,
            explanations=[
                RecommendationExplanation(
                    pattern_id=0,
                    antecedent=[],
                    consequent=[product.item_id],
                    support=0,
                    support_ratio=0.0,
                    pattern_score=round(product.fallback_score, 6),
                    matched_cart_items=[],
                    raw_pattern="fallback:weighted_popular",
                )
            ],
        )

    def _build_explanation(
        self,
        contribution: PatternContribution,
    ) -> RecommendationExplanation:
        pattern = contribution.pattern
        return RecommendationExplanation(
            pattern_id=pattern.id,
            antecedent=list(pattern.antecedent),
            consequent=list(pattern.consequent),
            support=pattern.support,
            support_ratio=pattern.support_ratio,
            pattern_score=round(contribution.contribution, 6),
            matched_cart_items=list(contribution.matched_cart_items),
            raw_pattern=pattern.raw_pattern,
        )

    def _normalize_cart(self, cart_item_ids: list[int] | tuple[int, ...]) -> set[int]:
        if not isinstance(cart_item_ids, (list, tuple)):
            raise ValueError("cart item IDs must be positive integers")
        normalized = set()
        for item_id in cart_item_ids:
            if type(item_id) is not int:
                raise ValueError("cart item IDs must be positive integers")
            normalized.add(item_id)
        if any(item_id <= 0 for item_id in normalized):
            raise ValueError("cart item IDs must be positive integers")
        return normalized

    def _normalize_top_k(self, top_k: int) -> int:
        if type(top_k) is not int:
            raise ValueError("top_k must be a positive integer")
        normalized = top_k
        if normalized <= 0:
            raise ValueError("top_k must be a positive integer")
        return min(normalized, MAX_TOP_K)
