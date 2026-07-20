from pathlib import Path

import pytest

from app.database import connect_database
from app.recommendation_service import RecommendationService
from app.repositories import SqliteRecommendationRepository
from tests.recommendation_service_helpers import write_recommendation_fixture


def build_service(database_path: Path) -> RecommendationService:
    return RecommendationService(SqliteRecommendationRepository(database_path))


def test_recommendations_match_subset_antecedent_and_exclude_cart_item(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.sqlite"
    write_recommendation_fixture(database_path)

    recommendations = build_service(database_path).recommend([1], top_k=5)

    assert [item.product_id for item in recommendations] == [6, 7]
    assert recommendations[0].score == 3.0
    assert recommendations[0].product_name == "Coffee"
    assert recommendations[0].explanations[0].antecedent == [1]
    assert recommendations[0].explanations[0].matched_cart_items == [1]
    assert recommendations[0].explanations[0].support == 3
    assert recommendations[0].explanations[0].pattern_score == 3.0


def test_recommendations_sum_multiple_matching_pattern_scores(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.sqlite"
    write_recommendation_fixture(database_path)

    recommendations = build_service(database_path).recommend([1, 2], top_k=3)

    assert [item.product_id for item in recommendations] == [6, 7]
    assert recommendations[0].score == 7.0
    assert [explanation.pattern_score for explanation in recommendations[0].explanations] == [
        4.0,
        3.0,
    ]


def test_recommendations_use_deterministic_score_tie_break(tmp_path: Path) -> None:
    database_path = tmp_path / "app.sqlite"
    write_recommendation_fixture(database_path)

    recommendations = build_service(database_path).recommend([7], top_k=5)

    assert [item.product_id for item in recommendations] == [8, 9]
    assert recommendations[0].score == 0.5


def test_recommendations_fallback_to_weighted_popular_products(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.sqlite"
    write_recommendation_fixture(database_path)

    recommendations = build_service(database_path).recommend([999], top_k=3)

    assert [item.product_id for item in recommendations] == [7, 6, 2]
    assert all(item.is_fallback for item in recommendations)
    assert recommendations[0].explanations[0].raw_pattern == "fallback:weighted_popular"


def test_recommendations_fallback_excludes_real_cart_items(tmp_path: Path) -> None:
    database_path = tmp_path / "app.sqlite"
    write_recommendation_fixture(database_path)

    recommendations = build_service(database_path).recommend([7], top_k=3)

    assert 7 not in [item.product_id for item in recommendations]


def test_recommendations_empty_cart_uses_fallback(tmp_path: Path) -> None:
    database_path = tmp_path / "app.sqlite"
    write_recommendation_fixture(database_path)

    recommendations = build_service(database_path).recommend([], top_k=2)

    assert [item.product_id for item in recommendations] == [7, 6]


def test_recommendations_limit_top_k_and_clamp_oversized_top_k(tmp_path: Path) -> None:
    database_path = tmp_path / "app.sqlite"
    write_recommendation_fixture(database_path)
    service = build_service(database_path)

    assert len(service.recommend([999], top_k=1)) == 1
    assert len(service.recommend([999], top_k=100)) == 6


@pytest.mark.parametrize("cart", [["abc"], [0], [1.9], [True], "12"])
def test_recommendations_reject_malformed_cart(
    tmp_path: Path,
    cart: object,
) -> None:
    database_path = tmp_path / "app.sqlite"
    write_recommendation_fixture(database_path)
    service = build_service(database_path)

    with pytest.raises(ValueError, match="cart item IDs must be positive integers"):
        service.recommend(cart, top_k=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", ["many", 0, 1.9, True])
def test_recommendations_reject_malformed_top_k(
    tmp_path: Path,
    top_k: object,
) -> None:
    database_path = tmp_path / "app.sqlite"
    write_recommendation_fixture(database_path)
    service = build_service(database_path)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        service.recommend([1], top_k=top_k)  # type: ignore[arg-type]


def test_recommendations_ignore_pattern_candidate_without_product_row(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.sqlite"
    write_recommendation_fixture(database_path)
    with connect_database(database_path) as connection:
        connection.execute("delete from products where item_id = ?", (6,))
        connection.commit()

    recommendations = build_service(database_path).recommend([1], top_k=5)

    assert [item.product_id for item in recommendations] == [7]
