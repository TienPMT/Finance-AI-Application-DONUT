import pandas as pd
import pytest

from mining.parse_patterns import parse_patterns_text
from mining.score_patterns import (
    ProductCatalogError,
    build_product_weights,
    score_patterns,
)


def products_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": 1,
                "product_source_id": "A",
                "product_name": "Apple",
                "unit_price": 2.0,
                "transaction_count": 5,
                "country": "United Kingdom",
            },
            {
                "item_id": 6,
                "product_source_id": "B",
                "product_name": "Bread",
                "unit_price": 10.0,
                "transaction_count": 3,
                "country": "United Kingdom",
            },
            {
                "item_id": 7,
                "product_source_id": "C",
                "product_name": "Coffee",
                "unit_price": 5.0,
                "transaction_count": 2,
                "country": "France",
            },
        ]
    )


def test_weight_normalization_uses_max_product_price() -> None:
    weights = build_product_weights(products_fixture())

    assert weights == {1: 0.2, 6: 1.0, 7: 0.5}


def test_scoring_uses_support_and_average_consequent_weight() -> None:
    patterns = parse_patterns_text("1 -1 6 7 -1 #SUP: 3\n6 7 -1 #SUP: 2\n")

    scored = score_patterns(patterns, products_fixture(), sequence_count=6)

    assert len(scored) == 1
    assert scored[0].antecedent == (1,)
    assert scored[0].consequent == (6, 7)
    assert scored[0].support == 3
    assert scored[0].support_ratio == 0.5
    assert scored[0].weight_value == 0.75
    assert scored[0].pattern_score == 2.25


def test_scoring_fails_when_pattern_references_unknown_item() -> None:
    patterns = parse_patterns_text("1 -1 99 -1 #SUP: 3\n")

    with pytest.raises(ProductCatalogError, match="Unknown item IDs"):
        score_patterns(patterns, products_fixture(), sequence_count=6)


def test_scoring_rejects_support_above_sequence_count() -> None:
    patterns = parse_patterns_text("1 -1 6 -1 #SUP: 3\n")

    with pytest.raises(ValueError, match="support cannot exceed sequence_count"):
        score_patterns(patterns, products_fixture(), sequence_count=2)
