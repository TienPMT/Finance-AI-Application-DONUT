from pathlib import Path

import pandas as pd
import pytest

from mining.evaluate_recommender import (
    BaselineRecommender,
    compute_hit_rate_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_recommender,
)
from mining.models import CustomerSequence
from mining.run_experiments import split_sequences_last_basket
from tests.recommendation_service_helpers import write_recommendation_fixture


class TestHitRateAtK:
    def test_exact_score_single_basket_all_hit(self) -> None:
        predictions = [[1, 2, 3]]
        ground_truth = [{1, 2}]
        assert compute_hit_rate_at_k(predictions, ground_truth, k=3) == 1.0

    def test_exact_score_single_basket_miss(self) -> None:
        predictions = [[4, 5]]
        ground_truth = [{1, 2}]
        assert compute_hit_rate_at_k(predictions, ground_truth, k=2) == 0.0

    def test_exact_score_two_baskets_one_hit(self) -> None:
        predictions = [[1, 2], [3, 4]]
        ground_truth = [{1}, {5}]
        assert compute_hit_rate_at_k(predictions, ground_truth, k=2) == 0.5

    def test_hit_truncated_at_k(self) -> None:
        predictions = [[1, 2, 3]]
        ground_truth = [{3}]
        assert compute_hit_rate_at_k(predictions, ground_truth, k=1) == 0.0
        assert compute_hit_rate_at_k(predictions, ground_truth, k=3) == 1.0

    def test_no_predictions_zero_hit(self) -> None:
        predictions: list[list[int]] = [[]]
        ground_truth = [{1}]
        assert compute_hit_rate_at_k([[]], [{1}], k=5) == 0.0

    def test_no_ground_truth_zero_hit(self) -> None:
        predictions = [[1, 2]]
        ground_truth: list[set[int]] = [set()]
        assert compute_hit_rate_at_k(predictions, ground_truth, k=5) == 0.0

    def test_empty_input_zero(self) -> None:
        assert compute_hit_rate_at_k([], [], k=5) == 0.0

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_hit_rate_at_k([[1]], [{1}, {2}], k=5)

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError, match="k must be a positive integer"):
            compute_hit_rate_at_k([[1]], [{1}], k=0)

    def test_k_larger_than_predictions_uses_all(self) -> None:
        predictions = [[1, 2]]
        ground_truth = [{2}]
        assert compute_hit_rate_at_k(predictions, ground_truth, k=100) == 1.0


class TestPrecisionAtK:
    def test_exact_precision_one_basket(self) -> None:
        predictions = [[1, 2, 3, 4]]
        ground_truth = [{1, 2}]
        assert compute_precision_at_k(predictions, ground_truth, k=4) == 0.5

    def test_exact_precision_two_baskets(self) -> None:
        predictions = [[1, 2], [3, 4]]
        ground_truth = [{1}, {5}]
        assert compute_precision_at_k(predictions, ground_truth, k=2) == 0.25

    def test_precision_truncated_at_k(self) -> None:
        predictions = [[1, 2, 3]]
        ground_truth = [{2, 3, 4}]
        assert compute_precision_at_k(predictions, ground_truth, k=2) == 0.5

    def test_no_predictions_zero_precision(self) -> None:
        assert compute_precision_at_k([[]], [{1, 2}], k=5) == 0.0

    def test_no_ground_truth_zero_precision(self) -> None:
        assert compute_precision_at_k([[1, 2]], [set()], k=5) == 0.0

    def test_empty_input_zero(self) -> None:
        assert compute_precision_at_k([], [], k=5) == 0.0


class TestRecallAtK:
    def test_exact_recall_one_basket(self) -> None:
        predictions = [[1, 2]]
        ground_truth = [{1, 2, 3, 4}]
        assert compute_recall_at_k(predictions, ground_truth, k=2) == 0.5

    def test_exact_recall_two_baskets(self) -> None:
        predictions = [[1, 2], [3, 4]]
        ground_truth = [{1, 5}, {3}]
        # Basket 1: |{1} ∩ {1,5}| / |{1,5}| = 0.5
        # Basket 2: |{3} ∩ {3}| / |{3}| = 1.0
        # Average: 0.75
        assert compute_recall_at_k(predictions, ground_truth, k=2) == 0.75

    def test_recall_truncated_at_k(self) -> None:
        predictions = [[1, 2, 3, 4]]
        ground_truth = [{2, 4}]
        assert compute_recall_at_k(predictions, ground_truth, k=2) == 0.5

    def test_no_ground_truth_zero_recall(self) -> None:
        assert compute_recall_at_k([[1, 2]], [set()], k=5) == 0.0

    def test_empty_input_zero(self) -> None:
        assert compute_recall_at_k([], [], k=5) == 0.0


class TestBaselineRecommender:
    def test_recommends_top_weighted_popular_products(self, tmp_path: Path) -> None:
        database_path = tmp_path / "app.sqlite"
        write_recommendation_fixture(database_path)

        recommender = BaselineRecommender(database_path)
        recommendations = recommender.recommend([1], top_k=3)

        assert len(recommendations) == 3
        assert recommendations[0] == 7
        assert recommendations[1] == 6
        assert recommendations[2] == 2

    def test_excludes_cart_items_from_recommendations(self, tmp_path: Path) -> None:
        database_path = tmp_path / "app.sqlite"
        write_recommendation_fixture(database_path)

        recommender = BaselineRecommender(database_path)
        recommendations = recommender.recommend([7], top_k=3)

        assert 7 not in recommendations

    def test_empty_cart_returns_top_products(self, tmp_path: Path) -> None:
        database_path = tmp_path / "app.sqlite"
        write_recommendation_fixture(database_path)

        recommender = BaselineRecommender(database_path)
        recommendations = recommender.recommend([], top_k=2)

        assert recommendations == [7, 6]

    def test_top_k_limit_respected(self, tmp_path: Path) -> None:
        database_path = tmp_path / "app.sqlite"
        write_recommendation_fixture(database_path)

        recommender = BaselineRecommender(database_path)
        assert len(recommender.recommend([], top_k=2)) == 2
        assert len(recommender.recommend([], top_k=4)) == 4


class TestEvaluateRecommender:
    def test_perfect_recommender_scores_one(self) -> None:
        def perfect_recommend(
            cart: list[int], top_k: int
        ) -> list[int]:
            return [10, 20, 30][:top_k]

        test_baskets: list[tuple[list[int], set[int]]] = [
            ([1], {10, 20}),
            ([2], {10}),
        ]
        result = evaluate_recommender(perfect_recommend, test_baskets, k=3)

        assert result["hit_rate@3"] == 1.0
        assert result["precision@3"] == pytest.approx(0.5)
        assert result["recall@3"] == 1.0

    def test_worst_recommender_scores_zero(self) -> None:
        def worst_recommend(
            cart: list[int], top_k: int
        ) -> list[int]:
            return [99, 98, 97][:top_k]

        test_baskets: list[tuple[list[int], set[int]]] = [
            ([1], {10, 20}),
        ]
        result = evaluate_recommender(worst_recommend, test_baskets, k=3)

        assert result["hit_rate@3"] == 0.0
        assert result["precision@3"] == 0.0
        assert result["recall@3"] == 0.0

    def test_empty_test_baskets_returns_zero_metrics(self) -> None:
        def any_recommend(cart: list[int], top_k: int) -> list[int]:
            return [1, 2]

        result = evaluate_recommender(any_recommend, [], k=5)
        assert result["hit_rate@5"] == 0.0
        assert result["precision@5"] == 0.0
        assert result["recall@5"] == 0.0

    def test_records_basket_count(self) -> None:
        def any_recommend(cart: list[int], top_k: int) -> list[int]:
            return [1]

        test_baskets: list[tuple[list[int], set[int]]] = [
            ([1], {2}),
            ([2], {3}),
            ([3], {4}),
        ]
        result = evaluate_recommender(any_recommend, test_baskets, k=1)
        assert result["basket_count"] == 3


class TestSplitSequencesLastBasket:
    def test_splits_sequences_with_multiple_itemsets(self) -> None:
        sequences = [
            CustomerSequence("c1", itemsets=((1, 2), (3,), (4, 5))),
            CustomerSequence("c2", itemsets=((6,), (7, 8))),
        ]
        train, test = split_sequences_last_basket(sequences)

        assert len(train) == 2
        assert len(test) == 2

        assert train[0].customer_id == "c1"
        assert train[0].itemsets == ((1, 2), (3,))

        assert test[0][0] == [1, 2, 3]
        assert test[0][1] == {4, 5}

        assert train[1].customer_id == "c2"
        assert train[1].itemsets == ((6,),)
        assert test[1][0] == [6]
        assert test[1][1] == {7, 8}

    def test_skips_single_itemset_sequences(self) -> None:
        sequences = [
            CustomerSequence("c1", itemsets=((1, 2),)),
            CustomerSequence("c2", itemsets=((3,), (4,))),
        ]
        train, test = split_sequences_last_basket(sequences)

        assert len(train) == 1
        assert train[0].customer_id == "c2"
        assert test[0][0] == [3]
        assert test[0][1] == {4}

    def test_empty_input_returns_empty(self) -> None:
        train, test = split_sequences_last_basket([])
        assert train == []
        assert test == []

    def test_all_single_itemset_returns_empty(self) -> None:
        sequences = [
            CustomerSequence("c1", itemsets=((1,),)),
            CustomerSequence("c2", itemsets=((2,),)),
        ]
        train, test = split_sequences_last_basket(sequences)
        assert train == []
        assert test == []

    def test_cart_includes_all_unique_items_from_training(self) -> None:
        sequences = [
            CustomerSequence("c1", itemsets=((1, 2), (1, 3), (4,))),
        ]
        train, test = split_sequences_last_basket(sequences)
        assert test[0][0] == [1, 2, 3]
        assert test[0][1] == {4}

    def test_held_out_itemset_preserved_as_set(self) -> None:
        sequences = [
            CustomerSequence("c1", itemsets=((1,), (2, 2, 3))),
        ]
        train, test = split_sequences_last_basket(sequences)
        assert test[0][0] == [1]
        assert test[0][1] == {2, 3}
