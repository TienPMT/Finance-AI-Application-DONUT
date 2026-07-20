from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from tests.recommendation_service_helpers import write_recommendation_fixture


def build_client(database_path: Path) -> TestClient:
    return TestClient(create_app(database_path))


def test_all_retail_analytics_pages_render_from_fixture_database(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.sqlite"
    write_recommendation_fixture(database_path)
    client = build_client(database_path)

    expected_content = {
        "/": "Phân tích bán lẻ",
        "/products": "Danh mục sản phẩm",
        "/cart?items=1,2": "Giỏ hàng",
        "/patterns": "Mẫu tuần tự",
    }

    for path, content in expected_content.items():
        response = client.get(path)

        assert response.status_code == 200
        assert content in response.text


def test_catalog_filters_products_by_name_source_id_and_price(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.sqlite"
    write_recommendation_fixture(database_path)
    client = build_client(database_path)

    by_name = client.get("/products", params={"q": "coffee"})
    by_source_id = client.get("/products", params={"q": "D"})
    by_price = client.get("/products", params={"min_price": 4, "max_price": 5})
    by_popularity = client.get("/products", params={"min_popularity": 10})

    assert "Coffee" in by_name.text
    assert "Apple" not in by_name.text
    assert "Dates" in by_source_id.text
    assert "Bread" in by_price.text
    assert "Dates" in by_price.text
    assert "Coffee" not in by_price.text
    assert "Dates" in by_popularity.text
    assert "Coffee" not in by_popularity.text


def test_recommendation_fragment_renders_pattern_evidence(tmp_path: Path) -> None:
    database_path = tmp_path / "app.sqlite"
    write_recommendation_fixture(database_path)
    client = build_client(database_path)

    response = client.post("/recommend", json={"cart_item_ids": [1], "top_k": 3})

    assert response.status_code == 200
    assert "Coffee" in response.text
    assert "Support: 3" in response.text
    assert "Điểm đóng góp: 3.000" in response.text
    assert "Mẫu gốc" in response.text


def test_cart_page_renders_selected_products_and_recommendations(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "app.sqlite"
    write_recommendation_fixture(database_path)
    client = build_client(database_path)

    response = client.get("/cart", params={"items": "1,2"})

    assert response.status_code == 200
    assert "Apple" in response.text
    assert "Bread" in response.text
    assert "Coffee" in response.text
    assert "Tổng giá tham khảo" in response.text


def test_missing_database_returns_friendly_service_message(tmp_path: Path) -> None:
    client = build_client(tmp_path / "missing.sqlite")

    response = client.get("/")
    recommendation = client.post(
        "/recommend",
        json={"cart_item_ids": [1], "top_k": 3},
    )
    cart_with_items = client.get("/cart", params={"items": "1,2"})

    assert response.status_code == 503
    assert "Chưa có dữ liệu khai phá" in response.text
    assert "Traceback" not in response.text
    assert recommendation.status_code == 503
    assert "Chưa có dữ liệu khai phá" in recommendation.text
    assert cart_with_items.status_code == 503
    assert "Chưa có dữ liệu khai phá" in cart_with_items.text


def test_recommendation_rejects_invalid_or_oversized_cart(tmp_path: Path) -> None:
    database_path = tmp_path / "app.sqlite"
    write_recommendation_fixture(database_path)
    client = build_client(database_path)

    negative = client.post(
        "/recommend",
        json={"cart_item_ids": [-1], "top_k": 3},
    )
    boolean = client.post(
        "/recommend",
        json={"cart_item_ids": [True], "top_k": 3},
    )
    oversized = client.post(
        "/recommend",
        json={"cart_item_ids": list(range(1, 52)), "top_k": 3},
    )

    assert negative.status_code == 422
    assert boolean.status_code == 422
    assert oversized.status_code == 422


def test_static_assets_are_mounted(tmp_path: Path) -> None:
    database_path = tmp_path / "app.sqlite"
    write_recommendation_fixture(database_path)
    client = build_client(database_path)

    stylesheet = client.get("/static/styles.css")
    cart_script = client.get("/static/cart.js")

    assert stylesheet.status_code == 200
    assert "--color-primary" in stylesheet.text
    assert cart_script.status_code == 200
    assert "localStorage" in cart_script.text
