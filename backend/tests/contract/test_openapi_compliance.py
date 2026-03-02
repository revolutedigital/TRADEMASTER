"""OpenAPI compliance: ensure all API endpoints are documented."""

import pytest
from app.main import create_app


def test_openapi_schema_exists():
    app = create_app()
    schema = app.openapi()
    assert "paths" in schema
    assert "info" in schema
    assert schema["info"]["title"] == "TradeMaster"


def test_all_api_endpoints_in_schema():
    app = create_app()
    schema = app.openapi()
    paths = set(schema.get("paths", {}).keys())

    for route in app.routes:
        if hasattr(route, "path") and route.path.startswith("/api/v1/"):
            # OpenAPI paths should contain this route
            assert route.path in paths or any(
                route.path.replace("{", "").replace("}", "") in p for p in paths
            ), f"Route {route.path} not found in OpenAPI schema"


def test_health_endpoint_documented():
    app = create_app()
    schema = app.openapi()
    assert "/api/v1/system/health" in schema["paths"]
