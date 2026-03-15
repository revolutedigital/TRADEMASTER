"""Unit tests for schema registry."""
import pytest
import tempfile
from pathlib import Path
from app.core.schema_registry import SchemaRegistry


class TestSchemaRegistry:
    def test_register_and_validate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SchemaRegistry(storage_dir=tmpdir)

            schema = {
                "required": ["symbol", "price"],
                "fields": {"symbol": "str", "price": "float"},
                "ranges": {"price": {"min": 0}},
            }
            sv = reg.register("test_schema", schema)
            assert sv.version == 1

            # Valid data
            ok, errors = reg.validate("test_schema", {"symbol": "BTCUSDT", "price": 50000.0})
            assert ok
            assert not errors

            # Missing field
            ok, errors = reg.validate("test_schema", {"symbol": "BTCUSDT"})
            assert not ok
            assert any("price" in e for e in errors)

    def test_version_increment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SchemaRegistry(storage_dir=tmpdir)

            sv1 = reg.register("test", {"required": ["a"]})
            assert sv1.version == 1

            # Same schema -> same version
            sv2 = reg.register("test", {"required": ["a"]})
            assert sv2.version == 1

            # Different schema -> new version
            sv3 = reg.register("test", {"required": ["a", "b"]})
            assert sv3.version == 2

    def test_range_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SchemaRegistry(storage_dir=tmpdir)
            reg.register("price", {
                "required": ["value"],
                "fields": {"value": "float"},
                "ranges": {"value": {"min": 0, "max": 1000000}},
            })

            ok, _ = reg.validate("price", {"value": 50000})
            assert ok

            ok, errors = reg.validate("price", {"value": -1})
            assert not ok
