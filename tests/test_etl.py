"""Test suite for text cleaning functions."""

import pytest
from src.etl.processor import clean_text


@pytest.mark.parametrize(
    "input_text, expected",
    [
        # Test case: Basic cleaning with accents, punctuation, and case
        ("Péssimo produto!!! Não comprem :(", "pessimo produto nao comprem"),
        # Test case: Empty string should return empty string
        ("", ""),
        # Test case: None input should return empty string
        (None, ""),
        # Test case: Stripping numbers and extra spaces
        ("  Produto 123   Excelente!!!  ", "produto excelente"),
        # Test case: Only numbers and punctuation
        ("123 !@#$%", ""),
    ],
)
def test_clean_text(input_text, expected):
    """Test the clean_text function with various inputs."""
    assert clean_text(input_text) == expected