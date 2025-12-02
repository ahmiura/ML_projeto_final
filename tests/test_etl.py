import pytest
from src.etl.processor import clean_text

def test_clean_text_basic():
    input_text = "Péssimo produto!!! Não comprem :("
    expected = "pessimo produto nao comprem"
    assert clean_text(input_text) == expected

def test_clean_text_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""

def test_clean_text_strip_numbers_and_extra_spaces():
    input_text = "  Produto 123   Excelente!!!  "
    out = clean_text(input_text)
    assert "123" not in out
    assert out == "produto excelente"