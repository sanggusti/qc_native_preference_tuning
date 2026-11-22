"""No-network tests for the covariate scripts with fakes and fixtures."""

import math

from pipeline.covariates.bits_per_byte import aggregate, bits_per_byte
from pipeline.covariates.fertility import fertility_for_texts, fertility_stats, summarize
from pipeline.covariates.reference_audit import orthography_report, register_scores
from pipeline.covariates.similarity import (
    chrf_plusplus,
    content_tokens,
    identical_content_token_share,
)

FIXTURE = [
    "Budi has three apples and buys four more",
    "Siti gives away three of her ten marbles",
    "The school opens its new library today",
    "Farmers plant rice before the rains come",
    "My grandmother cooks rendang every Sunday",
    "Children play football after school",
    "The river floods the village each year",
    "She reads a book every single night",
    "Prices rise faster than wages here",
    "We take the night bus to Bandung",
]


class FakeTokenizer:
    """Whitespace tokenizer with stable ids; stands in for a base tokenizer."""

    def encode(self, text, add_special_tokens=False):
        return [abs(hash(word)) % 10000 for word in text.split()]


def test_fertility_stats_counts_tokens_chars_bytes():
    stats = fertility_stats([1, 2, 3], "ab cd")
    assert stats["tokens_per_item"] == 3
    assert stats["tokens_per_word"] == 1.5
    assert stats["chars_per_item"] == 5
    assert stats["bytes_per_item"] == 5


def test_fertility_for_texts_with_fake_tokenizer():
    summary = fertility_for_texts(FakeTokenizer(), FIXTURE)
    assert summary["n_items"] == 10
    assert summary["tokens_per_word"] == 1.0
    assert summary["tokens_per_item"] > 0
    assert summary["bytes_per_item"] >= summary["chars_per_item"]


def test_summarize_empty_input_yields_zeros():
    assert summarize([])["n_items"] == 0


def test_bits_per_byte_converts_nats_to_bits():
    assert bits_per_byte(math.log(2), 1) == 1.0
    assert bits_per_byte(0.0, 10) == 0.0
    assert bits_per_byte(5.0, 0) == 0.0


def test_aggregate_reports_spread_and_counts():
    result = aggregate([math.log(2), math.log(4)], [1, 1])
    assert result["bits_per_byte"] == 1.5
    assert result["stdev_per_item"] == 0.5
    assert result["n_items"] == 2
    assert result["total_bytes"] == 2


def test_content_tokens_excludes_numerals_proper_nouns_preserved():
    tokens = content_tokens("Budi has 3 apples in Bandung", frozenset({"apples"}))
    assert "3" not in tokens
    assert "bandung" not in tokens
    assert "apples" not in tokens
    assert "has" in tokens


def test_identical_content_token_share_boundaries():
    assert identical_content_token_share("a b c", "a b c") == 1.0
    assert identical_content_token_share("a b", "c d") == 0.0
    assert identical_content_token_share("", "") == 1.0


def test_chrf_plusplus_boundaries_and_monotonicity():
    assert chrf_plusplus("abc def", "abc def") == 100.0
    assert chrf_plusplus("abc", "xyz") == 0.0
    close = chrf_plusplus("the cat sits", "the cat sat")
    far = chrf_plusplus("the cat sits", "dogs run fast")
    assert 0.0 < far < close < 100.0


def test_register_scores_labels_ngoko_krama_mixed_unknown():
    ngoko, krama = ["aku", "kowe"], ["kula", "panjenengan"]
    assert register_scores("aku mangan", ngoko, krama)["label"] == "ngoko"
    assert register_scores("kula nedha", ngoko, krama)["label"] == "krama"
    assert register_scores("aku lan panjenengan", ngoko, krama)["label"] == "mixed"
    assert register_scores("mangan turu", ngoko, krama)["label"] == "unknown"


def test_orthography_report_counts_non_ascii():
    report = orthography_report(["caf\u00e9", "plain"])
    assert report["n_chars"] == 9
    assert report["non_ascii_share"] == 1 / 9
    assert report["top_codepoints"][0]["count"] == 1
    assert "00E9" in report["top_codepoints"][0]["codepoint"]
