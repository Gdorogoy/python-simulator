"""Distance curriculum (app/training/distance_buckets.py): the --buckets parser, its summary, and the time clock.
Plain asserts, like the rest of app/test (no pytest needed)."""
from app.training.distance_buckets import (
    DEFAULT_BUCKETS,
    describe_buckets,
    parse_buckets,
    time_bonus_budget_s,
)


def test_default_buckets_parse_to_three_ranges():
    assert parse_buckets(DEFAULT_BUCKETS) == [
        (100.0, 200.0, 0.25),
        (500.0, 800.0, 0.35),
        (1000.0, 1500.0, 0.40),
    ]


def test_parse_buckets_accepts_optional_weights_and_whitespace():
    assert parse_buckets(" 200-400 : 2 , 800-1200 ") == [(200.0, 400.0, 2.0), (800.0, 1200.0, 1.0)]
    assert parse_buckets(None) is None
    for off in ("", " none ", "off", "0"):
        assert parse_buckets(off) is None


def test_parse_buckets_rejects_bad_items():
    for spec in ("0-100:1", "200-100:1", "100-200:0", "100-200:-1", "abc", "100-200:x", "none,100-200:1"):
        try:
            parse_buckets(spec)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad spec {spec!r}")


def test_describe_buckets_normalises_weights():
    assert describe_buckets([(100.0, 200.0, 1.0), (500.0, 800.0, 3.0)]) == "100-200 m 25% | 500-800 m 75%"


def test_time_bonus_budget_follows_distance_and_clamps():
    assert time_bonus_budget_s(1200.0, 30.0, 7.5, 62.5) == 40.0
    assert time_bonus_budget_s(150.0, 30.0, 7.5, 62.5) == 7.5      # floor
    assert time_bonus_budget_s(1e6, 30.0, 7.5, 62.5) == 62.5       # ceiling
    assert time_bonus_budget_s(1200.0, 0.0, 7.5, 62.5) is None     # 0 = keep the fixed clock
