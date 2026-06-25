from wwra.config import AIRPORTS, WEATHER_BIN_ORDER
from wwra.utils import parse_months


def test_config_and_month_parser():
    assert "ORD" in AIRPORTS
    assert WEATHER_BIN_ORDER == ["W1 mild", "W2 moderate", "W3 severe", "W4 extreme"]
    assert parse_months("2024-01:2024-03") == {"2024-01", "2024-02", "2024-03"}
