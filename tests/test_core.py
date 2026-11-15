import sqlite3

import pytest

from tms.core import audit, calibration, geometry, plates, rules


@pytest.mark.parametrize("raw, text", [
    ("UP32AB1234", "UP32AB1234"),
    ("up 32 ab 1234", "UP32AB1234"),
    ("MH-12-DE-1433", "MH12DE1433"),
    ("IND DL8CAF5030", "DL8CAF5030"),
    ("KA05MX8421", "KA05MX8421"),
    ("22BH1234AA", "22BH1234AA"),
    ("UP32ABI234", "UP32AB1234"),
    ("UP3ZAB1Z34", "UP3ZAB1234"),
    ("MHO1AB1234", "MH01AB1234"),
    ("22BHI234AA", "22BH1234AA"),
])
def test_valid_plates(raw, text):
    plate = plates.parse(raw)
    assert plate.valid and plate.text == text


@pytest.mark.parametrize("raw", ["", "HELLO", "XX32AB1234", "UP32AB12", "1234567890", "UPAB1234"])
def test_invalid_plates(raw):
    assert not plates.parse(raw).valid


def test_riders_link_to_the_bike_under_them():
    bikes = {1: (100, 200, 300, 400), 2: (600, 200, 800, 400)}
    persons = {10: (150, 80, 250, 360), 11: (180, 90, 280, 370), 12: (650, 80, 750, 360),
               13: (400, 100, 460, 380)}
    assert geometry.link_riders(persons, bikes) == {1: [10, 11], 2: [12]}


def test_crop_boxes_stay_inside_their_parent():
    head = geometry.head_box((100, 100, 200, 400))
    assert head == (100, 100, 200, 190)
    x1, y1, x2, y2 = geometry.windscreen_box((0, 0, 400, 300))
    assert 0 <= x1 < x2 <= 400 and 0 <= y1 < y2 <= 300


def test_decide_needs_enough_clear_frames():
    assert rules.decide([("no_helmet", 0.9)] * 4, "no_helmet") is None
    assert rules.decide([("no_helmet", 0.9)] * 4 + [("unclear", 0.0)] * 20, "no_helmet") is None
    decision = rules.decide([("no_helmet", 0.9)] * 8 + [("helmet", 0.9)] * 2, "no_helmet")
    assert decision.hits == 8 and decision.usable == 10
    assert decision.confidence == pytest.approx(0.72)


def test_decide_ignores_a_few_bad_frames():
    assert rules.decide([("helmet", 0.9)] * 8 + [("no_helmet", 0.99)] * 2, "no_helmet") is None


def test_violations_by_vehicle_kind():
    votes = {"helmet": [("no_helmet", 0.9)] * 6, "riders": [("triple", 0.8)] * 6,
             "seatbelt": [("no_belt", 0.9)] * 6}
    assert [v[0] for v in rules.violations("motorcycle", votes)] == ["no_helmet", "triple_riding"]
    assert [v[0] for v in rules.violations("car", votes)] == ["no_seatbelt"]


def test_wilson_bound_is_cautious_on_small_samples():
    assert calibration.wilson_lower(20, 20) < 0.9
    assert calibration.wilson_lower(1000, 1000) > 0.99
    assert calibration.wilson_lower(0, 0) == 0


def test_best_threshold_needs_volume_and_precision():
    lucky = [(0.9, 1)] * 20
    assert calibration.best_threshold(lucky, 0.99, 300) is None
    noisy = [(0.6, i % 5 != 0) for i in range(500)] + [(0.9, 1) for _ in range(600)]
    best = calibration.best_threshold(noisy, 0.99, 300)
    assert best["threshold"] == 0.9 and best["n"] == 600
    assert calibration.best_threshold([(0.9, i % 10 != 0) for i in range(600)], 0.99, 300) is None


def test_audit_chain_detects_tampering(conn):
    for i in range(3):
        audit.append(conn, None, "test", f"item:{i}", {"i": i})
    conn.commit()
    assert audit.verify(conn) is None
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE audit_log SET action = 'x' WHERE id = 2")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM audit_log WHERE id = 2")
    conn.execute("DROP TRIGGER audit_no_update")
    conn.execute("UPDATE audit_log SET target = 'item:9' WHERE id = 2")
    assert audit.verify(conn) == 2
