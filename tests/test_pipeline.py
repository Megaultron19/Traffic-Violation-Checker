from fakes import Scene, frame, labeller, plate_reader

from tms.pipeline import Pipeline


def run(pipe, frames=10):
    found = []
    for idx in range(1, frames + 1):
        found += pipe.step(idx, frame())
    return found + pipe.flush()


def test_no_helmet_and_no_seatbelt_become_findings():
    pipe = Pipeline(Scene(), labeller("no_helmet"), labeller("no_belt"), plate_reader("UP32AB1234"))
    found = {f.violation: f for f in run(pipe)}
    assert set(found) == {"no_helmet", "no_seatbelt"}
    helmet = found["no_helmet"]
    assert helmet.track_id == 1 and helmet.plate.valid and helmet.plate.text == "UP32AB1234"
    assert helmet.frame_jpg[:2] == b"\xff\xd8" and helmet.crop_jpg and helmet.plate_jpg


def test_compliant_traffic_produces_nothing():
    pipe = Pipeline(Scene(), labeller("helmet"), labeller("belt"), plate_reader("UP32AB1234"))
    assert run(pipe) == []


def test_unclear_frames_never_make_a_case():
    pipe = Pipeline(Scene(), labeller("unclear"), labeller("unclear"))
    assert run(pipe, 40) == []


def test_triple_riding_needs_three_riders():
    assert [f.violation for f in run(Pipeline(Scene(riders=3, car=False)))] == ["triple_riding"]
    assert run(Pipeline(Scene(riders=2, car=False))) == []


def test_too_few_frames_is_not_enough():
    pipe = Pipeline(Scene(), labeller("no_helmet"), labeller("no_belt"))
    assert run(pipe, 3) == []


def test_track_closes_once_it_leaves_the_scene():
    scene = Scene(car=False)
    pipe = Pipeline(scene, labeller("no_helmet"), stale_after=5)
    for idx in range(1, 11):
        assert pipe.step(idx, frame()) == []
    scene.__class__ = type("Empty", (), {"__call__": lambda self, image: []})
    found = []
    for idx in range(11, 20):
        found += pipe.step(idx, frame())
    assert [f.violation for f in found] == ["no_helmet"]
    assert pipe.flush() == []


def test_unreadable_plate_is_kept_but_marked_invalid():
    pipe = Pipeline(Scene(car=False), labeller("no_helmet"), plates=plate_reader("??7"))
    finding = run(pipe)[0]
    assert finding.plate is not None and not finding.plate.valid
