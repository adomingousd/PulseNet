import numpy as np

from world_model import pamap2
from world_model.metrics import CHANNELS, forecast_report, persistence_forecast, zero_forecast
from tests.fake_pamap2 import make_subject_file


def _build(tmp_path, subject=101, **kw):
    path = make_subject_file(tmp_path / "Protocol" / f"subject{subject}.dat", subject, **kw)
    return pamap2.build_subject(pamap2.read_subject(path), subject)


def test_shapes_and_dtypes(tmp_path):
    d = _build(tmp_path)
    n = len(d["split"])
    assert d["x"].shape == (n, 19, 100)
    assert d["y"].shape == (n, 19, 10)
    assert d["x"].dtype == np.float32
    assert list(d["channels"]) == CHANNELS
    assert not np.isnan(d["x"]).any() and not np.isnan(d["y"]).any()


def test_transient_activity_dropped(tmp_path):
    d = _build(tmp_path)
    assert 0 not in set(d["activity"].tolist())


def test_every_long_bout_has_validation(tmp_path):
    d = _build(tmp_path)
    val_acts = set(d["activity"][d["split"] == 1].tolist())
    # the 5 s stair bout (250 steps) is too short to split; the long ones must have val data
    assert {1, 4, 5} <= val_acts
    assert 12 not in val_acts


def test_no_train_val_overlap(tmp_path):
    d = _build(tmp_path)
    for seg in np.unique(d["segment"]):
        m = d["segment"] == seg
        tr = d["start"][m & (d["split"] == 0)]
        va = d["start"][m & (d["split"] == 1)]
        if len(va):
            last_train_end = tr.max() + pamap2.SAMPLE_LEN
            assert va.min() - last_train_end >= pamap2.GAP


def test_train_split_is_standardized(tmp_path):
    d = _build(tmp_path)
    tr = d["x"][d["split"] == 0]
    # windows overlap, so stats are approximate, but close to 0 / 1
    assert np.abs(tr.mean(axis=(0, 2))).max() < 0.25
    assert np.abs(tr.std(axis=(0, 2)) - 1).max() < 0.25


def test_long_dropout_splits_segment(tmp_path):
    with_gap = _build(tmp_path / "a", long_dropout=True)
    without = _build(tmp_path / "b", long_dropout=False)
    assert len(np.unique(with_gap["segment"])) == len(np.unique(without["segment"])) + 1


def test_find_raw_file_prefers_protocol(tmp_path):
    make_subject_file(tmp_path / "Optional" / "subject101.dat", 101)
    make_subject_file(tmp_path / "Protocol" / "subject101.dat", 101)
    assert "Protocol" in str(pamap2.find_raw_file(tmp_path, 101))


def test_save_roundtrip(tmp_path):
    d = _build(tmp_path)
    path = pamap2.save_subject(d, tmp_path / "processed")
    z = np.load(path)
    assert z["x"].shape == d["x"].shape and int(z["subject"]) == 101


def test_stream_matches_training_window(tmp_path):
    path = make_subject_file(tmp_path / "Protocol" / "subject101.dat", 101)
    df = pamap2.read_subject(path)
    d = pamap2.build_subject(df, 101)
    stream = pamap2.build_stream(df, 101)

    # Last window: its segment and start are both non-zero, so a wrong segment
    # ID or row offset in the stream would make this test fail.
    i = len(d["split"]) - 1
    start = d["start"][i]
    assert d["segment"][i] > 0 and start > 0

    # Take the same 110 rows from the stream, z-score them with the training
    # file's stats, and flip [time, channels] to the windows' [channels, time].
    rows = stream["rows"][stream["segment"] == d["segment"][i]]
    z = ((rows[start : start + pamap2.SAMPLE_LEN] - d["mean"]) / d["std"]).T

    # atol allows for float32 rounding, which the default np.allclose tolerance
    # can reject for z-scores near zero. An off-by-one row is far larger than this.
    assert np.allclose(z[:, :pamap2.INPUT_LEN], d["x"][i], atol=1e-4)
    assert np.allclose(z[:, pamap2.INPUT_LEN:], d["y"][i], atol=1e-4)


def test_stream_and_windows_keep_same_segments(tmp_path):
    # build_subject() and build_stream() each have their own copy of the rule
    # that drops segments under 110 steps. This test fails if the two copies
    # stop matching, because the segment IDs would no longer line up.
    # At 100 Hz raw: activity 4 lasts 218 samples = 109 steps at 50 Hz (dropped),
    # activity 5 lasts 220 samples = 110 steps (kept). Activity 6 comes last, so
    # its ID shifts if the dropped segment is counted by only one of the builders.
    bouts = [(1, 30), (0, 1), (4, 2.18), (0, 1), (5, 2.2), (0, 1), (6, 30)]
    path = make_subject_file(tmp_path / "Protocol" / "subject101.dat", 101,
                             bouts=bouts, long_dropout=False)
    df = pamap2.read_subject(path)
    d = pamap2.build_subject(df, 101)
    stream = pamap2.build_stream(df, 101)

    # Make sure the fake data really lands on each side of the 110-step limit.
    assert 4 not in d["activity"] and 5 in d["activity"]

    # Both builders must end up with the same segment IDs.
    assert np.array_equal(np.unique(stream["segment"]), np.unique(d["segment"]))

    # Same IDs could still point at different data, so check every window too.
    for i in range(len(d["split"])):
        rows = stream["rows"][stream["segment"] == d["segment"][i]]
        start = d["start"][i]
        z = ((rows[start : start + pamap2.SAMPLE_LEN] - d["mean"]) / d["std"]).T
        assert np.allclose(z[:, :pamap2.INPUT_LEN], d["x"][i], atol=1e-4)
        assert np.allclose(z[:, pamap2.INPUT_LEN:], d["y"][i], atol=1e-4)


def test_baseline_metrics():
    x = np.ones((4, 19, 100), dtype=np.float32)
    y = np.ones((4, 19, 10), dtype=np.float32)
    assert forecast_report(persistence_forecast(x, 10), y)["mse"] == 0.0
    r = forecast_report(zero_forecast(x, 10), y)
    assert r["mse"] == 1.0 and r["mse_h10"] == 1.0 and r["mse_hr"] == 1.0
