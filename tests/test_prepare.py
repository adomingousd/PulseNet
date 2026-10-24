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


def test_baseline_metrics():
    x = np.ones((4, 19, 100), dtype=np.float32)
    y = np.ones((4, 19, 10), dtype=np.float32)
    assert forecast_report(persistence_forecast(x, 10), y)["mse"] == 0.0
    r = forecast_report(zero_forecast(x, 10), y)
    assert r["mse"] == 1.0 and r["mse_h10"] == 1.0 and r["mse_hr"] == 1.0
