import numpy as np
import pytest

torch = pytest.importorskip("torch")

from world_model.data import make_loader  # noqa: E402
from world_model.metrics import persistence_forecast, forecast_report  # noqa: E402
from world_model.tcn import TCN, TCNConfig  # noqa: E402
from world_model.train import evaluate, set_seed, train  # noqa: E402


def _sine_windows(n=256, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(110)
    phase = rng.uniform(0, 2 * np.pi, size=(n, 19, 1))
    freq = rng.uniform(0.05, 0.2, size=(n, 19, 1))
    w = np.sin(freq * t + phase).astype(np.float32)
    return w[:, :, :100], w[:, :, 100:]


def test_training_reduces_loss():
    set_seed(0)
    x, y = _sine_windows()
    model = TCN()
    loader = make_loader(x, y, batch_size=32, shuffle=True)
    before = evaluate(model, x, y)["mse"]
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(15):
        train(model, loader, epochs=1, optimizer=opt)
    after = evaluate(model, x, y)["mse"]
    assert after < 0.7 * before


def test_same_seed_is_reproducible():
    x, y = _sine_windows(64)
    results = []
    for _ in range(2):
        set_seed(3)
        model = TCN()
        train(model, make_loader(x, y, 16, shuffle=True, seed=3), epochs=2)
        results.append(evaluate(model, x, y)["mse"])
    assert results[0] == pytest.approx(results[1], rel=1e-6)


def test_train_reports_examples():
    x, y = _sine_windows(40)
    m = train(TCN(), make_loader(x, y, 16, shuffle=False), epochs=1)
    assert m["num_examples"] == 40 and m["train_loss"] > 0


def test_fedprox_term_keeps_weights_closer():
    x, y = _sine_windows(64)
    drift = {}
    for mu in (0.0, 10.0):
        set_seed(0)
        model = TCN()
        start = [p.detach().clone() for p in model.parameters()]
        train(model, make_loader(x, y, 16, shuffle=True), epochs=3, lr=3e-3, proximal_mu=mu, global_params=start)
        drift[mu] = sum(((p - s) ** 2).sum().item() for p, s in zip(model.parameters(), start))
    assert drift[10.0] < drift[0.0]


def test_residual_model_starts_near_persistence():
    x, y = _sine_windows(64)
    set_seed(0)
    model = TCN(TCNConfig(residual_to_last=True))
    torch.nn.init.zeros_(model.head.weight)
    torch.nn.init.zeros_(model.head.bias)
    assert evaluate(model, x, y)["mse"] == pytest.approx(
        forecast_report(persistence_forecast(x, 10), y)["mse"], rel=1e-5)
