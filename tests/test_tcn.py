import pytest

torch = pytest.importorskip("torch")

from world_model.tcn import TCN, TCNConfig, count_parameters, expected_parameters, receptive_field  # noqa: E402


def test_output_shape():
    m = TCN()
    out = m(torch.randn(4, 19, 100))
    assert out.shape == (4, 19, 10)


def test_parameter_count_matches_spec():
    m = TCN()
    assert count_parameters(m) == expected_parameters(m.cfg) == 30_494


def test_receptive_field_is_61():
    cfg = TCNConfig()
    assert receptive_field(cfg) == 61
    m = TCN(cfg)
    x = torch.randn(1, 19, 100, requires_grad=True)
    m(x).sum().backward()
    used = (x.grad.abs().sum(dim=1)[0] > 0).nonzero().flatten()
    # only the last 61 input steps can affect the forecast
    assert used.min().item() == 100 - 61
    assert used.max().item() == 99


def test_causal_future_does_not_leak():
    torch.manual_seed(0)
    m = TCN().eval()
    x = torch.randn(1, 19, 100)
    x2 = x.clone()
    x2[:, :, :39] += 5.0          # change steps outside the receptive field
    assert torch.allclose(m(x), m(x2))


def test_residual_to_last():
    torch.manual_seed(0)
    m = TCN(TCNConfig(residual_to_last=True))
    torch.nn.init.zeros_(m.head.weight)
    torch.nn.init.zeros_(m.head.bias)
    x = torch.randn(2, 19, 100)
    out = m(x)
    assert torch.allclose(out, x[:, :, -1:].expand(-1, -1, 10))


def test_same_seed_same_weights():
    torch.manual_seed(1)
    a = TCN()
    torch.manual_seed(1)
    b = TCN()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_config_roundtrip():
    cfg = TCNConfig(hidden=16, dilations=(1, 2, 4))
    assert TCNConfig.from_dict(cfg.to_dict()) == cfg
