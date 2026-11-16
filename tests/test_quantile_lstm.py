"""Invariant tests for `caudal.models.quantile_lstm` (H3).

Two properties carry the experiment and would corrupt it silently if broken: the pinball loss
must actually be the pinball loss (an asymmetry error would train the wrong target), and
predicted quantiles must never cross (a q90 below q50 makes "the 90th percentile forecast"
meaningless while still producing plausible numbers).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from caudal.models.quantile_lstm import (
    DEFAULT_QUANTILES,
    QuantileLSTM,
    pinball_loss,
    predict_quantiles,
)


class TestPinballLoss:
    def test_matches_closed_form(self):
        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([2.5])
        q = torch.tensor([0.1, 0.5, 0.9])
        # errors: 1.5, 0.5, -0.5
        expected = np.mean(
            [
                max(0.1 * 1.5, (0.1 - 1) * 1.5),
                max(0.5 * 0.5, (0.5 - 1) * 0.5),
                max(0.9 * -0.5, (0.9 - 1) * -0.5),
            ]
        )
        assert pinball_loss(pred, target, q).item() == pytest.approx(expected)

    def test_is_zero_only_at_perfect_prediction(self):
        q = torch.tensor([0.25, 0.5, 0.75])
        target = torch.tensor([4.0])
        perfect = torch.tensor([[4.0, 4.0, 4.0]])
        assert pinball_loss(perfect, target, q).item() == pytest.approx(0.0)
        assert pinball_loss(perfect + 0.3, target, q).item() > 0

    def test_high_quantile_penalises_under_prediction_more(self):
        """The whole point: a q90 head is pushed UP, which the Gaussian mean never is."""
        q = torch.tensor([0.9])
        target = torch.tensor([10.0])
        under = pinball_loss(torch.tensor([[9.0]]), target, q).item()  # missed low by 1
        over = pinball_loss(torch.tensor([[11.0]]), target, q).item()  # missed high by 1
        assert under > over
        assert under == pytest.approx(0.9)
        assert over == pytest.approx(0.1)

    def test_low_quantile_penalises_over_prediction_more(self):
        q = torch.tensor([0.1])
        target = torch.tensor([10.0])
        under = pinball_loss(torch.tensor([[9.0]]), target, q).item()
        over = pinball_loss(torch.tensor([[11.0]]), target, q).item()
        assert over > under

    def test_median_head_is_symmetric(self):
        q = torch.tensor([0.5])
        target = torch.tensor([10.0])
        a = pinball_loss(torch.tensor([[9.0]]), target, q).item()
        b = pinball_loss(torch.tensor([[11.0]]), target, q).item()
        assert a == pytest.approx(b)


class TestQuantileLSTM:
    def test_output_shape_and_order(self):
        m = QuantileLSTM(n_features=5, hidden_size=8)
        out = m(torch.randn(7, 12, 5))
        assert out.shape == (7, len(DEFAULT_QUANTILES))

    def test_quantiles_never_cross(self):
        """Structural, so it must hold even at random init and on random inputs."""
        torch.manual_seed(0)
        m = QuantileLSTM(n_features=5, hidden_size=8)
        out = m(torch.randn(64, 12, 5)).detach().numpy()
        diffs = np.diff(out, axis=1)
        assert (diffs >= -1e-6).all(), f"crossing found, min step {diffs.min()}"

    def test_no_crossing_after_training_steps(self):
        torch.manual_seed(0)
        m = QuantileLSTM(n_features=4, hidden_size=8)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        q = m._q
        for _ in range(25):
            x, y = torch.randn(32, 10, 4), torch.randn(32) * 5
            opt.zero_grad()
            loss = pinball_loss(m(x), y, q)
            loss.backward()
            opt.step()
        out = m(torch.randn(32, 10, 4)).detach().numpy()
        assert (np.diff(out, axis=1) >= -1e-6).all()

    def test_rejects_unsorted_or_invalid_quantiles(self):
        with pytest.raises(ValueError, match="sorted"):
            QuantileLSTM(n_features=3, quantiles=(0.5, 0.1))
        with pytest.raises(ValueError, match="strictly inside"):
            QuantileLSTM(n_features=3, quantiles=(0.0, 0.5))
        with pytest.raises(ValueError, match="strictly inside"):
            QuantileLSTM(n_features=3, quantiles=(0.5, 1.0))

    def test_median_index(self):
        m = QuantileLSTM(n_features=3, quantiles=(0.1, 0.5, 0.9))
        assert m.median_index == 1

    def test_loss_and_parameterisation_recover_true_quantiles(self):
        """Optimised as free parameters, the monotone head must hit the empirical quantiles.

        This isolates the loss + parameterisation from the LSTM. It is the check that told us
        the collapse seen at high learning rates was an optimisation artifact and not a bug in
        either (see the module docstring).
        """
        torch.manual_seed(0)
        y = torch.distributions.LogNormal(0.0, 1.0).sample((4096,))
        q = torch.tensor([0.1, 0.5, 0.9])
        truth = np.quantile(y.numpy(), [0.1, 0.5, 0.9])

        raw = torch.zeros(1, 3, requires_grad=True)
        opt = torch.optim.Adam([raw], lr=0.05)
        for _ in range(4000):
            opt.zero_grad()
            base, steps = raw[:, :1], torch.nn.functional.softplus(raw[:, 1:])
            pred = torch.cat([base, base + torch.cumsum(steps, 1)], 1)
            pinball_loss(pred.expand(len(y), 3), y, q).backward()
            opt.step()
        base, steps = raw[:, :1], torch.nn.functional.softplus(raw[:, 1:])
        got = torch.cat([base, base + torch.cumsum(steps, 1)], 1).detach().numpy()[0]
        assert np.allclose(got, truth, atol=0.02), f"{got} vs {truth}"

    def test_learns_an_asymmetric_target(self):
        """On a skewed target the q90 head must sit well above q50 - the property the Gaussian
        mean cannot express, and the reason this model exists.

        lr=1e-3 deliberately: higher rates saturate the LSTM and collapse the heads together
        (module docstring). This is the regression test for that failure mode.
        """
        torch.manual_seed(0)
        x = torch.randn(1024, 6, 3)
        y = torch.distributions.LogNormal(0.0, 1.0).sample((1024,))
        m = QuantileLSTM(n_features=3, hidden_size=16, quantiles=(0.1, 0.5, 0.9))
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(1500):
            opt.zero_grad()
            pinball_loss(m(x), y, m._q).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        pred = m(x).detach().numpy().mean(axis=0)
        assert pred[0] < pred[1] < pred[2]
        # true spread is ~3.4; anything under 1.0 means the heads have collapsed
        assert pred[2] - pred[0] > 1.0, f"quantiles collapsed: {pred}"


class TestPredictQuantiles:
    def test_skips_gap_days_and_aligns_outputs(self):
        import pandas as pd

        idx = pd.date_range("2000-01-01", periods=120, freq="D")
        df = pd.DataFrame({"a": np.random.randn(120), "b": np.random.randn(120),
                           "streamflow": np.abs(np.random.randn(120))}, index=idx)
        df.loc[df.index[40:45], "streamflow"] = np.nan  # gap days yield no sample

        qp, qobs, qidx = predict_quantiles(QuantileLSTM(n_features=2, hidden_size=8), df, ["a", "b"], "streamflow", lookback=10)
        assert qp.shape == (len(qidx), len(DEFAULT_QUANTILES)) and len(qobs) == len(qidx) == 120 - 10 - 5
        assert not set(qidx) & set(df.index[40:45])
