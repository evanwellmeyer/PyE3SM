"""Test-wide guards for the port.

`scalar / tensor` in PyTorch is reciprocal-then-multiply, not IEEE division,
so it breaks exact agreement with the reference. Port code must use
eamxx_torch.ekat.rdiv. This autouse fixture makes the operator raise while
port code runs under test.
"""
import pytest
import torch


@pytest.fixture(autouse=True)
def _forbid_scalar_over_tensor(monkeypatch):
    def _raise(self, other):
        raise AssertionError("scalar / tensor is inexact in PyTorch; use eamxx_torch.ekat.rdiv")
    monkeypatch.setattr(torch.Tensor, "__rtruediv__", _raise)
    yield
