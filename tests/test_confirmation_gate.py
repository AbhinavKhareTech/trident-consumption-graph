"""Tests for confirmation gate."""
from bgi_trident.orchestrator.confirmation import ConfirmationGate


def test_confirm_flow():
    gate = ConfirmationGate()
    gate.request_confirmation("S001", "Biryani Rs 350", 350.0, "kn", [])
    assert gate.is_pending("S001")
    result = gate.process_response("S001", "haan")
    assert result == "confirmed"
    assert not gate.is_pending("S001")


def test_cancel_flow():
    gate = ConfirmationGate()
    gate.request_confirmation("S002", "Test", 100.0, "en", [])
    result = gate.process_response("S002", "cancel")
    assert result == "cancelled"


def test_unclear_response():
    gate = ConfirmationGate()
    gate.request_confirmation("S003", "Test", 100.0, "en", [])
    result = gate.process_response("S003", "maybe later")
    assert result == "unclear"
    assert gate.is_pending("S003")
