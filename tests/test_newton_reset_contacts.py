import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "newton_cell"))
from sim import Cell


class FakeModel:
    def __init__(self):
        self.calls = []

    def collide(self, state, contacts):
        self.calls.append((state, contacts))


class NewtonResetContactTests(unittest.TestCase):
    def test_refresh_contacts_uses_reset_state_and_shared_contact_buffer(self):
        cell = Cell.__new__(Cell)
        cell.state_0 = object()
        cell.contacts = object()
        cell.model = FakeModel()

        cell.refresh_contacts()

        self.assertEqual(cell.model.calls, [(cell.state_0, cell.contacts)])

    def test_refresh_contacts_accepts_explicit_state(self):
        cell = Cell.__new__(Cell)
        cell.state_0 = object()
        cell.contacts = object()
        cell.model = FakeModel()
        alternate_state = object()

        cell.refresh_contacts(alternate_state)

        self.assertEqual(cell.model.calls, [(alternate_state, cell.contacts)])


if __name__ == "__main__":
    unittest.main()
