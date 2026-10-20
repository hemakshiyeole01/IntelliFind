"""App smoke test in demo mode. Run: python -m tests.test_app  (needs streamlit)"""
import os
from pathlib import Path
os.environ["ESCI_DEMO"] = "1"
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_app_demo():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception, at.exception
    at.text_input(key="q").set_value("samsung s23 case").run()
    assert not at.exception, at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "S23" in body, body
    at.sidebar.selectbox[0].set_value("hybrid").run()      # hybrid controls appear
    assert not at.exception, at.exception
    assert len(at.sidebar.slider) >= 2
    at.sidebar.selectbox[1].set_value("rrf").run()
    assert not at.exception, at.exception


if __name__ == "__main__":
    test_app_demo(); print("app smoke test passed")
