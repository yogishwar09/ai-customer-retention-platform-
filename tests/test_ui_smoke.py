"""UI smoke tests: every page renders for a signed-in admin of the seeded demo org."""

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted(p.name for p in (ROOT / "app_pages").glob("*.py"))


@pytest.fixture(scope="module")
def admin(seeded):
    from retentioniq.storage.accounts import Accounts
    from tests.conftest import DEMO_PASSWORD
    return Accounts().authenticate("admin", DEMO_PASSWORD)


def _page(name, principal):
    at = AppTest.from_file(str(ROOT / "app_pages" / name), default_timeout=90)
    at.session_state["principal"] = principal
    at.session_state["last_activity"] = time.time()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, admin):
    at = _page(page, admin)
    at.run()
    assert not at.exception, f"{page}: {at.exception}"


def test_score_page_scores_a_customer(admin):
    at = _page("score.py", admin)
    at.run()
    next(b for b in at.button if b.label == "Score customer").click().run()
    assert not at.exception
    assert any("Estimated churn risk" in m.value for m in at.markdown)


def test_login_gate(seeded):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    assert not at.exception
    assert any(b.label == "Sign in" for b in at.button) or len(at.text_input) == 2


def test_viewer_cannot_open_admin_pages(seeded):
    from retentioniq.storage.accounts import Accounts
    from tests.conftest import DEMO_PASSWORD
    viewer = Accounts().authenticate("viewer", DEMO_PASSWORD)
    at = _page("admin.py", viewer)
    at.run()
    assert any("does not have access" in e.value for e in at.error)
