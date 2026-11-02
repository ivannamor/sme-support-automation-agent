"""
Tests for backend/auth.py and the account scripts. The clock is controlled via
auth._now; the database is the isolated test one.
"""

import re

import pytest
from streamlit.testing.v1 import AppTest

from backend import auth
from backend.auth import (
    LOGIN_LOCKOUT_SECONDS,
    MAX_FAILED_LOGINS,
    authenticate,
    hash_password,
    register_user,
    set_password,
)
from backend.database import create_user
from scripts import create_users
from scripts import set_password as set_password_script

PASSWORD = "a-long-enough-password"


@pytest.fixture(autouse=True)
def _fast_bcrypt(monkeypatch):
    gensalt = auth.bcrypt.gensalt
    monkeypatch.setattr(auth.bcrypt, "gensalt", lambda: gensalt(rounds=4))


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 1_000_000.0}
    monkeypatch.setattr(auth, "_now", lambda: now["t"])
    return now


@pytest.fixture
def manager(test_db, clock):
    register_user("mario", PASSWORD, "manager")


def _fail(username: str, times: int) -> None:
    for _ in range(times):
        assert authenticate(username, "wrong").status == "invalid"


def test_right_password_returns_the_role(manager):
    result = authenticate("mario", PASSWORD)
    assert result.status == "ok"
    assert result.role == "manager"


def test_wrong_password_and_unknown_user_look_the_same(manager):
    assert authenticate("mario", "wrong") == authenticate("nobody", "wrong")


def test_unknown_user_still_runs_bcrypt(manager, monkeypatch):
    calls = []
    monkeypatch.setattr(auth, "verify_password", lambda *args: calls.append(args) or False)
    authenticate("nobody", "wrong")
    assert len(calls) == 1


def test_account_locks_after_too_many_failures(manager):
    _fail("mario", MAX_FAILED_LOGINS)
    assert authenticate("mario", PASSWORD).status == "locked"


def test_right_password_on_the_last_allowed_attempt_works(manager):
    _fail("mario", MAX_FAILED_LOGINS - 1)
    assert authenticate("mario", PASSWORD).status == "ok"


def test_success_resets_the_count(manager):
    _fail("mario", MAX_FAILED_LOGINS - 1)
    authenticate("mario", PASSWORD)
    _fail("mario", MAX_FAILED_LOGINS - 1)
    assert authenticate("mario", PASSWORD).status == "ok"


def test_lock_ends_and_attempts_while_locked_do_not_extend_it(manager, clock):
    _fail("mario", MAX_FAILED_LOGINS)
    clock["t"] += LOGIN_LOCKOUT_SECONDS - 1
    assert authenticate("mario", PASSWORD).status == "locked"
    clock["t"] += 1
    assert authenticate("mario", PASSWORD).status == "ok"


def test_unknown_users_lock_too(test_db, clock):
    _fail("nobody", MAX_FAILED_LOGINS)
    assert authenticate("nobody", "wrong").status == "locked"


def test_username_variants_share_the_count(manager):
    _fail(" MARIO ", MAX_FAILED_LOGINS)
    assert authenticate("mario", PASSWORD).status == "locked"


def test_lock_on_one_account_leaves_the_others_alone(manager):
    register_user("anna", PASSWORD, "installer")
    _fail("mario", MAX_FAILED_LOGINS)
    assert authenticate("anna", PASSWORD).status == "ok"


def test_old_default_password_is_refused(test_db, clock):
    create_user("old", hash_password("change-me"), "manager")
    result = authenticate("old", "change-me")
    assert result.status == "default_password"
    assert result.role is None


@pytest.mark.parametrize("bad", ["short", "x" * 73, "è" * 40])
def test_weak_or_oversized_passwords_are_rejected(test_db, bad):
    with pytest.raises(ValueError):
        register_user("mario", bad, "manager")


def test_set_password_replaces_it_and_lifts_the_lock(manager):
    _fail("mario", MAX_FAILED_LOGINS)
    assert set_password("mario", "another-long-password")
    assert authenticate("mario", "another-long-password").status == "ok"


def test_set_password_on_unknown_user(test_db):
    assert not set_password("nobody", PASSWORD)


def test_create_users_script_prints_working_random_passwords(test_db, clock, capsys):
    create_users.main()
    output = capsys.readouterr().out
    assert "change-me" not in output
    found = dict(re.findall(r"Created \w+: (\w+)  password: (\S+)", output))
    assert set(found) == {"installer", "manager"}
    assert found["installer"] != found["manager"]
    assert authenticate("manager", found["manager"]).role == "manager"


def test_create_users_script_leaves_existing_accounts_alone(manager, capsys):
    register_user("manager", PASSWORD, "manager")
    create_users.main()
    assert "Already exists: manager" in capsys.readouterr().out
    assert authenticate("manager", PASSWORD).status == "ok"


def _login_page(username: str, password: str) -> AppTest:
    page = AppTest.from_file("app/pages/login.py", default_timeout=30).run()
    page.text_input[0].set_value(username)
    page.text_input[1].set_value(password)
    return page.button[-1].click().run()


def test_login_page_explains_each_refusal(test_db):
    register_user("mario", PASSWORD, "manager")
    create_user("old", hash_password("change-me"), "manager")

    assert _login_page("mario", "wrong").error[0].value == "Invalid credentials."
    assert _login_page("nobody", "wrong").error[0].value == "Invalid credentials."
    assert "default password" in _login_page("old", "change-me").error[0].value

    for _ in range(MAX_FAILED_LOGINS - 1):
        _login_page("mario", "wrong")
    assert "Too many failed attempts" in _login_page("mario", PASSWORD).error[0].value


def test_login_page_lets_the_right_password_in(test_db):
    register_user("mario", PASSWORD, "manager")
    assert _login_page("mario", PASSWORD).session_state.role == "manager"


def _typed(monkeypatch, *answers: str) -> None:
    replies = iter(answers)
    monkeypatch.setattr(set_password_script.getpass, "getpass", lambda _: next(replies))


def test_set_password_script(manager, monkeypatch):
    _typed(monkeypatch, "another-long-password", "another-long-password")
    assert set_password_script.main(["mario"]) == 0
    assert authenticate("mario", "another-long-password").status == "ok"


@pytest.mark.parametrize(
    ("username", "answers", "message"),
    [
        ("mario", ("first-long-password", "other-long-password"), "don't match"),
        ("mario", ("short", "short"), "at least"),
        ("nobody", (PASSWORD, PASSWORD), "No account"),
    ],
)
def test_set_password_script_failures(manager, monkeypatch, capsys, username, answers, message):
    _typed(monkeypatch, *answers)
    assert set_password_script.main([username]) == 1
    assert message in capsys.readouterr().out
    assert authenticate("mario", PASSWORD).status == "ok"
