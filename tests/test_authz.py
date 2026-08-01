"""Tests for the optional Basic Auth helpers.

``authz`` takes an explicit ``env`` mapping so these run without Flask and
without touching the real environment.
"""

from __future__ import annotations

import authz

CREDS = {"CHAT_AUTH_USER": "tj", "CHAT_AUTH_PASSWORD": "hunter2"}


class TestConfiguredCredentials:
    def test_explicit_pair(self):
        assert authz.configured_credentials(CREDS) == ("tj", "hunter2")

    def test_shorthand(self):
        assert authz.configured_credentials({"CHAT_AUTH": "tj:hunter2"}) == ("tj", "hunter2")

    def test_shorthand_password_may_contain_colons(self):
        env = {"CHAT_AUTH": "tj:a:b:c"}
        assert authz.configured_credentials(env) == ("tj", "a:b:c")

    def test_shorthand_without_colon_is_ignored(self):
        assert authz.configured_credentials({"CHAT_AUTH": "nocolon"}) == ("", "")

    def test_username_is_trimmed(self):
        assert authz.configured_credentials({"CHAT_AUTH": "  tj  :pw"}) == ("tj", "pw")

    def test_explicit_pair_wins_over_shorthand(self):
        env = dict(CREDS, CHAT_AUTH="other:pw")
        assert authz.configured_credentials(env) == ("tj", "hunter2")

    def test_unset(self):
        assert authz.configured_credentials({}) == ("", "")


class TestAuthEnabled:
    def test_off_when_unset(self):
        assert authz.auth_enabled({}) is False

    def test_on_with_both(self):
        assert authz.auth_enabled(CREDS) is True

    def test_off_with_user_only(self):
        assert authz.auth_enabled({"CHAT_AUTH_USER": "tj"}) is False

    def test_off_with_password_only(self):
        assert authz.auth_enabled({"CHAT_AUTH_PASSWORD": "hunter2"}) is False


class TestCredentialsMatch:
    def test_correct_pair(self):
        assert authz.credentials_match("tj", "hunter2", CREDS) is True

    def test_wrong_password(self):
        assert authz.credentials_match("tj", "nope", CREDS) is False

    def test_wrong_user(self):
        assert authz.credentials_match("someone", "hunter2", CREDS) is False

    def test_denies_everything_when_auth_is_not_configured(self):
        # The important case: an unconfigured app must not accept blank creds.
        assert authz.credentials_match("", "", {}) is False
        assert authz.credentials_match("tj", "hunter2", {}) is False

    def test_none_values_are_rejected_not_crashed(self):
        assert authz.credentials_match(None, None, CREDS) is False
