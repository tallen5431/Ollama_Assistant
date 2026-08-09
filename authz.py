"""Optional HTTP Basic Auth for exposing the chat beyond the LAN.

Pure and dependency-light (only ``os`` + ``hmac``) so it can be unit-tested
without Flask; ``app.py`` wires it into a Flask ``before_request`` hook.

Enable by setting either:
  * ``CHAT_AUTH_USER`` and ``CHAT_AUTH_PASSWORD``, or
  * ``CHAT_AUTH`` = ``"user:password"`` (shorthand).

When neither is set, auth is OFF and the app behaves as a plain LAN service.
Basic Auth sends the password on every request, so only expose the app over
HTTPS (e.g. a Tailscale Funnel or Cloudflare Tunnel that terminates TLS).
"""

from __future__ import annotations

import hmac
import os
from typing import Mapping, Optional, Tuple


def configured_credentials(env: Optional[Mapping[str, str]] = None) -> Tuple[str, str]:
    """Return the (user, password) the operator configured, or ("", "")."""
    env = env if env is not None else os.environ
    user = (env.get("CHAT_AUTH_USER") or "").strip()
    pw = env.get("CHAT_AUTH_PASSWORD") or ""
    if not user:
        combined = env.get("CHAT_AUTH") or ""
        if ":" in combined:
            user, pw = combined.split(":", 1)
            user = user.strip()
    return user, pw


def auth_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """True only when a non-empty user AND password are configured."""
    user, pw = configured_credentials(env)
    return bool(user and pw)


def misconfigured(env: Optional[Mapping[str, str]] = None) -> str:
    """What is wrong with a half-configured pair, or "" when nothing is.

    Auth off is the default and a legitimate choice — behind Tailscale there is
    nothing to authenticate to. Auth off *because a setting did not take* is a
    different thing entirely, and it looked exactly the same: the app came up,
    said "Auth: OFF (LAN only)", and served every request to anyone who could
    reach it, while whoever set CHAT_AUTH_USER believed otherwise.

    Reported rather than fatal. Refusing to start would lock someone out of
    their own box over a typo, from a machine they may only reach through the
    app they have just stopped.
    """
    env = env if env is not None else os.environ
    user = (env.get("CHAT_AUTH_USER") or "").strip()
    pw = env.get("CHAT_AUTH_PASSWORD") or ""
    combined = env.get("CHAT_AUTH") or ""
    if auth_enabled(env):
        # Both forms set and disagreeing: CHAT_AUTH_USER wins the name and
        # CHAT_AUTH is ignored entirely, which is not what setting both looks
        # like it should do.
        if user and ":" in combined and combined.split(":", 1)[0].strip() != user:
            return ("CHAT_AUTH and CHAT_AUTH_USER are both set and name different "
                    "users. CHAT_AUTH_USER wins and CHAT_AUTH is ignored.")
        return ""
    if user and not pw:
        return ("CHAT_AUTH_USER is set but CHAT_AUTH_PASSWORD is empty, so "
                "authentication is OFF and every request is being served.")
    if pw and not user:
        return ("CHAT_AUTH_PASSWORD is set but CHAT_AUTH_USER is empty, so "
                "authentication is OFF and every request is being served.")
    if combined and ":" not in combined:
        return ("CHAT_AUTH is set but has no ':' in it — it must be "
                "\"user:password\" — so authentication is OFF.")
    if combined and not combined.split(":", 1)[1]:
        return ("CHAT_AUTH names a user with an empty password, so "
                "authentication is OFF.")
    return ""


def credentials_match(
    username: Optional[str],
    password: Optional[str],
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Constant-time check of supplied credentials against the configured pair.

    Returns False when auth isn't configured, so callers can't be tricked into
    allowing access with empty credentials.
    """
    user, pw = configured_credentials(env)
    if not (user and pw):
        return False
    # Compare both fields in constant time; AND the results so timing can't
    # reveal which of the two was wrong. Encode first: compare_digest rejects
    # non-ASCII *str* with a TypeError, so a username like "üser" — or a
    # configured password like "café" — would otherwise raise straight out of
    # the auth hook as a 500, and in the configured case lock everyone out.
    user_ok = hmac.compare_digest((username or "").encode("utf-8"), user.encode("utf-8"))
    pw_ok = hmac.compare_digest((password or "").encode("utf-8"), pw.encode("utf-8"))
    return user_ok and pw_ok
