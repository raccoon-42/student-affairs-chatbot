"""Google Sign-In: verify ID tokens, keep signed-in sessions.

The frontend obtains an ID token from Google Identity Services and posts
it here; we verify it against Google's public keys and issue our own
opaque session token as an HttpOnly cookie. Anyone with a Google account
can sign in — an @iyte.edu.tr address (any subdomain) marks the user as
a member, everyone else is a visitor.

Users and session tokens are persisted in SQLite (app/storage.py), so
signing in survives server restarts.
"""
import uuid

from app import storage
from config import settings

MEMBER_DOMAINS = ("iyte.edu.tr",)

# "aday" = prospective student — the audience the public bot exists for
EDUCATION_TYPES = ("aday", "lisans", "yukseklisans", "doktora")


def is_member(email: str) -> bool:
    domain = email.rsplit("@", 1)[-1].lower()
    return any(domain == d or domain.endswith("." + d) for d in MEMBER_DOMAINS)


def verify_google_token(credential: str) -> dict:
    """Raises ValueError when the token is invalid, expired, or not ours."""
    from google.oauth2 import id_token
    from google.auth.transport import requests as google_requests

    claims = id_token.verify_oauth2_token(
        credential, google_requests.Request(), settings.GOOGLE_CLIENT_ID
    )
    return {
        "email": claims["email"],
        "name": claims.get("name", ""),
        "picture": claims.get("picture", ""),
        "member": is_member(claims["email"]),
    }


class SessionStore:
    """Thin wrapper over storage so callers never touch SQL."""

    def create(self, user: dict) -> str:
        storage.upsert_user(user)
        token = uuid.uuid4().hex
        storage.create_auth_session(token, user["email"])
        return token

    def get(self, token):
        return storage.get_auth_user(token)

    def drop(self, token):
        storage.drop_auth_session(token)


sessions = SessionStore()
