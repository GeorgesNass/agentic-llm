'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "JWT authentication core: password hashing, token generation, token decoding, refresh flow and FastAPI dependencies."
'''

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext

## ============================================================
## CONFIGURATION
## ============================================================
SECRET_KEY: str = "CHANGE_ME"
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

## ============================================================
## SECURITY PRIMITIVES
## ============================================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

## ============================================================
## DATA MODELS
## ============================================================
@dataclass(slots=True)
class AuthUser:
    """
        Minimal authenticated user payload

        Args:
            username: Unique user identifier
            roles: User roles
            scopes: OAuth scopes
            is_active: Active flag
    """
    
    username: str
    roles: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    is_active: bool = True

## ============================================================
## TOKEN STORAGE
## ============================================================
BLACKLISTED_TOKENS: set[str] = set()
USED_REFRESH_TOKENS: set[str] = set()

## ============================================================
## PASSWORD HELPERS
## ============================================================
def hash_password(password: str) -> str:
    """
        Hash password using bcrypt

        Args:
            password: Raw password

        Returns:
            Hashed password
    """

    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
        Verify password against hash

        Args:
            plain_password: Raw password
            hashed_password: Stored hash

        Returns:
            True if valid
    """

    return pwd_context.verify(plain_password, hashed_password)

## ============================================================
## TOKEN HELPERS
## ============================================================
def _now_utc() -> datetime:
    """
        Get current UTC datetime

        Returns:
            UTC datetime
    """

    return datetime.now(timezone.utc)

def _build_token_payload(
    subject: str,
    token_type: str,
    expires_delta: timedelta,
    roles: list[str] | None = None,
    scopes: list[str] | None = None,
) -> dict[str, Any]:
    """
        Build JWT payload

        Args:
            subject: User identifier
            token_type: access or refresh
            expires_delta: Expiration delta

        Returns:
            Payload dict
    """
    
    ## Compute timestamps
    issued_at = _now_utc()
    expires_at = issued_at + expires_delta

    return {
        "sub": subject,
        "roles": roles or [],
        "scopes": scopes or [],
        "type": token_type,
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
    }

def _encode_token(payload: dict[str, Any]) -> str:
    """
        Encode JWT payload

        Args:
            payload: JWT payload

        Returns:
            Token string
    """

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def create_access_token(
    subject: str,
    roles: list[str] | None = None,
    scopes: list[str] | None = None,
) -> str:
    """
        Create access token

        Args:
            subject: User identifier

        Returns:
            Access token
    """

    payload = _build_token_payload(
        subject=subject,
        token_type="access",
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        roles=roles,
        scopes=scopes,
    )

    return _encode_token(payload)

def create_refresh_token(subject: str) -> str:
    """
        Create refresh token

        Args:
            subject: User identifier

        Returns:
            Refresh token
    """

    payload = _build_token_payload(
        subject=subject,
        token_type="refresh",
        expires_delta=timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES),
    )

    return _encode_token(payload)

def decode_token(token: str) -> dict[str, Any]:
    """
        Decode and validate JWT token

        Args:
            token: JWT token

        Returns:
            Payload

        Raises:
            HTTPException: Invalid or expired token
    """
    
    try:
        ## Decode token
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        ) from exc

    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        ) from exc

def blacklist_token(token: str) -> None:
    """
        Add token to blacklist

        Args:
            token: JWT token
    """

    BLACKLISTED_TOKENS.add(token)

def is_token_blacklisted(token: str) -> bool:
    """
        Check if token is revoked

        Args:
            token: JWT token

        Returns:
            True if blacklisted
    """

    return token in BLACKLISTED_TOKENS

def mark_refresh_token_as_used(refresh_token: str) -> None:
    """
        Mark refresh token as used

        Args:
            refresh_token: Refresh token
    """
    
    USED_REFRESH_TOKENS.add(refresh_token)

def is_refresh_token_reused(refresh_token: str) -> bool:
    """
        Check refresh token reuse

        Args:
            refresh_token: Refresh token

        Returns:
            True if reused
    """
    
    return refresh_token in USED_REFRESH_TOKENS

## ============================================================
## USER SERIALIZATION
## ============================================================
def payload_to_auth_user(payload: dict[str, Any]) -> AuthUser:
    """
        Convert payload to AuthUser

        Args:
            payload: JWT payload

        Returns:
            AuthUser

        Raises:
            HTTPException: Missing subject
    """
    
    ## Extract subject
    subject = payload.get("sub")

    if not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    ## Build user
    return AuthUser(
        username=subject,
        roles=payload.get("roles", []),
        scopes=payload.get("scopes", []),
        is_active=payload.get("is_active", True),
    )

## ============================================================
## AUTH FLOW
## ============================================================
def authenticate_user(
    username: str,
    password: str,
    user_db: dict[str, dict[str, Any]],
) -> AuthUser | None:
    """
        Authenticate user

        Args:
            username: Username
            password: Password
            user_db: User store

        Returns:
            AuthUser or None
    """

    user_record = user_db.get(username)

    if not user_record:
        return None

    ## Verify password
    if not verify_password(password, user_record.get("hashed_password", "")):
        return None

    return AuthUser(
        username=user_record["username"],
        roles=user_record.get("roles", []),
        scopes=user_record.get("scopes", []),
        is_active=user_record.get("is_active", True),
    )

def login_user(
    username: str,
    password: str,
    user_db: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """
        Login and generate tokens

        Args:
            username: Username
            password: Password

        Returns:
            Token dict

        Raises:
            HTTPException: Invalid credentials
    """
    
    ## Authenticate
    user = authenticate_user(username, password, user_db)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    ## Create tokens
    access_token = create_access_token(user.username, user.roles, user.scopes)
    refresh_token = create_refresh_token(user.username)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }

def refresh_access_token(refresh_token: str) -> dict[str, str]:
    """
        Refresh token flow

        Args:
            refresh_token: Refresh token

        Returns:
            New tokens

        Raises:
            HTTPException: Invalid token
    """
    
    ## Check blacklist
    if is_token_blacklisted(refresh_token):
        raise HTTPException(status_code=401, detail="Revoked token")

    ## Check reuse
    if is_refresh_token_reused(refresh_token):
        raise HTTPException(status_code=401, detail="Token reuse detected")

    payload = decode_token(refresh_token)

    ## Validate type
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")

    subject = payload.get("sub")

    if not subject:
        raise HTTPException(status_code=401, detail="Invalid payload")

    ## Revoke old token
    blacklist_token(refresh_token)
    mark_refresh_token_as_used(refresh_token)

    ## Create new tokens
    return {
        "access_token": create_access_token(subject),
        "refresh_token": create_refresh_token(subject),
        "token_type": "bearer",
    }

def logout_user(token: str) -> None:
    """
        Logout user by revoking token

        Args:
            token: JWT token
    """
    
    ## Blacklist token
    blacklist_token(token)

## ============================================================
## FASTAPI DEPENDENCIES
## ============================================================
def get_current_user(token: str = Depends(oauth2_scheme)) -> AuthUser:
    """
        Get current user from token

        Args:
            token: Bearer token

        Returns:
            AuthUser

        Raises:
            HTTPException: Invalid token
    """
    
    ## Check blacklist
    if is_token_blacklisted(token):
        raise HTTPException(status_code=401, detail="Revoked token")

    payload = decode_token(token)

    ## Validate type
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")

    return payload_to_auth_user(payload)

def get_current_active_user(
    current_user: AuthUser = Depends(get_current_user),
) -> AuthUser:
    """
        Ensure user is active

        Args:
            current_user: AuthUser

        Returns:
            AuthUser

        Raises:
            HTTPException: Inactive user
    """
    ## Check active flag
    if not current_user.is_active:
        raise HTTPException(status_code=403, detail="Inactive user")

    return current_user