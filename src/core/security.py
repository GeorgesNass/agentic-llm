'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Security layer: RBAC, permission checks, request helpers and JWT middleware for protected FastAPI routes."
'''

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from core.auth import (
    AuthUser,
    decode_token,
    is_token_blacklisted,
    payload_to_auth_user,
)

## ============================================================
## EXCEPTIONS
## ============================================================
class SecurityError(Exception):
    """
        Base security exception

        Args:
            message: Error message
    """

    def __init__(self, message: str = "Security error") -> None:
        ## Store error message
        self.message = message
        super().__init__(message)

class UnauthorizedError(SecurityError):
    """
        Error raised when authentication fails

        Args:
            message: Error message
    """

    def __init__(self, message: str = "Unauthorized access") -> None:
        ## Initialize unauthorized error
        super().__init__(message=message)

class ForbiddenError(SecurityError):
    """
        Error raised when authorization fails

        Args:
            message: Error message
    """

    def __init__(self, message: str = "Forbidden access") -> None:
        ## Initialize forbidden error
        super().__init__(message=message)

class TokenTypeError(SecurityError):
    """
        Error raised when a token type is invalid

        Args:
            message: Error message
    """

    def __init__(self, message: str = "Invalid token type") -> None:
        ## Initialize token type error
        super().__init__(message=message)

## ============================================================
## RBAC CONFIGURATION
## ============================================================
ROLE_PERMISSIONS: dict[str, list[str]] = {
    "admin": ["read", "write", "delete", "manage"],
    "manager": ["read", "write"],
    "editor": ["read", "write"],
    "user": ["read"],
    "viewer": ["read"],
    "service": ["read", "write"],
}

DEFAULT_PUBLIC_PATHS: list[str] = [
    "/", "/docs",  "/redoc", "/openapi.json",
    "/health", "/login", "/token", "/refresh",
]

## ============================================================
## PERMISSION HELPERS
## ============================================================
def get_user_permissions(user: AuthUser) -> set[str]:
    """
        Resolve the effective permissions of a user from their roles

        Args:
            user: Authenticated user

        Returns:
            Set of resolved permissions
    """
    
    permissions: set[str] = set()

    ## Aggregate permissions from every assigned role
    for role in user.roles:
        permissions.update(ROLE_PERMISSIONS.get(role, []))

    return permissions

def has_any_role(user: AuthUser, allowed_roles: list[str]) -> bool:
    """
        Check whether a user owns at least one allowed role

        Args:
            user: Authenticated user
            allowed_roles: List of accepted roles

        Returns:
            True if any role matches
    """

    return any(role in allowed_roles for role in user.roles)

def has_all_scopes(user: AuthUser, required_scopes: list[str]) -> bool:
    """
        Check whether a user owns all required scopes

        Args:
            user: Authenticated user
            required_scopes: Required scopes

        Returns:
            True if all scopes are present
    """

    return all(scope in user.scopes for scope in required_scopes)

def has_permission(user: AuthUser, required_permission: str) -> bool:
    """
        Check whether a user owns a required permission

        Args:
            user: Authenticated user
            required_permission: Permission to verify

        Returns:
            True if permission is granted
    """

    return required_permission in get_user_permissions(user)

## ============================================================
## ENFORCEMENT HELPERS
## ============================================================
def enforce_roles(user: AuthUser, allowed_roles: list[str]) -> None:
    """
        Enforce a role-based authorization rule

        Args:
            user: Authenticated user
            allowed_roles: Accepted roles

        Raises:
            HTTPException: If user does not own a valid role
    """

    if not has_any_role(user=user, allowed_roles=allowed_roles):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient role privileges",
        )

def enforce_permission(user: AuthUser, required_permission: str) -> None:
    """
        Enforce a permission-based authorization rule

        Args:
            user: Authenticated user
            required_permission: Permission to verify

        Raises:
            HTTPException: If permission is missing
    """

    if not has_permission(user=user, required_permission=required_permission):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )

def enforce_scopes(user: AuthUser, required_scopes: list[str]) -> None:
    """
        Enforce a scope-based authorization rule

        Args:
            user: Authenticated user
            required_scopes: Required scopes

        Raises:
            HTTPException: If one or more scopes are missing
    """
        
    if not has_all_scopes(user=user, required_scopes=required_scopes):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing required scopes",
        )

## ============================================================
## FASTAPI DEPENDENCIES
## ============================================================
def require_roles(allowed_roles: list[str]):
    """
        Build a FastAPI dependency enforcing role access

        Args:
            allowed_roles: Accepted roles

        Returns:
            FastAPI dependency
    """

    def dependency(request: Request) -> AuthUser:
        """
            Validate request user against allowed roles

            Args:
                request: Current FastAPI request

            Returns:
                Authenticated user

            Raises:
                HTTPException: If user is missing or role check fails
        """
        
        ## Extract user from request
        user = get_request_user(request=request)

        ## Enforce roles
        enforce_roles(user=user, allowed_roles=allowed_roles)

        return user

    return Depends(dependency)

def require_permission(required_permission: str):
    """
        Build a FastAPI dependency enforcing a permission check

        Args:
            required_permission: Required permission

        Returns:
            FastAPI dependency
    """

    def dependency(request: Request) -> AuthUser:
        """
            Validate request user against a permission

            Args:
                request: Current FastAPI request

            Returns:
                Authenticated user

            Raises:
                HTTPException: If user is missing or permission check fails
        """
        
        ## Extract user from request
        user = get_request_user(request=request)

        ## Enforce permission
        enforce_permission(user=user, required_permission=required_permission)

        return user

    return Depends(dependency)

def require_scopes(required_scopes: list[str]):
    """
        Build a FastAPI dependency enforcing required scopes

        Args:
            required_scopes: Required scopes

        Returns:
            FastAPI dependency
    """

    def dependency(request: Request) -> AuthUser:
        """
            Validate request user against required scopes

            Args:
                request: Current FastAPI request

            Returns:
                Authenticated user

            Raises:
                HTTPException: If user is missing or scope check fails
        """
        
        ## Extract user from request
        user = get_request_user(request=request)

        ## Enforce scopes
        enforce_scopes(user=user, required_scopes=required_scopes)

        return user

    return Depends(dependency)

## ============================================================
## REQUEST HELPERS
## ============================================================
def get_request_user(request: Request) -> AuthUser:
    """
        Retrieve the authenticated user attached to the current request

        Args:
            request: Current FastAPI request

        Returns:
            Authenticated user

        Raises:
            HTTPException: If no user is attached to request state
    """
    
    ## Retrieve user from request state
    user = getattr(request.state, "user", None)

    ## Ensure user exists
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not authenticated",
        )

    return user

def attach_anonymous_user(request: Request) -> None:
    """
        Attach an anonymous user placeholder

        Args:
            request: Current FastAPI request
    """
    
    request.state.user = None

def attach_authenticated_user(request: Request, user: AuthUser) -> None:
    """
        Attach authenticated user to request

        Args:
            request: Current FastAPI request
            user: Authenticated user
    """
    
    request.state.user = user

## ============================================================
## TOKEN HELPERS
## ============================================================
def extract_bearer_token(request: Request) -> str | None:
    """
        Extract a bearer token from Authorization header

        Args:
            request: Current FastAPI request

        Returns:
            Raw token or None if header is missing or invalid
    """
    
    ## Read header
    authorization_header = request.headers.get("Authorization")

    if not authorization_header:
        return None

    ## Validate Bearer format
    if not authorization_header.startswith("Bearer "):
        return None

    ## Extract token
    return authorization_header.removeprefix("Bearer ").strip()

def validate_access_token(token: str) -> AuthUser:
    """
        Validate access token and return user

        Args:
            token: Raw JWT access token

        Returns:
            Authenticated user

        Raises:
            HTTPException: If token is revoked, invalid or not an access token
    """
    
    ## Check blacklist
    if is_token_blacklisted(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token revoked",
        )

    payload = decode_token(token)

    ## Validate type
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token type",
        )

    ## Convert payload to user
    return payload_to_auth_user(payload=payload)

## ============================================================
## MIDDLEWARE
## ============================================================
class JWTMiddleware(BaseHTTPMiddleware):
    """
        Attach authenticated user to request.state

        Args:
            app: FastAPI application
            public_paths: Public routes excluded from JWT validation
            attach_user_on_public_routes: Whether to attach anonymous user
    """

    def __init__(
        self,
        app: Any,
        public_paths: list[str] | None = None,
        attach_user_on_public_routes: bool = True,
    ) -> None:
        ## Initialize parent middleware
        super().__init__(app)

        ## Initialize middleware configuration
        self.public_paths = public_paths or DEFAULT_PUBLIC_PATHS
        self.attach_user_on_public_routes = attach_user_on_public_routes

    async def dispatch(self, request: Request, call_next: Any) -> JSONResponse:
        """
            Process incoming request and attach authenticated user when valid

            Args:
                request: Current FastAPI request
                call_next: Next ASGI callable

            Returns:
                Downstream response or JSON auth error response
        """
        
        request_path = request.url.path

        ## Skip public routes
        if self._is_public_path(request_path=request_path):
            if self.attach_user_on_public_routes:
                attach_anonymous_user(request=request)
            return await call_next(request)

        token = extract_bearer_token(request=request)

        ## Reject missing token
        if token is None:
            return self._build_error_response(
                detail="Missing bearer token",
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        try:
            ## Validate token
            user = validate_access_token(token=token)

            ## Attach user
            attach_authenticated_user(request=request, user=user)

            return await call_next(request)

        except HTTPException as exc:
            return self._build_error_response(
                detail=str(exc.detail),
                status_code=exc.status_code,
            )

        except Exception:
            return self._build_error_response(
                detail="Authentication processing error",
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

    def _is_public_path(self, request_path: str) -> bool:
        """
            Check whether a request path is public

            Args:
                request_path: Current request path

            Returns:
                True if path is public
        """

        return any(
            request_path == path or request_path.startswith(path)
            for path in self.public_paths
        )

    @staticmethod
    def _build_error_response(detail: str, status_code: int) -> JSONResponse:
        """
            Build a JSON authentication error response

            Args:
                detail: Error detail message
                status_code: HTTP status code

            Returns:
                JSONResponse carrying the error payload
        """

        return JSONResponse(
            status_code=status_code,
            content={"detail": detail},
        )