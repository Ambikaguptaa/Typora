"""Access Control and Role-Based Authorization Module.

Implements lightweight role-based access control (RBAC) governing participant data,
research workflows, aggregate statistics, and administrative operations.
"""

from enum import Enum
from typing import Dict, Set


class Role(str, Enum):
    """User and system roles."""

    USER = "USER"
    RESEARCHER = "RESEARCHER"
    ADMIN = "ADMIN"


class Permission(str, Enum):
    """Granular system permissions."""

    VIEW_OWN_TYPING_DEVIATION = "VIEW_OWN_TYPING_DEVIATION"
    EXPORT_OWN_DATA = "EXPORT_OWN_DATA"
    TRIGGER_DATA_PURGE = "TRIGGER_DATA_PURGE"
    VIEW_AGGREGATE_METRICS = "VIEW_AGGREGATE_METRICS"
    TRAIN_RESEARCH_MODELS = "TRAIN_RESEARCH_MODELS"
    MANAGE_RETENTION_POLICIES = "MANAGE_RETENTION_POLICIES"
    VIEW_SECURITY_AUDIT_LOGS = "VIEW_SECURITY_AUDIT_LOGS"
    VIEW_RAW_KEYSTROKE_TEXT = "VIEW_RAW_KEYSTROKE_TEXT"  # Strictly Prohibited for all


class PermissionDeniedError(Exception):
    """Raised when an operation violates access control policy."""
    pass


# Permissions that are completely prohibited across all roles by system invariant
PROHIBITED_PERMISSIONS: Set[str] = {
    Permission.VIEW_RAW_KEYSTROKE_TEXT.value,
}

# Role permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[str]] = {
    Role.USER: {
        Permission.VIEW_OWN_TYPING_DEVIATION.value,
        Permission.EXPORT_OWN_DATA.value,
        Permission.TRIGGER_DATA_PURGE.value,
    },
    Role.RESEARCHER: {
        Permission.VIEW_OWN_TYPING_DEVIATION.value,
        Permission.EXPORT_OWN_DATA.value,
        Permission.VIEW_AGGREGATE_METRICS.value,
        Permission.TRAIN_RESEARCH_MODELS.value,
    },
    Role.ADMIN: {
        Permission.VIEW_OWN_TYPING_DEVIATION.value,
        Permission.EXPORT_OWN_DATA.value,
        Permission.TRIGGER_DATA_PURGE.value,
        Permission.VIEW_AGGREGATE_METRICS.value,
        Permission.TRAIN_RESEARCH_MODELS.value,
        Permission.MANAGE_RETENTION_POLICIES.value,
        Permission.VIEW_SECURITY_AUDIT_LOGS.value,
    },
}


class AccessController:
    """RBAC evaluation engine."""

    @staticmethod
    def has_permission(
        role: Role,
        permission: str,
        is_own_data: bool = True,
    ) -> bool:
        """Check whether a given role holds the requested permission.

        Args:
            role: The actor's role.
            permission: The permission string being requested.
            is_own_data: Flag indicating if the request targets the actor's own data.

        Returns:
            bool: True if authorized, False otherwise.
        """
        # Strict policy: Zero-raw-text invariant
        if permission in PROHIBITED_PERMISSIONS:
            return False

        if not is_own_data and role == Role.USER:
            # Users can never access other participants' data
            return False

        allowed = ROLE_PERMISSIONS.get(role, set())
        return permission in allowed

    @classmethod
    def enforce_permission(
        cls,
        role: Role,
        permission: str,
        is_own_data: bool = True,
    ) -> None:
        """Enforce permission, raising an exception if unauthorized.

        Args:
            role: The actor's role.
            permission: The requested permission.
            is_own_data: Whether data belongs to the actor.

        Raises:
            PermissionDeniedError: If unauthorized.
        """
        if not cls.has_permission(role, permission, is_own_data=is_own_data):
            raise PermissionDeniedError(
                f"Access Denied: Role '{role}' lacks permission '{permission}' "
                f"(is_own_data={is_own_data})."
            )
