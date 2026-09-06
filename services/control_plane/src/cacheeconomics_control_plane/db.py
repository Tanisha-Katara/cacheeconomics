from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from .settings import Settings


def create_database_engine(settings: Settings) -> Engine:
    options = {"pool_pre_ping": True}
    if settings.database_url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
    return create_engine(settings.database_url, **options)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def set_identity_context(
    session: Session,
    *,
    user_id: Optional[uuid.UUID] = None,
    organization_id: Optional[uuid.UUID] = None,
) -> None:
    """Set transaction-local values read by PostgreSQL row-security policies."""

    session.info["current_user_id"] = user_id
    session.info["current_organization_id"] = organization_id
    if session.bind is None or session.bind.dialect.name != "postgresql":
        return

    if user_id is not None:
        session.execute(
            text("SELECT set_config('app.current_user_id', :value, true)"),
            {"value": str(user_id)},
        )
    if organization_id is not None:
        session.execute(
            text("SELECT set_config('app.current_organization_id', :value, true)"),
            {"value": str(organization_id)},
        )
