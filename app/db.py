"""
db.py — Async SQLAlchemy 2.0 engine and session factory.

Connection pool tuning:
  pool_size=10        — keep 10 connections warm at all times
  max_overflow=20     — allow up to 20 extra connections under load burst
  pool_pre_ping=True  — validate connection before checkout (survives PG restart)
  pool_recycle=1800   — recycle connections every 30 min (avoids PG idle timeout)

Session options:
  expire_on_commit=False — attributes remain accessible after commit()
                           without triggering a lazy SELECT (safe in async code)
"""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=1800,
    echo=settings.ENV == "dev",   # log SQL only in dev
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ---------------------------------------------------------------------------
# Declarative base (shared by all ORM models)
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass
