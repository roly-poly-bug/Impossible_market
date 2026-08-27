from decimal import Decimal

from sqlalchemy import String
from sqlalchemy.types import TypeDecorator


class MoneyAmount(TypeDecorator[Decimal]):
    """Store an exact Decimal as text because SQLite has no native Decimal type."""

    impl = String(64)
    cache_ok = True

    def process_bind_param(self, value: Decimal | None, dialect: object) -> str | None:
        if value is None:
            return None
        amount = Decimal(value)
        if amount < 0:
            raise ValueError("price must not be negative")
        return format(amount, "f")

    def process_result_value(self, value: str | None, dialect: object) -> Decimal | None:
        return Decimal(value) if value is not None else None
