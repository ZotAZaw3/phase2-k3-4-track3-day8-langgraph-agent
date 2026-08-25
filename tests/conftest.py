"""Load .env before test modules evaluate their API-key skip conditions."""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass
