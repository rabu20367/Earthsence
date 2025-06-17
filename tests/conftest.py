"""
Configuration and fixtures for testing the EarthSense application.
"""
import os
import tempfile
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import app, get_db
from src.models.base import Base

# Test database setup
TEST_DATABASE_URL = "sqlite:///./test_earthsense.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create test database tables
Base.metadata.create_all(bind=engine)

def override_get_db():
    """Override the get_db dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

# Override the database dependency in the FastAPI app
app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module")
def test_client():
    ""
    Create a test client for the FastAPI application.
    """
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="function")
def db_session():
    ""
    Create a new database session for each test case.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(scope="module")
def temp_upload_dir():
    ""
    Create a temporary directory for testing file uploads.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        old_upload_dir = app.state.UPLOAD_DIR
        app.state.UPLOAD_DIR = temp_dir
        yield temp_dir
        app.state.UPLOAD_DIR = old_upload_dir
