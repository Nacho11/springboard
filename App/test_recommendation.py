from flask_micro import app
import pytest
from unittest.mock import Mock

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_default(client):
    rv = client.post('/recommend')
    assert b'BIC Round Stic Xtra Life Ballpoint Pen' in rv.data
