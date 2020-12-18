from fastapi.testclient import TestClient
import pytest

from .main import app, load_model

client = TestClient(app)


def test_predict_correct():
    """
    test predict_label endpoint
    """

    # correct request 
    response = client.post(
        "/predict_label",
        json={"url": "https://www.hello.com", "title": "health style", "text": "health and style is important"},
    )
    assert response.status_code == 200
    assert response.json()['label'] == 'fashion_beauty_lifestyle'

    # bad request
    response = client.post(
        "/predict_label",
        data="hello",
    )
    assert response.status_code == 400
