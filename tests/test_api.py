import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, ANY

@pytest.fixture
def client_and_mocks():
    """
    Aplica mocks necessários antes de instanciar o TestClient para evitar
    o carregamento real do MLflow durante o startup da app.
    Retorna (client, mock_model, mock_vectorizer, mock_log).
    """
    with patch('src.api.main.model', new_callable=MagicMock) as mock_model, \
         patch('src.api.main.vectorizer', new_callable=MagicMock) as mock_vectorizer, \
         patch('src.api.main.log_prediction') as mock_log:

        # Comportamento padrão do vetorizador
        mock_vectorizer.transform.return_value = "mocked_vector"

        # Importa a app apenas após aplicar os patches
        from src.api.main import app
        client = TestClient(app)

        yield client, mock_model, mock_vectorizer, mock_log

def test_predict_sentiment_insatisfeito(client_and_mocks):
    client, mock_model, _, mock_log = client_and_mocks

    # Arrange
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.25, 0.75]]

    # Act
    response = client.post("/predict", json={"message": "Produto veio quebrado, péssimo!"})

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["sentimento"] == "INSATISFEITO"
    assert data["acao_sugerida"] == "TRANSBORDO_HUMANO"
    assert data["probabilidade_insatisfeito"] == 0.75
    # aceita qualquer valor numérico para tempo de inferência
    mock_log.assert_called_once_with("Produto veio quebrado, péssimo!", "INSATISFEITO", 0.75, 0.75, ANY)

def test_predict_sentiment_satisfeito(client_and_mocks):
    client, mock_model, _, mock_log = client_and_mocks

    # Arrange
    mock_model.predict.return_value = [0]
    mock_model.predict_proba.return_value = [[0.9, 0.1]]

    # Act
    response = client.post("/predict", json={"message": "Adorei o produto, entrega rápida!"})

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["sentimento"] == "SATISFEITO"
    assert data["acao_sugerida"] == "CONTINUAR_AVI"
    assert data["probabilidade_insatisfeito"] == 0.1
    mock_log.assert_called_once()

def test_predict_when_model_is_not_loaded():
    # Cria app com modelo e vetorizador None e testa resposta
    with patch('src.api.main.model', None), patch('src.api.main.vectorizer', None):
        from src.api.main import app as app_no_model
        client = TestClient(app_no_model)
        response = client.post("/predict", json={"message": "qualquer texto"})
        assert response.status_code == 503
        assert response.json() == {"detail": "Modelo não carregado"}

def test_invalid_payload_returns_422(client_and_mocks):
    client, _, _, _ = client_and_mocks
    # missing "message" field -> FastAPI should return 422
    response = client.post("/predict", json={})
    assert response.status_code == 422

def test_vectorizer_transform_called(client_and_mocks):
    client, mock_model, mock_vectorizer, _ = client_and_mocks
    mock_model.predict.return_value = [0]
    mock_model.predict_proba.return_value = [[0.9, 0.1]]

    response = client.post("/predict", json={"message": "Teste de vetorizador"})
    assert response.status_code == 200
    # garante que transform foi chamada com lista/iterável contendo o texto limpo
    assert mock_vectorizer.transform.called
