from tempfile import TemporaryDirectory

from galahad.client.api_client import GalahadClient
import pytest
from pathlib import Path
import uvicorn

from galahad.server import GalahadServer


@pytest.fixture
def server():
    tmp = TemporaryDirectory()

    global tmpdir
    tmpdir = Path(tmp.name)

    host = "127.0.0.1"
    port = 8000

    global address
    address = "https://" + host + "/" + str(port)

    server = GalahadServer(data_dir=tmpdir)
    uvicorn.run(server, host=host, port=port)

    yield server
    tmp.cleanup()


@pytest.fixture
def client(server):
    # peter = server.state.data_dir
    yield GalahadClient(address)


def test_creation(client: GalahadClient):
    assert client.endpoint_url == address


def test_is_connected(client: GalahadClient):
    assert client.is_connected()[0]
