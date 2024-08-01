import pytest
from engine.engine import Engine


@pytest.fixture()
def engine_inst() -> Engine:
    return Engine()
