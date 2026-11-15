import os
import tempfile

os.environ["TMS_DATA"] = tempfile.mkdtemp(prefix="tms-test-")
os.environ["TMS_MIN_FREE_GB"] = "0"

import pytest

from tms import config, db


@pytest.fixture
def conn():
    for suffix in ("", "-wal", "-shm"):
        path = config.db_path().with_name(config.db_path().name + suffix)
        path.unlink(missing_ok=True)
    db.init()
    connection = db.connect()
    yield connection
    connection.close()
