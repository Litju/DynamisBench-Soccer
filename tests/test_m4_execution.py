import os
import socket
from pathlib import Path

from dynamisbench_soccer.m4.runner import execute_payload


def test_payload_is_bounded_and_workspace_is_cleaned(tmp_path: Path) -> None:
    private = tmp_path / "private.txt"
    private.write_text("host-only")
    with private.open() as stream, socket.socket(socket.AF_UNIX) as server:
        os.set_inheritable(stream.fileno(), True)
        address = str(tmp_path / "host.sock")
        server.bind(address)
        server.listen()
        source = f"""
import os
import socket
import sys
assert sys.stdin.buffer.read() == b'input'
try:
    os.fstat({stream.fileno()})
except OSError:
    pass
else:
    raise AssertionError('extra descriptor visible')
with socket.socket(socket.AF_UNIX) as client:
    try:
        client.connect({address!r})
    except OSError:
        pass
    else:
        raise AssertionError('host socket visible')
print('ok')
"""
        attempt = execute_payload(b"input", source.encode())
    assert attempt.outcome == "graded"
    assert attempt.stdout == b"ok\n"
    assert attempt.stdout_bytes == len(attempt.stdout)
    assert attempt.wall_seconds > 0
    assert attempt.workspace_removed


def test_failed_execution_cleans_workspace() -> None:
    attempt = execute_payload(b"", b"raise RuntimeError")
    assert attempt.outcome == "execution_error"
    assert attempt.workspace_removed
