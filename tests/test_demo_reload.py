import os
import signal
import subprocess
import sys
import time


def test_reloader_restarts_for_sources_and_stops_child(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo.py").write_text("# fixture\n")
    source = tmp_path / "src" / "example.py"
    source.write_text("first\n")
    child_script = tmp_path / "child.py"
    child_script.write_text(
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "with Path(sys.argv[1]).open('a') as stream:\n"
        "    stream.write(str(os.getpid()) + '\\n')\n"
        "    stream.flush()\n"
        "while True: time.sleep(1)\n"
    )
    events = tmp_path / "started.txt"
    wrapper = (
        "import sys; from pathlib import Path; from demo import run_reloader; "
        "root=Path(sys.argv[1]); "
        "run_reloader([sys.executable, str(root/'child.py'), str(root/'started.txt')], root)"
    )
    parent = subprocess.Popen([sys.executable, "-c", wrapper, str(tmp_path)])

    def wait_for_starts(count):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if parent.poll() is not None:
                raise AssertionError(f"Reloader exited early: {parent.returncode}")
            lines = events.read_text().splitlines() if events.exists() else []
            if len(lines) >= count:
                return [int(line) for line in lines]
            time.sleep(0.05)
        raise AssertionError(f"Timed out waiting for {count} server starts")

    try:
        first = wait_for_starts(1)[0]
        source.write_text("second version\n")
        second = wait_for_starts(2)[1]
        assert second != first
        (tmp_path / "demo" / "new.js").write_text("console.log('updated');\n")
        third = wait_for_starts(3)[2]
        assert len({first, second, third}) == 3
        (tmp_path / "checkpoint.pt").write_text("changed checkpoint")
        time.sleep(2.2)
        assert len(events.read_text().splitlines()) == 3
    finally:
        parent.send_signal(signal.SIGTERM)
        parent.wait(timeout=15)
    assert parent.returncode == 0
    try:
        os.kill(third, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("Reloader left its server process running")
