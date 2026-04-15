import subprocess
import tempfile
import os


class LocalExecutor:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def execute(self, code: str, test_code: str) -> dict:
        full = f"{code.strip()}\n\n{test_code.strip()}\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full)
            path = f.name
        try:
            result = subprocess.run(
                ["python", path],
                capture_output=True, text=True, timeout=self.timeout
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": "TimeoutExpired"}
        finally:
            os.unlink(path)
