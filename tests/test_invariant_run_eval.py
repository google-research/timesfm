import pytest
import subprocess
import sys
import os


@pytest.mark.parametrize("payload", [
    "__import__('os').system('echo PWNED')",
    "__import__('subprocess').call(['id'])",
    "open('/etc/passwd').read()",
])
def test_eval_does_not_execute_arbitrary_code(payload):
    """Invariant: eval() usage in run_eval.py must not allow arbitrary code execution from external input."""
    # The run_eval.py file uses eval() which could be dangerous if fed external input.
    # We verify that passing adversarial strings as arguments does not result in code execution.
    script_path = os.path.join(
        os.path.dirname(__file__),
        "v1", "experiments", "long_horizon_benchmarks", "run_eval.py"
    )
    
    # If the script doesn't exist at relative path, try from repo root
    if not os.path.exists(script_path):
        script_path = os.path.join(
            "v1", "experiments", "long_horizon_benchmarks", "run_eval.py"
        )
    
    # Run the script with adversarial payload as an argument
    # The script should either reject the input, fail gracefully, or not execute the payload
    result = subprocess.run(
        [sys.executable, script_path, "--exp_name", payload],
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    )
    
    # The payload should never successfully execute - "PWNED" should not appear in output
    assert "PWNED" not in result.stdout, "Arbitrary code was executed via eval()"
    assert "PWNED" not in result.stderr, "Arbitrary code was executed via eval()"
    # Ensure no evidence of /etc/passwd content leaking
    assert "root:" not in result.stdout, "File read was executed via eval()"
    assert "root:" not in result.stderr, "File read was executed via eval()"