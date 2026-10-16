import subprocess
import sys

from configs.mnist.clean_baseline_config import CONFIGS


def run_config(config_name: str) -> int:
    print(f"[CleanBaselineRunner] starting {config_name}", flush=True)
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--config",
            config_name,
        ],
    )

    if result.returncode == 0:
        print(f"[CleanBaselineRunner] finished {config_name}", flush=True)
    else:
        print(
            f"[CleanBaselineRunner] failed {config_name} "
            f"(returncode={result.returncode})",
            flush=True,
        )

    return result.returncode


def main() -> None:
    failed_configs = []

    for config_name in CONFIGS:
        returncode = run_config(config_name)
        if returncode != 0:
            failed_configs.append((config_name, returncode))

    if not failed_configs:
        print("[CleanBaselineRunner] all configs finished successfully", flush=True)
        return

    print("[CleanBaselineRunner] failed configs:", flush=True)
    for config_name, returncode in failed_configs:
        print(f"- {config_name} (returncode={returncode})", flush=True)


if __name__ == "__main__":
    main()
