import subprocess
import sys

from configs.mnist.clean_baseline_config import CONFIGS as CLEAN_BASELINE_CONFIGS
from configs.mnist.other_attacks_config import CONFIGS as OTHER_ATTACK_CONFIGS
from configs.mnist.pca_sign_config import CONFIGS as PCA_SIGN_CONFIGS


def run_config(config_name: str) -> int:
    print(f"[Runner] starting {config_name}", flush=True)
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--config",
            config_name,
        ],
    )
    if result.returncode == 0:
        print(f"[Runner] finished {config_name}", flush=True)
    else:
        print(
            f"[Runner] failed {config_name} " f"(returncode={result.returncode})",
            flush=True,
        )

    return result.returncode


def main() -> None:
    pca_config_names = list(PCA_SIGN_CONFIGS)
    clean_baseline_config_names = list(CLEAN_BASELINE_CONFIGS)
    other_attack_config_names = list(OTHER_ATTACK_CONFIGS)

    failed_configs = []

    # for config_name in clean_baseline_config_names:
    #     returncode = run_config(config_name)
    #     if returncode != 0:
    #         failed_configs.append((config_name, returncode))

    for config_name in pca_config_names:
        returncode = run_config(config_name)
        if returncode != 0:
            failed_configs.append((config_name, returncode))

    # for config_name in other_attack_config_names:
    #     returncode = run_config(config_name)
    #     if returncode != 0:
    #         failed_configs.append((config_name, returncode))

    if not failed_configs:
        print("[Runner] all configs finished successfully", flush=True)
        return

    print("[Runner] failed configs:", flush=True)
    for config_name, returncode in failed_configs:
        print(f"- {config_name} (returncode={returncode})", flush=True)


if __name__ == "__main__":
    main()
