import subprocess
import sys

EXPERIMENTS = [
    ("mamba_mpy", "readmission_30_days"),
    ("mamba2_mpy", "readmission_30_days"),
    ("mamba_mpy", "length_of_stay_prediction"),
    ("mamba2_mpy", "length_of_stay_prediction"),
    ("mamba_mpy", "mortality_prediction"),
    ("mamba2_mpy", "mortality_prediction"),
]


def run_all_experiments():
    for model, task in EXPERIMENTS:
        cmd = [sys.executable, "main.py", "train", model, task]
        print(f"Running {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Experiment {model}/{task} failed with exit code {exc.returncode}. Aborting remaining runs.")
            break


if __name__ == "__main__":
    run_all_experiments()
