from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_medicalner_qwen3.sh"
CONFIG_DIR = ROOT / "configs/llamafactory"


def read_scalar_yaml(path: Path) -> dict[str, str]:
    """Parse the flat scalar keys used by the repository's training YAML files."""

    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


class MedicalNERRunnerProfileTests(unittest.TestCase):
    def dry_run(
        self, *args: str, config_env: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        if config_env is None:
            env.pop("CONFIG_YAML", None)
        else:
            env["CONFIG_YAML"] = config_env
        return subprocess.run(
            ["bash", str(RUNNER), *args, "--dry-run"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_schema_v2_routes_all_backend_and_task_combinations(self) -> None:
        cases = [
            (
                "cuda",
                "train",
                "scripts/run_medicalner_qwen3_pro858.sh",
                "configs/llamafactory/qwen3_8b_lora_schema_v2.yaml",
            ),
            (
                "cuda",
                "smoke",
                "scripts/run_medicalner_qwen3_smoke.sh",
                "configs/llamafactory/qwen3_8b_lora_schema_v2_smoke.yaml",
            ),
            (
                "npu",
                "train",
                "scripts/run_medicalner_qwen3_pro858_npu.sh",
                "configs/llamafactory/qwen3_8b_lora_schema_v2_npu.yaml",
            ),
            (
                "npu",
                "smoke",
                "scripts/run_medicalner_qwen3_smoke.sh",
                "configs/llamafactory/qwen3_8b_lora_schema_v2_smoke_npu.yaml",
            ),
        ]
        for backend, task, target_script, config in cases:
            with self.subTest(backend=backend, task=task):
                result = self.dry_run(
                    "--backend",
                    backend,
                    "--task",
                    task,
                    "--data-profile",
                    "schema-v2",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"backend    : {backend}", result.stdout)
                self.assertIn("data profile: schema-v2", result.stdout)
                self.assertIn(f"script     : {target_script}", result.stdout)
                self.assertIn(f"config     : {config}", result.stdout)

    def test_legacy_remains_the_default_and_does_not_force_a_config(self) -> None:
        result = self.dry_run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("backend    : cuda", result.stdout)
        self.assertIn("task       : train", result.stdout)
        self.assertIn("data profile: legacy", result.stdout)
        self.assertIn(
            "script     : scripts/run_medicalner_qwen3_pro858.sh", result.stdout
        )
        self.assertNotIn("config     :", result.stdout)

    def test_user_config_overrides_environment_and_profile_selection(self) -> None:
        result = self.dry_run(
            "--backend",
            "npu",
            "--task",
            "smoke",
            "--data-profile",
            "schema-v2",
            "--config",
            "configs/custom-user.yaml",
            config_env="configs/from-environment.yaml",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("config     : configs/custom-user.yaml", result.stdout)
        self.assertNotIn("configs/from-environment.yaml", result.stdout)
        self.assertNotIn("qwen3_8b_lora_schema_v2_smoke_npu.yaml", result.stdout)

    def test_invalid_data_profile_is_rejected(self) -> None:
        result = self.dry_run("--data-profile", "unknown")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unsupported data profile: unknown", result.stderr)

    def test_schema_v2_yaml_hardware_contracts(self) -> None:
        cuda_full = read_scalar_yaml(CONFIG_DIR / "qwen3_8b_lora_schema_v2.yaml")
        cuda_smoke = read_scalar_yaml(
            CONFIG_DIR / "qwen3_8b_lora_schema_v2_smoke.yaml"
        )
        npu_full = read_scalar_yaml(CONFIG_DIR / "qwen3_8b_lora_schema_v2_npu.yaml")
        npu_smoke = read_scalar_yaml(
            CONFIG_DIR / "qwen3_8b_lora_schema_v2_smoke_npu.yaml"
        )

        for name, config in (
            ("cuda_full", cuda_full),
            ("cuda_smoke", cuda_smoke),
            ("npu_full", npu_full),
            ("npu_smoke", npu_smoke),
        ):
            with self.subTest(config=name):
                self.assertEqual(config["dataset"], "medicalner_schema_v2_train")
                self.assertEqual(config["cutoff_len"], "16384")
                self.assertEqual(config["val_size"], "0")
                self.assertEqual(config["finetuning_type"], "lora")
                self.assertEqual(config["bf16"], "true")

        for config in (cuda_full, cuda_smoke):
            self.assertEqual(config["quantization_bit"], "4")
            self.assertEqual(config["quantization_method"], "bitsandbytes")
            self.assertEqual(config["flash_attn"], "auto")

        for config in (npu_full, npu_smoke):
            self.assertNotIn("quantization_bit", config)
            self.assertNotIn("quantization_method", config)
            self.assertEqual(config["flash_attn"], "auto")

        for config in (cuda_full, npu_full):
            self.assertEqual(
                config["eval_dataset"], "medicalner_schema_v2_validation"
            )
            self.assertEqual(config["eval_strategy"], "steps")
            self.assertEqual(config["load_best_model_at_end"], "true")

        for config in (cuda_smoke, npu_smoke):
            self.assertEqual(config["max_samples"], "10")
            self.assertEqual(config["eval_strategy"], "no")


if __name__ == "__main__":
    unittest.main()
