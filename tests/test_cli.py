import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def _run_completed(
        self,
        *args: str,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        return subprocess.run(
            [sys.executable, "-m", "hidetext.cli", *args],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            input=input_text,
        )

    def _run(self, *args: str) -> dict[str, object]:
        completed = self._run_completed(*args)
        return json.loads(completed.stdout)

    def test_encode_then_decode(self) -> None:
        prompt = "Write a calm and readable English paragraph."
        passphrase = "cli-pass"
        message = "CLI roundtrip works."

        encoded = self._run_completed(
            "encode",
            "--prompt",
            prompt,
            "--passphrase",
            passphrase,
            "--message",
            message,
            "--seed",
            "11",
        )
        self.assertTrue(encoded.stdout)
        self.assertFalse(encoded.stdout.startswith("{"))
        decoded = self._run_completed(
            "decode",
            "--prompt",
            prompt,
            "--passphrase",
            passphrase,
            "--seed",
            "11",
            input_text=encoded.stdout,
        )
        self.assertEqual(decoded.stdout, message)

    def test_encode_then_decode_with_files(self) -> None:
        prompt = "请写一段温柔自然的中文短文。"
        passphrase = "cli-file-pass"
        message = "文件参数也能正确往返。"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prompt_path = tmp_path / "prompt.txt"
            passphrase_path = tmp_path / "passphrase.txt"
            seed_path = tmp_path / "seed.txt"

            prompt_path.write_text(prompt + "\n", encoding="utf-8")
            passphrase_path.write_text(passphrase + "\n", encoding="utf-8")
            seed_path.write_text("19\n", encoding="utf-8")

            encoded = self._run(
                "encode",
                "--json",
                "--prompt-file",
                str(prompt_path),
                "--passphrase-file",
                str(passphrase_path),
                "--message",
                message,
                "--seed-file",
                str(seed_path),
            )
            decoded = self._run(
                "decode",
                "--json",
                "--prompt-file",
                str(prompt_path),
                "--passphrase-file",
                str(passphrase_path),
                "--text",
                str(encoded["text"]),
                "--seed-file",
                str(seed_path),
            )
            self.assertEqual(decoded["plaintext"], message)
            self.assertIn("tail_tokens", encoded)
            self.assertIn("trailing_tokens", decoded)

    def test_encode_and_decode_json_flag_preserves_detailed_output(self) -> None:
        prompt = "Write a calm and readable English paragraph."
        passphrase = "cli-json-pass"
        message = "CLI json mode works."

        encoded = self._run(
            "encode",
            "--json",
            "--prompt",
            prompt,
            "--passphrase",
            passphrase,
            "--message",
            message,
            "--seed",
            "17",
        )
        decoded = self._run(
            "decode",
            "--json",
            "--prompt",
            prompt,
            "--passphrase",
            passphrase,
            "--text",
            str(encoded["text"]),
            "--seed",
            "17",
        )

        self.assertEqual(decoded["plaintext"], message)
        self.assertIn("packet_tokens", encoded)
        self.assertIn("tail_tokens", encoded)
        self.assertIn("trailing_tokens", decoded)

    def test_eval_progress_goes_to_stderr(self) -> None:
        completed = self._run_completed(
            "eval",
            "--prompt",
            "请写一段温柔自然的中文短文。",
            "--passphrase",
            "progress-pass",
            "--message",
            "进度日志可见",
            "--seed",
            "13",
            "--show-progress",
            "--progress-token-interval",
            "400",
        )
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["roundtrip_ok"])
        self.assertIn("tail_tokens", payload)
        self.assertIn("decode_trailing_tokens", payload)
        self.assertIn("[encode][header]", completed.stderr)
        self.assertIn("bits=", completed.stderr)
        self.assertIn("tps=", completed.stderr)
        self.assertIn("bpt=", completed.stderr)

    def test_stall_patience_argument_is_accepted(self) -> None:
        payload = self._run(
            "encode",
            "--json",
            "--prompt",
            "Write a calm and readable English paragraph.",
            "--passphrase",
            "cli-pass",
            "--message",
            "CLI roundtrip works.",
            "--seed",
            "11",
            "--natural-tail-max-tokens",
            "8",
            "--stall-patience-tokens",
            "64",
            "--low-entropy-window-tokens",
            "16",
            "--low-entropy-threshold-bits",
            "0.1",
            "--max-encode-attempts",
            "3",
        )
        self.assertIn("text", payload)
        self.assertIn("attempts_used", payload)
        self.assertIn("tail_tokens", payload)


if __name__ == "__main__":
    unittest.main()
