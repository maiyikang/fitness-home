from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


from core.paths import DATA_DIR


RAW_DATASET_FILE = DATA_DIR / "lora_training_data.jsonl"

FILTERED_DATASET_FILE = DATA_DIR / "fitness_home_lora_dataset_v2.jsonl"

REPORT_FILE = DATA_DIR / "dataset_filter_report.json"


MIN_REASON_LENGTH = 50

MAX_REASON_LENGTH = 1200


class DatasetFilter:

    def __init__(self) -> None:

        self.total_samples = 0

        self.valid_samples = 0

        self.invalid_samples = 0

        self.duplicate_samples = 0

        self.filtered_dataset: List[Dict[str, Any]] = []

        self.report: Dict[str, int] = {
            "missing_restaurant": 0,
            "missing_menu": 0,
            "missing_reason": 0,
            "invalid_reason_length": 0,
            "duplicate": 0,
            "invalid_json": 0,
        }

    def load_dataset(self) -> List[Dict[str, Any]]:

        dataset: List[Dict[str, Any]] = []

        with RAW_DATASET_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                if not line.strip():
                    continue

                try:

                    dataset.append(json.loads(line))

                except Exception:

                    self.report["invalid_json"] += 1

        return dataset

    def extract_restaurant(
        self,
        output: str,
    ) -> str:

        if "Recommended Restaurant:" not in output:
            return ""

        text = output.split(
            "Recommended Restaurant:",
            1,
        )[1]

        return text.splitlines()[1].strip()

    def extract_menu(
        self,
        output: str,
    ) -> str:

        if "Recommended Menu:" not in output:
            return ""

        text = output.split(
            "Recommended Menu:",
            1,
        )[1]

        return text.splitlines()[1].strip()

    def extract_reason(
        self,
        output: str,
    ) -> str:

        if "Recommended Reason:" not in output:
            return ""

        return output.split(
            "Recommended Reason:",
            1,
        )[1].strip()

    def is_valid(
        self,
        sample: Dict[str, Any],
        seen: set,
    ) -> bool:

        output = sample.get("output", "")

        restaurant = self.extract_restaurant(output)

        menu = self.extract_menu(output)

        reason = self.extract_reason(output)

        if restaurant == "":

            self.report["missing_restaurant"] += 1

            return False

        if menu == "":

            self.report["missing_menu"] += 1

            return False

        if reason == "":

            self.report["missing_reason"] += 1

            return False

        if len(reason) < MIN_REASON_LENGTH:

            self.report["invalid_reason_length"] += 1

            return False

        if len(reason) > MAX_REASON_LENGTH:

            self.report["invalid_reason_length"] += 1

            return False

        duplicate_key = (
            restaurant,
            menu,
            reason,
        )

        if duplicate_key in seen:

            self.report["duplicate"] += 1

            self.duplicate_samples += 1

            return False

        seen.add(duplicate_key)

        return True

    def filter_dataset(self) -> None:

        dataset = self.load_dataset()

        self.total_samples = len(dataset)

        seen = set()

        for sample in dataset:

            if self.is_valid(sample, seen):

                self.filtered_dataset.append(sample)

                self.valid_samples += 1

            else:

                self.invalid_samples += 1

    def save_dataset(self) -> None:

        with FILTERED_DATASET_FILE.open(
            "w",
            encoding="utf-8",
        ) as file:

            for sample in self.filtered_dataset:

                file.write(
                    json.dumps(
                        sample,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def save_report(self) -> None:

        report = {

            "total_samples": self.total_samples,

            "valid_samples": self.valid_samples,

            "invalid_samples": self.invalid_samples,

            "duplicate_samples": self.duplicate_samples,

            "details": self.report,

        }

        with REPORT_FILE.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False,
            )

    def run(self) -> None:

        print("=" * 60)
        print("Fitness Home Dataset Filter")
        print("=" * 60)

        self.filter_dataset()

        self.save_dataset()

        self.save_report()

        print()

        print(f"Total Samples      : {self.total_samples}")

        print(f"Valid Samples      : {self.valid_samples}")

        print(f"Invalid Samples    : {self.invalid_samples}")

        print(f"Duplicate Samples  : {self.duplicate_samples}")

        print()

        print(f"Filtered Dataset : {FILTERED_DATASET_FILE}")

        print(f"Filter Report    : {REPORT_FILE}")

        print("=" * 60)


def main() -> None:

    DatasetFilter().run()


if __name__ == "__main__":
    main()