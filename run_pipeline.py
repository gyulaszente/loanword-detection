from __future__ import annotations

from datetime import datetime
from pathlib import Path

import papermill as pm


ROOT = Path(__file__).resolve().parent

NOTEBOOKS = [
    ROOT / "uesz_extract" / "01_pdf_to_txt_to_json_to_csv.ipynb",
    ROOT / "emmorph_audit" / "02_emmorph_test.ipynb",
    ROOT / "classification" / "03_log_reg_classification.ipynb",
]

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_DIR = ROOT / "pipeline_runs" / RUN_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    for notebook in NOTEBOOKS:
        output_notebook = OUTPUT_DIR / notebook.name

        print(f"\nRunning: {notebook.relative_to(ROOT)}")
        print(f"Working directory: {notebook.parent.relative_to(ROOT)}")
        print(f"Saving copy to: {output_notebook.relative_to(ROOT)}")

        pm.execute_notebook(
            input_path=str(notebook),
            output_path=str(output_notebook),
            cwd=str(notebook.parent),
        )

    print(f"\nPipeline finished. Executed notebooks are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()