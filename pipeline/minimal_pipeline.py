# Minimal evaluation pipeline
#
# Usage:
#   python -m pipeline.minimal_pipeline --config-name minimal_config
#
# Steps:
# 1. Load config
# 2. Load evaluation prompt set
# 3. Generate responses from the finetuned model
# 4. Generate responses from API baseline
# 5. Judge pairwise outputs
# 6. Save results to JSONL/CSV
# 7. Produce a summary table

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="minimal_config")
def main(cfg: DictConfig):
    pass


if __name__ == "__main__":
    main()
