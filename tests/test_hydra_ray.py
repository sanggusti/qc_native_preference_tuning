import logging
import time
import hydra
import ray

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

@ray.remote
def process_batch(batch_id: int, items:list[int], delay_s: float) -> dict:
    start = time.time()
    time.sleep(delay_s)  # Simulate work
    outputs = [x * x for x in items]
    return {
        "batch_id": batch_id,
        "inputs": items,
        "outputs": outputs,
        "processing_time": time.time() - start
    }

def chunked(values: list[int], size: int) -> list[list[int]]:
    return [values[i:i + size] for i in range(0, len(values), size)]


@hydra.main(version_base=None, config_path="../configs/test_configs", config_name="hydra_ray_sample")
def main(cfg: DictConfig):
    log.info(f"Effective config:\n%s", OmegaConf.to_yaml(cfg))

    batches = chunked(cfg.batch["items"], cfg.batch["batch_size"])
    log.info("Launching %d Ray tasks", len(batches))

    futures = [
        process_batch.options(num_cpus=cfg.ray_task["num_cpus"]).remote(
            batch_id=i,
            items=batch,
            delay_s=cfg.ray_task["artificial_delay"]
        )
        for i, batch in enumerate(batches)
    ]

    results = ray.get(futures)
    results = sorted(results, key=lambda row: row["batch_id"])
    flattened = [y for row in results for y in row["outputs"]]
    log.info("Results by batch: %s", results)
    log.info("Flattened results: %s", flattened)

if __name__ == "__main__":
    main()
