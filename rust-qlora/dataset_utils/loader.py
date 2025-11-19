import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, IterableDataset, load_dataset


class DatasetLoader:
    """
    Encapsulates dataset loading for cached and streaming modes.

    Responsibilities:
    - Handle HuggingFace cached datasets with filtering and pre-tokenization.
    - Handle multi-dataset interleaving via stream_rust.
    - Build streaming IterableDatasets with the correct worker overrides.
    - Surface metadata about the dataset for downstream logic.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        tokenizer,
        create_filter_function,
        stream_rust_fn,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.create_filter_function = create_filter_function
        self.stream_rust = stream_rust_fn
        self.logger = logger or logging.getLogger(__name__)

        self.dataset_cfg = cfg.get("dataset", {})
        self.train_cfg = cfg.get("train", {})
        self.max_seq_len = cfg["max_seq_len"]

    def load(self) -> Tuple[Any, Dict[str, Any]]:
        dataset_names = self.dataset_cfg.get(
            "names", self.cfg.get("dataset_name", "ammarnasr/the-stack-rust-clean")
        )
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        metadata: Dict[str, Any] = {
            "is_streaming": False,
            "pre_tokenized": False,
            "dataset_names": dataset_names,
        }

        use_cache = self.dataset_cfg.get("use_cache", True)
        if use_cache:
            if len(dataset_names) == 1:
                dataset = self._load_cached_single(dataset_names[0])
                metadata["pre_tokenized"] = True
            else:
                dataset = self._load_cached_multi(dataset_names)
        else:
            dataset = self._load_streaming(dataset_names)
            metadata["is_streaming"] = True

        return dataset, metadata

    # -------------------------------------------------------------------------
    # Cached dataset helpers
    # -------------------------------------------------------------------------
    def _load_cached_single(self, dataset_name: str):
        self.logger.info("Loading dataset in cached mode - using Dataset.filter() for multi-worker efficiency")
        self.logger.info(f"Loading dataset: {dataset_name}")

        cache_dir = self.dataset_cfg.get("cache_dir")
        cache_config = {"cache_dir": cache_dir} if cache_dir else {}

        dataset = load_dataset(dataset_name, split="train", streaming=False, **cache_config)
        filter_fn = self.create_filter_function(
            min_length=self.dataset_cfg.get("min_length", 64),
            max_length=self.dataset_cfg.get("max_length", 200_000),
            exclude_tests=self.dataset_cfg.get("exclude_tests", True),
            exclude_examples=self.dataset_cfg.get("exclude_examples", False),
            exclude_benches=self.dataset_cfg.get("exclude_benches", True),
            prefer_idiomatic=self.dataset_cfg.get("prefer_idiomatic", False),
            prefer_documented=self.dataset_cfg.get("prefer_documented", False),
            idiomatic_quality_ratio=self.dataset_cfg.get("idiomatic_quality_ratio", 2.0),
        )

        self.logger.info("Filtering dataset...")
        ds_filtered = dataset.filter(filter_fn, desc=f"Filtering {dataset_name}")
        ds_tokenized = self._pretokenize(ds_filtered)

        shuffle_seed = self.dataset_cfg.get("shuffle_seed")
        if shuffle_seed is not None:
            self.logger.info(f"Shuffling dataset with seed {shuffle_seed}")
            ds_tokenized = ds_tokenized.shuffle(seed=shuffle_seed)

        return ds_tokenized

    def _pretokenize(self, dataset: Dataset) -> Dataset:
        """Parallel pre-tokenization for cached datasets."""
        self.logger.info("Pre-tokenizing dataset with parallel processing...")
        num_proc = min(80, os.cpu_count() or 1)

        def tokenize_batch(examples: Dict[str, List[str]]):
            return self.tokenizer(
                examples["content"],
                truncation=True,
                max_length=self.max_seq_len,
                padding=False,
                return_overflowing_tokens=False,
            )

        ds_tokenized = dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=1000,
            remove_columns=[col for col in dataset.column_names if col != "content"],
            num_proc=num_proc,
            desc="Tokenizing dataset (parallel)",
        )

        columns_to_remove = [
            col for col in ds_tokenized.column_names if col not in ["input_ids", "attention_mask"]
        ]
        return ds_tokenized.remove_columns(columns_to_remove) if columns_to_remove else ds_tokenized

    def _load_cached_multi(self, dataset_names: List[str]):
        self.logger.info("Multiple datasets detected - using stream_rust for interleaving")
        all_items = list(
            self.stream_rust(
                dataset_names=dataset_names,
                cache_dir=self.dataset_cfg.get("cache_dir"),
                use_cache=True,
                min_length=self.dataset_cfg.get("min_length", 64),
                max_length=self.dataset_cfg.get("max_length", 200_000),
                exclude_tests=self.dataset_cfg.get("exclude_tests", True),
                exclude_examples=self.dataset_cfg.get("exclude_examples", False),
                exclude_benches=self.dataset_cfg.get("exclude_benches", True),
                prefer_idiomatic=self.dataset_cfg.get("prefer_idiomatic", False),
                prefer_documented=self.dataset_cfg.get("prefer_documented", False),
                idiomatic_quality_ratio=self.dataset_cfg.get("idiomatic_quality_ratio", 2.0),
                shuffle_seed=self.dataset_cfg.get("shuffle_seed"),
                interleave_mode=self.dataset_cfg.get("interleave_mode", "sequential"),
                dataset_weights=self.dataset_cfg.get("dataset_weights"),
            )
        )
        self.logger.info(f"Loaded {len(all_items)} filtered samples")
        dataset = Dataset.from_list(all_items)

        if self.train_cfg.get("dataloader_num_workers", 12) > 1:
            self.logger.warning(
                "Multiple datasets with interleaving - reducing workers to 1 (single-shard dataset)"
            )
            self.train_cfg["dataloader_num_workers"] = 1

        return dataset

    # -------------------------------------------------------------------------
    # Streaming dataset helpers
    # -------------------------------------------------------------------------
    def _load_streaming(self, dataset_names: List[str]):
        self.logger.info("Loading dataset in streaming mode - using IterableDataset with 0 workers")

        iterable = IterableDataset.from_generator(
            lambda: self.stream_rust(
                dataset_names=dataset_names,
                cache_dir=self.dataset_cfg.get("cache_dir"),
                use_cache=False,
                min_length=self.dataset_cfg.get("min_length", 64),
                max_length=self.dataset_cfg.get("max_length", 200_000),
                exclude_tests=self.dataset_cfg.get("exclude_tests", True),
                exclude_examples=self.dataset_cfg.get("exclude_examples", False),
                exclude_benches=self.dataset_cfg.get("exclude_benches", True),
                prefer_idiomatic=self.dataset_cfg.get("prefer_idiomatic", False),
                prefer_documented=self.dataset_cfg.get("prefer_documented", False),
                idiomatic_quality_ratio=self.dataset_cfg.get("idiomatic_quality_ratio", 2.0),
                shuffle_seed=self.dataset_cfg.get("shuffle_seed"),
                interleave_mode=self.dataset_cfg.get("interleave_mode", "sequential"),
                dataset_weights=self.dataset_cfg.get("dataset_weights"),
            )
        )

        if self.train_cfg.get("dataloader_num_workers", 2) > 0:
            self.logger.warning(
                "Streaming mode detected - setting dataloader_num_workers to 0 (workers don't work well with streaming)"
            )
            self.train_cfg["dataloader_num_workers"] = 0

        return iterable

