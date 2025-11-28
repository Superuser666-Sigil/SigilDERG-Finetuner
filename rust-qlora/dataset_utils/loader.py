"""
Dataset loader for cached and streaming datasets.

Handles HuggingFace dataset loading, filtering, and multi-dataset interleaving.

Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Version: 2.9.0
"""

import logging
import os
from typing import Any

from datasets import Dataset, IterableDataset, load_dataset

from .jsonl_loader import load_prompt_gen_jsonl


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
        cfg: dict[str, Any],
        tokenizer,
        create_filter_function,
        stream_rust_fn,
        logger: logging.Logger | None = None,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.create_filter_function = create_filter_function
        self.stream_rust = stream_rust_fn
        self.logger = logger or logging.getLogger(__name__)

        self.dataset_cfg = cfg.get("dataset", {})
        self.train_cfg = cfg.get("train", {})
        self.max_seq_len = cfg["max_seq_len"]

    def load(self) -> tuple[Any, dict[str, Any]]:
        dataset_names = self.dataset_cfg.get(
            "names", self.cfg.get("dataset_name", "ammarnasr/the-stack-rust-clean")
        )
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        metadata: dict[str, Any] = {
            "is_streaming": False,
            "pre_tokenized": False,
            "dataset_names": dataset_names,
        }

        # Check if any datasets are local JSONL files
        local_jsonl_files = [name for name in dataset_names if name.startswith("local:")]
        hf_datasets = [
            name
            for name in dataset_names
            if not name.startswith("local:") and not name.startswith("parquet:")
        ]
        parquet_files = [name for name in dataset_names if name.startswith("parquet:")]

        # Handle mixed datasets (local JSONL + HuggingFace)
        if local_jsonl_files or parquet_files:
            dataset = self._load_mixed_datasets(local_jsonl_files, parquet_files, hf_datasets)
            metadata["is_streaming"] = not self.dataset_cfg.get("use_cache", True)
        else:
            # Original HuggingFace-only path
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
        self.logger.info(
            "Loading dataset in cached mode - using Dataset.filter() for multi-worker efficiency"
        )
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

        def tokenize_batch(examples: dict[str, list[str]]):
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

    def _load_cached_multi(self, dataset_names: list[str]):
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
    def _load_streaming(self, dataset_names: list[str]):
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

    # -------------------------------------------------------------------------
    # Local JSONL and Parquet dataset helpers
    # -------------------------------------------------------------------------
    def _load_mixed_datasets(
        self,
        local_jsonl_files: list[str],
        parquet_files: list[str],
        hf_datasets: list[str],
    ):
        """
        Load mixed datasets: local JSONL files, Parquet files, and HuggingFace datasets.

        Args:
            local_jsonl_files: List of "local:path/to.jsonl" paths
            parquet_files: List of "parquet:path/to.parquet" paths
            hf_datasets: List of HuggingFace dataset names
        """
        all_generators = []

        # Load local JSONL files
        for jsonl_spec in local_jsonl_files:
            jsonl_path = jsonl_spec.replace("local:", "", 1)
            self.logger.info(f"Loading local JSONL: {jsonl_path}")
            generator = load_prompt_gen_jsonl(
                jsonl_path=jsonl_path,
                tokenizer=self.tokenizer,
                apply_chat_template=False,  # Will be applied during tokenization
                remove_metadata=True,
                task_weights=self.dataset_cfg.get("task_weights"),
            )
            all_generators.append(generator)

        # Load Parquet files
        for parquet_spec in parquet_files:
            parquet_path = parquet_spec.replace("parquet:", "", 1)
            self.logger.info(f"Loading Parquet: {parquet_path}")
            # Use HuggingFace to load Parquet
            cache_dir = self.dataset_cfg.get("cache_dir")
            cache_config = {"cache_dir": cache_dir} if cache_dir else {}
            parquet_ds = load_dataset(
                "parquet", data_files=parquet_path, split="train", **cache_config
            )

            # Convert to generator format (expects "content" field, convert to "text")
            def make_parquet_gen(ds):
                def parquet_gen():
                    for item in ds:
                        # Parquet files from pipeline should have "prompt" and "gen"
                        if "prompt" in item and "gen" in item:
                            text = f"{item['prompt']}\n\n{item['gen']}"
                        elif "text" in item:
                            text = item["text"]
                        elif "content" in item:
                            text = item["content"]
                        else:
                            continue
                        yield {"text": text}

                return parquet_gen

            all_generators.append(make_parquet_gen(parquet_ds)())

        # Load HuggingFace datasets using stream_rust
        if hf_datasets:
            hf_generator = self.stream_rust(
                dataset_names=hf_datasets,
                cache_dir=self.dataset_cfg.get("cache_dir"),
                use_cache=self.dataset_cfg.get("use_cache", True),
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
            all_generators.append(hf_generator)

        # Combine all generators
        use_cache = self.dataset_cfg.get("use_cache", True)
        if use_cache:
            # Load all into memory for cached mode
            self.logger.info("Loading mixed datasets in cached mode...")
            all_items = []
            for gen in all_generators:
                all_items.extend(list(gen))
            self.logger.info(f"Loaded {len(all_items)} total samples from mixed datasets")
            dataset = Dataset.from_list(all_items)

            if self.train_cfg.get("dataloader_num_workers", 12) > 1:
                self.logger.warning("Mixed datasets - reducing workers to 1 (single-shard dataset)")
                self.train_cfg["dataloader_num_workers"] = 1

            return dataset
        else:
            # Streaming mode: create interleaved generator
            self.logger.info("Loading mixed datasets in streaming mode...")

            def combined_generator():
                # Simple round-robin interleaving
                generators = [iter(gen) for gen in all_generators]
                active = [g for g in generators]

                while active:
                    for i, gen in enumerate(active):
                        try:
                            yield next(gen)
                        except StopIteration:
                            active.remove(gen)

            return IterableDataset.from_generator(combined_generator)
