#   Copyright 2020 DeepCode AG

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, cast

import numpy as np
import structlog
import torch
import utils.auxiliary as auxiliary
from sklearn.model_selection import train_test_split
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer
from utils.data_reader import DataPoint
from utils.data_reader import DataPointT5Representation

# Models for which we allow representation splitting
_ALLOWED_MODELS: Set[str] = set(["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"])


@dataclass
class SplittedDataSet:
    """Named class to hold return type of `split_filtered`"""

    train_inputs: List[str]
    train_labels: List[str]
    val_inputs: List[str]
    val_labels: List[str]
    test_inputs: List[str]
    test_labels: List[str]
    train_info: List[DataPoint]
    val_info: List[DataPoint]
    test_info: List[DataPoint]


@dataclass
class SplittedDataSetWithWarningIds:
    """Named class to hold return type of `create_data`. It has warning ID information."""

    train_inputs: List[str]
    train_labels: List[str]
    val_inputs: List[str]
    val_labels: List[str]
    test_inputs: Dict[str, List[str]]  # {warning_id -> List[inputs]}
    test_labels: Dict[str, List[str]]  # {warning_id -> List[labels]}
    train_info: List[DataPoint]
    val_info: List[DataPoint]
    test_info: Dict[str, List[DataPoint]]  # {warning_id -> List[datapoints]}


def extract_warning_types(data: List[DataPoint]) -> List[str]:
    all_warnings: Set[str] = set()
    for sample in data:
        all_warnings.add(sample.rule_report.rule_id)
    return list(all_warnings)


def filter_rule(data: List[DataPoint], rule_type: str) -> List[DataPoint]:
    """Filters `data` according to the linter (or DC, or equivalent) rule type id."""
    filtered_data = []
    for point in data:
        if point.rule_report.rule_id == rule_type:
            filtered_data.append(point)
    return filtered_data


def _split_filtered(
    filtered_data_train: List[DataPoint],
    filtered_data_val: List[DataPoint],
    filtered_data_test: List[DataPoint],
    rule_id: str,
    include_warning: bool,
    model_name: str,
    seed: int = 13,
) -> SplittedDataSet:
    """Splits the set of datapoints into train/val/test.

    Args:
        filtered_data_{train|val|test}: the result of filtering datapoints.
        include_warning: if set, the input representation of each datapoint will include the information
            about the warnings themselves.
        model_name: this function supports only `t5-***`, see `_ALLOWED_MODELS`.
    """
    log = structlog.get_logger().bind(function="prepare_data._split_filtered")

    if model_name not in _ALLOWED_MODELS:
        raise NotImplementedError("Provide your Getter function in data_reader.py")

    random.seed(seed)
    train_info = random.sample(filtered_data_train, len(filtered_data_train))
    t5_repr_train: List[DataPointT5Representation] = [
        data_point.get_t5_representation(include_warning=include_warning)
        for data_point in train_info
    ]

    random.seed(seed)
    val_info = random.sample(filtered_data_val, len(filtered_data_val))
    t5_repr_val: List[DataPointT5Representation] = [
        data_point.get_t5_representation(include_warning=include_warning)
        for data_point in val_info
    ]

    random.seed(seed)
    test_info = random.sample(filtered_data_test, len(filtered_data_test))
    t5_repr_test: List[DataPointT5Representation] = [
        data_point.get_t5_representation(include_warning=include_warning)
        for data_point in test_info
    ]

    return SplittedDataSet(
        train_inputs=[repr.input for repr in t5_repr_train],
        train_labels=[repr.output for repr in t5_repr_train],
        val_inputs=[repr.input for repr in t5_repr_val],
        val_labels=[repr.output for repr in t5_repr_val],
        test_inputs=[repr.input for repr in t5_repr_test],
        test_labels=[repr.output for repr in t5_repr_test],
        train_info=train_info,
        val_info=val_info,
        test_info=test_info,
    )


def create_data(
    data_train: List[DataPoint],
    data_val: List[DataPoint],
    data_test: List[DataPoint],
    linter_warnings: List[str],
    include_warning: bool,
    model_name: str,
) -> SplittedDataSetWithWarningIds:
    log = structlog.get_logger().bind(function="prepare_data.create_data")

    train: List[str] = []
    train_labels: List[str] = []
    val: List[str] = []
    val_labels: List[str] = []
    test, test_labels = defaultdict(list), defaultdict(list)
    n_test_samples = 0

    train_info, val_info = [], []
    test_info = defaultdict(list)
    aug_counter = 0
    for rule_id in linter_warnings:
        filtered_data_train = filter_rule(data_train, rule_id)
        filtered_data_val = filter_rule(data_val, rule_id)
        filtered_data_test = filter_rule(data_test, rule_id)
        split_dataset: SplittedDataSet = _split_filtered(
            filtered_data_train,
            filtered_data_val,
            filtered_data_test,
            rule_id=rule_id,
            include_warning=include_warning,
            model_name=model_name,
        )

        train += split_dataset.train_inputs
        train_labels += split_dataset.train_labels

        val += split_dataset.val_inputs
        val_labels += split_dataset.val_labels

        train_info += split_dataset.train_info
        val_info += split_dataset.val_info

        test[rule_id] = split_dataset.test_inputs
        test_labels[rule_id] = split_dataset.test_labels

        test_info[rule_id] = split_dataset.test_info

        n_test_samples += len(test[rule_id])

    log.info(
        "Data creation info ",
        train_size=len(train),
        val_size=len(val),
        test_size=n_test_samples,
        aug_size=aug_counter,
    )

    return SplittedDataSetWithWarningIds(
        train_inputs=train,
        train_labels=train_labels,
        val_inputs=val,
        val_labels=val_labels,
        test_inputs=test,
        test_labels=test_labels,
        train_info=train_info,
        val_info=val_info,
        test_info=test_info,
    )


class BugFixDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: BatchEncoding, targets: BatchEncoding):
        self.encodings = encodings
        self.target_encodings = targets

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(
            self.target_encodings["input_ids"][index], dtype=torch.long
        )
        return item

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])


def create_dataset(
    inputs: List[str],
    labels: List[str],
    tokenizer: PreTrainedTokenizer,
    pad_truncate: bool,
    max_length: Optional[int] = 256,
) -> BugFixDataset:
    if len(inputs) > 0:
        input_encodings = tokenizer(
            inputs, truncation=pad_truncate, padding=pad_truncate, max_length=max_length
        )
    else:
        input_encodings = BatchEncoding(None)
    if len(inputs) > 0:
        label_encodings = tokenizer(
            labels, truncation=pad_truncate, padding=pad_truncate, max_length=max_length
        )
    else:
        label_encodings = BatchEncoding(None)

    dataset = BugFixDataset(input_encodings, label_encodings)
    return dataset
