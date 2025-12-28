#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Dennis H. Wuitz <dennis.wuitz@wavelens.io>
#
# SPDX-License-Identifier: MIT

import argparse
import torch
torch.manual_seed(42)


def get_arguments(defaults):
    """Parse command line arguments to get the max number."""
    parser = argparse.ArgumentParser(description="Generate the power set of numbers from 0 to max_number - 1.")

    parser.add_argument(
        "--max-number",
        type=int,
        default=defaults["max_number"],
        help="The maximum number (exclusive) for generating the power set.",
    )

    parser.add_argument(
        "--train-test-split",
        type=float,
        default=defaults["train_test_split"],
        help="The train-test split ratio.",
    )

    args, _unknown = parser.parse_known_args()

    return {
     "max_number": args.max_number,
     "train_test_split": args.train_test_split
    }


def generate_subsets_set_of_two_numbers(max_number: int):
    subsets = []
    for num_a in range(max_number + 1):
        for num_b in range(max_number + 1):
            subsets.append((num_a, num_b))
    return subsets


def generate_vocabulary(max_number: int):
    vocab = {i: str(i)  for i in range(max_number + 1)}
    vocab[max_number + 1] = "="
    return vocab


def generate_dataset_from_number_subsets(subsets, vocab_size: int):
    equal_token_id = vocab_size - 1;

    dataset = []
    for num_a_token_id, num_b_token_id in subsets:
        num_a = num_a_token_id
        num_b = num_b_token_id

        num_sum = (num_a + num_b) % (vocab_size - 1)
        num_sum_token_id = num_sum

        dataset.append([num_a_token_id, num_b_token_id, equal_token_id, num_sum_token_id])

    dataset = torch.tensor(dataset, dtype=torch.long)
    return dataset

def format_addition_to_string(element, vocab):
    num_a_token_id, num_b_token_id, equal_token_id, sum_token_id = element
    return f"{vocab[num_a_token_id]} + {vocab[num_b_token_id]} {vocab[equal_token_id]} {vocab[sum_token_id]}"


def generate_dataset_splits(dataset, train_test_split: float):
    num_samples = dataset.size(0)
    num_train_samples = int(num_samples * train_test_split)

    train_dataset = dataset[:num_train_samples]
    test_dataset = dataset[num_train_samples:]

    return train_dataset, test_dataset


