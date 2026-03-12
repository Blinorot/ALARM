import argparse

import datasets
import numpy as np
from combined_dataset import ALL_DATASETS, DATA_PATH, ROOT_PATH
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from transformers import AutoTokenizer

INDEX_FILTERED_PATH = ROOT_PATH / "data" / "datasets" / "indexes_filtered"
INDEX_PATH = ROOT_PATH / "data" / "datasets" / "indexes"
RESPONSE_PATH = ROOT_PATH / "data" / "datasets" / "responses"


SEED = 123


def validate_elem(elem, tokens_max_limit, tokens_min_limit, tokenizer):
    llm_answer = elem["llm_answer_with_context"]
    llm_answer_ids = tokenizer(
        llm_answer,
        padding=False,
        truncation=False,
        add_special_tokens=False,
    ).input_ids
    if len(llm_answer_ids) > tokens_max_limit:
        return False
    if len(llm_answer_ids) < tokens_min_limit:
        return False
    return True


def process_dataset_indexes(
    ds, indexes, tokens_max_limit, tokens_min_limit, tokenizer, n_jobs=16
):
    indexes_mask = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(validate_elem)(elem, tokens_max_limit, tokens_min_limit, tokenizer)
        for elem in tqdm(ds)
    )
    return indexes[indexes_mask]


def get_response_name(
    dataset_name,
    rephrase,
    use_checker,
    actual_model_name,
    max_tokens,
    max_thinking_tokens,
):
    response_name = f"{dataset_name}"
    if rephrase:
        response_name += "_rephrased"
        if max_thinking_tokens >= 0:
            response_name += f"_{max_thinking_tokens}"
    if use_checker:
        response_name += "_filtered"
    response_name += f"_{actual_model_name}_{max_tokens}"
    return response_name


def validate_and_filter_answers(
    n_jobs,
    model_name,
    max_tokens,
    use_checker,
    rephrase,
    max_thinking_tokens,
    tokens_max_limit,
    tokens_min_limit,
    force_rewrite,
):
    """
    Validates responses and removes those which are too long probably due to
    some errors.
    Uses already filtered indexes from validate_and_filter_indexes.py.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    actual_model_name = model_name.split("/")[-1]
    for dataset_name in ALL_DATASETS:
        response_name = get_response_name(
            dataset_name,
            rephrase=rephrase,
            use_checker=use_checker,
            actual_model_name=actual_model_name,
            max_tokens=max_tokens,
            max_thinking_tokens=max_thinking_tokens,
        )
        response_index_path = INDEX_FILTERED_PATH / dataset_name
        response_index_path = response_index_path / actual_model_name / response_name
        if response_index_path.exists():
            if force_rewrite:
                print(
                    f"{dataset_name} filtered split exist, but overwriting is forced..."
                )
            else:
                print(f"{dataset_name} filtered split already exists, skipping...")
                continue
        response_path = RESPONSE_PATH / actual_model_name / response_name
        ds = datasets.load_from_disk(response_path)["train"]
        data_filtered_dir = INDEX_FILTERED_PATH / dataset_name
        response_index_path.mkdir(exist_ok=True, parents=True)

        train_indexes = np.load(data_filtered_dir / "train_indexes.npy")
        val_indexes = np.load(data_filtered_dir / "validation_indexes.npy")

        # sort for faster elem access
        train_indexes = np.sort(train_indexes)
        val_indexes = np.sort(val_indexes)

        train_split = ds.select(train_indexes)
        val_split = ds.select(val_indexes)

        filtered_train_indexes = process_dataset_indexes(
            train_split,
            train_indexes,
            n_jobs=n_jobs,
            tokens_max_limit=tokens_max_limit,
            tokens_min_limit=tokens_min_limit,
            tokenizer=tokenizer,
        )
        filtered_val_indexes = process_dataset_indexes(
            val_split,
            val_indexes,
            n_jobs=n_jobs,
            tokens_max_limit=tokens_max_limit,
            tokens_min_limit=tokens_min_limit,
            tokenizer=tokenizer,
        )

        print(f"Filtered {dataset_name}")
        train_len = train_indexes.shape[-1]
        filtered_train_len = filtered_train_indexes.shape[-1]
        train_removed = train_indexes.shape[-1] - filtered_train_indexes.shape[-1]
        train_percent = round(train_removed / train_indexes.shape[-1] * 100, 2)
        val_len = val_indexes.shape[-1]
        filtered_val_len = filtered_val_indexes.shape[-1]
        val_removed = val_indexes.shape[-1] - filtered_val_indexes.shape[-1]
        val_percent = round(val_removed / val_indexes.shape[-1] * 100, 2)
        print(f"Original/Filtered train: {train_len}/{filtered_train_len}")
        print(f"Removed: {train_removed} ({train_percent} %)")
        print(f"Original/Filtered val: {val_len}/{filtered_val_len}")
        print(f"Removed: {val_removed} ({val_percent} %)")

        np.save(response_index_path / "train_indexes.npy", filtered_train_indexes)
        np.save(response_index_path / "validation_indexes.npy", filtered_val_indexes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Validate responses in datasets")
    parser.add_argument(
        "--n-jobs",
        default=16,
        type=int,
        help="Number of joblib workers",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-0.6B",
        type=str,
        help="LLM name (Default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--max-tokens",
        default=512,
        type=int,
        help="Max tokens for generation (Default: 512)",
    )
    parser.add_argument(
        "--max-thinking-tokens",
        default=-1,
        type=int,
        help="Max tokens for internal thinking (Default: -1)",
    )
    parser.add_argument(
        "--no-use-checker",
        dest="use_checker",
        action="store_false",
        help="Disable the checker",
    )
    parser.set_defaults(use_checker=True)
    parser.add_argument(
        "--no-rephrase",
        dest="rephrase",
        action="store_false",
        help="Disable the rephrase stage",
    )
    parser.set_defaults(rephrase=True)
    parser.add_argument(
        "--tokens-max-limit",
        default=612,
        type=int,
        help="Max tokens allowed (Default: 612)",
    )
    parser.add_argument(
        "--tokens-min-limit",
        default=100,
        type=int,
        help="Min tokens allowed (Default: 100)",
    )
    parser.add_argument(
        "--force-rewrite",
        dest="force_rewrite",
        action="store_true",
        help="Force rewriting directories",
    )
    parser.set_defaults(force_rewrite=False)

    args = parser.parse_args()
    validate_and_filter_answers(
        n_jobs=args.n_jobs,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        use_checker=args.use_checker,
        rephrase=args.rephrase,
        max_thinking_tokens=args.max_thinking_tokens,
        tokens_max_limit=args.tokens_max_limit,
        tokens_min_limit=args.tokens_min_limit,
        force_rewrite=args.force_rewrite,
    )
