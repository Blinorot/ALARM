from collections import defaultdict

import datasets
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, get_audio_description

# https://arxiv.org/abs/2406.08820
DATASET_NAME = "disfluencyspeech"

ELEMENT_TYPES = {
    "F": "Filled pauses (e.g. 'uh', 'um')",
    "E": "Explicit editing terms (e.g. 'I mean', 'sorry')",
    "D": "Discourse markers (e.g. 'you know', 'well')",
    "C": "Coordinating conjuctions (e.g. 'and', 'but')",
    "A": "Asides (comments that interrupt fluent flow)",
}


def process_elem(elem):
    processed_elem = {}
    elem_info = {}
    elem_info["text"] = elem["transcript_annotated"].strip()

    audio = elem["audio"]

    duration = audio.get_all_samples().duration_seconds
    elem_info["duration"] = duration

    special_annotations = []
    for elem_type, description in ELEMENT_TYPES.items():
        if f"{elem_type}" in elem_info["text"]:
            key = "{" + elem_type + " ...}"
            special_annotations.append(f"{key}: {description}")
    if "[" in elem_info["text"]:
        key = "[... + ...]"
        special_annotations.append(f"{key}: Restarts")
    if "¡" in elem_info["text"] or "¿" in elem_info["text"]:
        key = "¡...¿"
        special_annotations.append(f"{key}: non-speech sounds")
    if len(special_annotations) == 0:
        elem_info["special annotations"] = "No special annotations"
    else:
        elem_info["special annotations"] = "/".join(special_annotations)

    audio_description = get_audio_description(text_type="speech", **elem_info)

    processed_elem["audio_description"] = audio_description
    for k, v in elem_info.items():
        processed_elem[k] = v

    # there is only one speaker
    processed_elem["unique_id_1"] = str(elem_info["text"])
    # extra word so it does not look like the corresponding metadata field
    # needed for prompt creation
    processed_elem["unique_id_2"] = f"annotations: {elem_info['special annotations']}"

    processed_elem["audio"] = audio
    return processed_elem


def download_dataset():
    ds = datasets.load_dataset(
        "amaai-lab/DisfluencySpeech",
        split="train",
        revision="b7da294fe3a70dd96df6640893f2a5dfc2c87638",
    )

    all_columns = set(ds.column_names)
    good_columns = {
        "text",
        "audio",
        "audio_description",
        "duration",
        "special annotations",
        "unique_id_1",
        "unique_id_2",
    }
    remove_columns = all_columns - good_columns

    ds = ds.map(
        process_elem,
        desc="Processing DisfluencySpeech",
        remove_columns=remove_columns,
    )

    # sort to ensure the same index across machines
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)


if __name__ == "__main__":
    download_dataset()
