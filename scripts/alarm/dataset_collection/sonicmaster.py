import json
import shutil
from collections import defaultdict

import datasets
import torchaudio
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, get_audio_description

# https://arxiv.org/abs/2508.03448
DATASET_NAME = "sonicmaster"

DESCRIPTIONS = {
    "Reverb_big": "the audio is convolved with a simulated big room impulse response",
    "Reverb_small": "the audio is convolved with a simulated small room impulse response",
    "EQ_boom": "the audio boominess is reduced by a low-shelf filter at 120 Hz by 10-20 dB",
    "Amplitude_volume": "the audio gain is adjusted to a maximum amplitude within the set {0.001, 0.003, 0.01, 0.05}",
    "Dynamics_comp": "the audio dynamics are modified by applying a feedforward compressor with attack 3-80 ms, release 80-250 ms, threshold -45 to -38 dB, ratio 6-45, and make-up gain 16-25 dB",
    "EQ_mud": "the audio muddiness is increased with a second-order Chebyshev Type II bandpass (200-500 Hz) by 6-15 dB",
    "EQ_dark": "the audio perceived brightness is increased with a high-shelf filter at 6 kHz by 6-15 dB",
    "Amplitude_clip": "the audio level is modified to a maximum amplitude within the {2, 3, 5} set and clipped",
    "EQ_airy": "the audio airness is reduced via a high-shelf filter at 10 kHz by 10-20 dB",
    "Stereo_stereo": "the left and right channels of the audio are combined to erase the spatial image",
    "EQ_warm": "the audio warmth is reduced by a low-shelf filter at 400 Hz by 6-20 dB",
    "EQ_vocal": "the vocal-range frequencies are attenuated using a second-order Chebyshev Type II bandpass (350-3500 Hz) by 6-20 dB",
    "Reverb_real": "the audio is convolved with a real room impulse response",
    "Reverb_mix": "the audio is convolved with a simulated mixed material room impulse response",
    "EQ_xband": "the audio is degraded by applying 8 to 12 band parameteric EQ with -6 to +6 range for each band",
    "EQ_mic": "the audio is degraded by convolving the signal with a microphone transfer function",
    "EQ_bright": "the audio brightness is reduced using a high-shelf filter at 6 kHz by 6-15 dB",
    "Dynamics_punch": "the audio dynamics are modified by appling a feedforward transient shaper with attack 3 ms, release 150 ms, adaptive threshold, and reduction of 8-15 dB",
    "EQ_clarity": "the audio clarity is degraded using a Butterworth low-pass filter (order 3-5) with cutoff at 2 kHz",
}


def process_elem(elem, tmp_dir, wav_id):
    processed_elem = {}
    elem_info = {}
    elem_info["text"] = "Music"
    meta = json.loads(elem["meta"])

    elem_info["song name"] = meta["name"]
    elem_info["genres"] = "/".join(meta["genres"])

    elem_info["instrumental or with vocals"] = meta["vocalinstrumental"]
    if meta["vocalinstrumental"] == "vocal":
        elem_info["singer gender"] = meta["gender"].title()

    elem_info["tags"] = "/".join(meta["vartags"])

    elem_info["degraded using"] = " and ".join(meta["degradations"])

    degradation_list = []
    for k, v in meta["degradation_tracking"].items():
        if len(v) != 0:
            v_type = v[0]
            effect = f"{k}_{v_type}"
            degradation_list.append(f"{effect.lower()} - {DESCRIPTIONS[effect]}")
    degradations_desc = " / ".join(degradation_list)
    elem_info["degradataion description"] = "[" + degradations_desc + "]"

    id = elem["id"]
    # name = meta["name"].replace(" ", "_")
    audio_path = tmp_dir / f"{wav_id}_{id}.wav"
    audio_path = str(audio_path)
    audio = elem["audio"].get_all_samples()
    audio_tensor = audio.data
    audio_sr = audio.sample_rate
    if audio_tensor.shape[0] != 1:
        audio_tensor = audio_tensor.mean(dim=0).unsqueeze(0)
    if audio_sr != SAMPLING_RATE:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, audio_sr, SAMPLING_RATE
        )
    torchaudio.save(audio_path, audio_tensor, sample_rate=audio_sr)

    duration = audio.duration_seconds
    elem_info["duration"] = duration

    audio_description = get_audio_description(text_type="caption", **elem_info)

    processed_elem["audio_description"] = audio_description

    elem_info["singer gender"] = elem_info.get("singer gender", "")

    for k, v in elem_info.items():
        processed_elem[k] = v

    processed_elem["id"] = elem["id"]
    processed_elem["unique_id_1"] = str(elem["id"])
    processed_elem["unique_id_2"] = str(meta["name"])
    # processed_elem["split"] = str(meta["split"])

    processed_elem["audio"] = audio_path
    return processed_elem


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    tmp_dir = data_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True, parents=True)

    ds = datasets.load_dataset(
        "amaai-lab/SonicMasterDataset",
        split="train",
        revision="8191774834e49b6661bf764361ae160bfd1f9fb4",
        verification_mode=datasets.VerificationMode("no_checks"),
    )

    ds = ds.remove_columns(["input_sr", "gt_flac", "gt_sr"])
    ds = ds.cast_column("input_flac", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.rename_column("input_flac", "audio")

    # effects = set()
    # print(ds)
    # for elem in tqdm(ds, total=len(ds)):
    #     meta = json.loads(elem["meta"])
    #     meta = meta["degradation_tracking"]
    #     for k, v in meta.items():
    #         if len(v) != 0:
    #             v_type = v[0]
    #             effect = f"{k}_{v_type}"
    #             effects.add(effect)
    # print(effects)

    wav_id = 0

    for elem in tqdm(ds):
        meta = json.loads(elem["meta"])
        split = str(meta["split"])
        if split != "train":
            continue

        processed_elem = process_elem(elem, tmp_dir, wav_id)
        wav_id += 1

        for k, v in processed_elem.items():
            dataset_dict[k].append(v)

    ds = datasets.Dataset.from_dict(dataset_dict)
    # sort to ensure the same index across machines
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    download_dataset()
