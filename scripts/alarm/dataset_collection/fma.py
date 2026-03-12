import ast
import json
import os
import shutil
from collections import defaultdict

import datasets
import pandas as pd
import torchaudio
from tqdm.auto import tqdm

from utils import DATA_PATH, SAMPLING_RATE, cli_download, get_audio_description

# https://arxiv.org/abs/1612.01840
DATASET_NAME = "fma"
URL_LINK = "https://os.unil.cloud.switch.ch/fma/fma_large.zip"
META_URL_LINK = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"


# https://github.com/mdeff/fma/blob/master/utils.py
def load_metadata(filepath):
    filename = os.path.basename(filepath)

    if "features" in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if "echonest" in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if "genres" in filename:
        return pd.read_csv(filepath, index_col=0)

    if "tracks" in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [
            ("track", "tags"),
            ("album", "tags"),
            ("artist", "tags"),
            ("track", "genres"),
            ("track", "genres_all"),
        ]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [
            ("track", "date_created"),
            ("track", "date_recorded"),
            ("album", "date_created"),
            ("album", "date_released"),
            ("artist", "date_created"),
            ("artist", "active_year_begin"),
            ("artist", "active_year_end"),
        ]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ("small", "medium", "large")
        try:
            tracks["set", "subset"] = tracks["set", "subset"].astype(
                "category", categories=SUBSETS, ordered=True
            )
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks["set", "subset"] = tracks["set", "subset"].astype(
                pd.CategoricalDtype(categories=SUBSETS, ordered=True)
            )

        COLUMNS = [
            ("track", "genre_top"),
            ("track", "license"),
            ("album", "type"),
            ("album", "information"),
            ("artist", "bio"),
        ]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype("category")

        return tracks


def get_full_genres(genres):
    full_genres = {}
    for _, row in genres.iterrows():
        main_genre = row["title"]
        genre_list = [main_genre]
        parent = row["parent"]
        while parent != 0:
            genre_row = genres.loc[parent]
            genre = genre_row["title"]
            parent = genre_row["parent"]
            genre_list.append(genre)
        full_genres[main_genre] = "/".join(genre_list)
    return full_genres


def download_dataset():
    dataset_dict = defaultdict(list)

    data_dir = DATA_PATH / DATASET_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / "fma.zip"
    cli_download(URL_LINK, output=str(arc_path))
    shutil.unpack_archive(arc_path, data_dir)
    os.remove(arc_path)

    arc_path = data_dir / "meta.zip"
    cli_download(META_URL_LINK, output=str(arc_path))
    shutil.unpack_archive(arc_path, data_dir)
    os.remove(arc_path)

    metadata_dir = data_dir / "fma_metadata"
    tracks = load_metadata(metadata_dir / "tracks.csv")
    tracks = tracks.loc[tracks["set", "subset"] <= "large"]
    tracks = tracks.loc[tracks["set", "split"] == "training"]
    genres = load_metadata(metadata_dir / "genres.csv")

    tmp_dir = data_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True, parents=True)

    full_genres = get_full_genres(genres)
    index = tracks.index.tolist()
    n_broken = 0
    for track_id in tqdm(index):
        track_info = tracks.loc[track_id]

        full_track_id = f"{track_id:06d}"
        audio_dir = full_track_id[:3]
        audio_path = data_dir / "fma_large" / audio_dir / f"{full_track_id}.mp3"
        audio_path = str(audio_path)

        try:
            audio_frames, audio_sr = torchaudio.load(audio_path)
        except:
            n_broken += 1
            print(f"Broken files count {n_broken}, file: {full_track_id}")
            continue

        if audio_frames.shape[0] != 1:
            audio_frames = audio_frames.mean(dim=0).unsqueeze(0)
        if audio_sr != SAMPLING_RATE:
            audio_frames = torchaudio.functional.resample(
                audio_frames, audio_sr, SAMPLING_RATE
            )

        # resave because otherwise torchcodec will think these files are corrupted
        new_audio_dir = tmp_dir / audio_dir
        new_audio_dir.mkdir(exist_ok=True, parents=True)
        new_audio_path = new_audio_dir / f"{full_track_id}.wav"
        torchaudio.save(new_audio_path, audio_frames, sample_rate=SAMPLING_RATE)

        duration = audio_frames.shape[-1] / SAMPLING_RATE

        genre_list = track_info["track", "genres"]
        full_genre_list = []
        for genre_id in genre_list:
            genre_name = genres.loc[genre_id]["title"]
            full_genre_list.append(full_genres[genre_name])
        full_genre = " and ".join(full_genre_list)

        title = track_info["track", "title"]
        artist = track_info["artist", "name"]
        album = track_info["album", "title"]
        language_code = track_info["track", "language_code"]

        album_tags = track_info["album", "tags"]
        artist_tags = track_info["artist", "tags"]
        music_tags = track_info["track", "tags"]

        full_song_name = f"{artist} - {title}"

        elem_info = {}

        elem_info["duration"] = duration
        elem_info["text"] = "Music"
        elem_info["genres"] = full_genre
        elem_info["artist name"] = artist
        elem_info["song title"] = title

        # to avoid NaNs
        if isinstance(album, str):
            elem_info["album title"] = album

        if isinstance(language_code, str):
            elem_info["language code"] = language_code

        if len(album_tags) > 0:
            elem_info["album tags"] = "/".join(album_tags)
        if len(artist_tags) > 0:
            elem_info["artist tags"] = "/".join(artist_tags)
        if len(music_tags) > 0:
            elem_info["song tags"] = "/".join(music_tags)

        audio_description = get_audio_description(text_type="caption", **elem_info)

        # to ensure that all elements have value
        elem_info["album tags"] = elem_info.get("album tags", "")
        elem_info["artist tags"] = elem_info.get("artist tags", "")
        elem_info["song tags"] = elem_info.get("song tags", "")
        elem_info["album title"] = elem_info.get("album title", "")
        elem_info["language code"] = elem_info.get("language code", "")

        dataset_dict["audio_description"].append(audio_description)
        for k, v in elem_info.items():
            dataset_dict[k].append(v)
        dataset_dict["audio"].append(str(new_audio_path))
        dataset_dict["track_id"].append(full_track_id)
        dataset_dict["unique_id_1"].append(str(full_track_id))
        dataset_dict["full song name"].append(full_song_name)
        dataset_dict["unique_id_2"].append(str(full_song_name))

    ds = datasets.Dataset.from_dict(dataset_dict)
    ds = ds.sort(["unique_id_1", "unique_id_2"])
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.add_column("dataset_index", list(range(len(ds))))
    ds = datasets.DatasetDict({"train": ds})
    ds.save_to_disk(DATA_PATH / DATASET_NAME)

    # delete raw data
    shutil.rmtree(data_dir / "fma_metadata")
    shutil.rmtree(data_dir / "fma_large")
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    download_dataset()
