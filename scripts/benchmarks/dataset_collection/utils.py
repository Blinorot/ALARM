import subprocess
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT_PATH / "data" / "datasets" / "raw"
SAMPLING_RATE = 16000


def cli_download(url, output, method="curl"):
    if method == "wget":
        subprocess.run(["wget", "-c", "-O", output, url], check=True)
    elif method == "wget_no_check":
        subprocess.run(
            ["wget", "-c", "--no-check-certificate", "-O", output, url], check=True
        )
    elif method == "curl":
        subprocess.run(["curl", "-L", "-C", "-", "-o", output, url], check=True)
    elif method == "gdown_folder":
        subprocess.run(["gdown", "--folder", url, "-O", output], check=True)
    elif method == "gdown_file":
        subprocess.run(["gdown", "--fuzzy", url, "-O", output], check=True)
    else:
        raise NotImplementedError()


def get_audio_description(duration, text, text_type="speech", **metadata):
    if isinstance(text, str):
        duration = round(duration)

        if text_type == "caption":
            text = f"|{text}|"
        else:  # speech
            pass
        audio_minutes = duration // 60
        audio_seconds = duration % 60
        audio_time = f"{audio_minutes:02}:{audio_seconds:02}"
        time_info = "[" + "00:00-" + audio_time + "]"

        text = time_info + " " + text
    elif isinstance(text, list):
        total_duration = 0

        text_list = []
        for text_i, (start_time, end_time, speaker) in zip(text, duration):
            start_time = round(start_time)
            end_time = round(end_time)

            if text_type == "caption":
                text_i = f"|{text_i}|"
            else:  # speech
                pass
            audio_minutes = end_time // 60
            audio_seconds = end_time % 60
            audio_time = f"{audio_minutes:02}:{audio_seconds:02}"

            start_minutes = start_time // 60
            start_seconds = start_time % 60

            start_time = f"{start_minutes:02}:{start_seconds:02}"
            time_info = "[" + start_time + "-" + audio_time + f", {speaker}]"

            total_duration = end_time

            text_list.append(time_info + " " + text_i)

        text = " ".join(text_list)
        duration = total_duration
    else:
        raise NotImplementedError

    extra_comments = [f"duration: {duration}s"]
    for k, v in metadata.items():
        extra_comments.append(f"{k.lower()}: {v}")
    extra_comments = "(" + ", ".join(extra_comments) + ")"

    audio_description = text + " " + extra_comments
    return audio_description
