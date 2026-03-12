from src.model.audio.feature_extractor_implementations import (
    feature_extractor_model_dict,
)


class AudioFeatureExtractor:
    def __init__(
        self,
        feature_extractor_configs: list[dict],
    ):
        feature_extractors_dict = {}
        for feature_extractor_config in feature_extractor_configs:
            encoder_name = feature_extractor_config["encoder_name"]
            feature_extractor_cls = feature_extractor_model_dict[encoder_name]
            feature_extractor = feature_extractor_cls(**feature_extractor_config)
            feature_extractors_dict[encoder_name] = feature_extractor

        # sort to ensure that the order is fixed
        sorted_encoder_names = sorted(feature_extractors_dict.keys())
        self.feature_extractors = {}
        for name in sorted_encoder_names:
            self.feature_extractors[name] = feature_extractors_dict[name]

    def __call__(self, audio):
        # to avoid reading multiple times
        audio_data = audio.get_all_samples()
        audio = audio_data.data
        if audio.shape[0] != 1:
            # average of channels
            audio = audio.mean(dim=0)
            audio = audio.unsqueeze(0)
        sr = audio_data.sample_rate

        feature_extractor_audio = {}
        feature_extractor_attention = {}
        feature_extractor_lengths = {}
        for encoder_name, feature_extractor in self.feature_extractors.items():
            processed_audio, audio_attention, audio_lengths = feature_extractor(
                audio=audio, sr=sr
            )
            feature_extractor_audio[encoder_name] = processed_audio
            feature_extractor_attention[encoder_name] = audio_attention
            feature_extractor_lengths[encoder_name] = audio_lengths
        return {
            "audio": feature_extractor_audio,
            "audio_attention": feature_extractor_attention,
            "audio_lengths": feature_extractor_lengths,
        }
