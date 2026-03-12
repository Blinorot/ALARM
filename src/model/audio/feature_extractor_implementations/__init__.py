from src.model.audio.feature_extractor_implementations.muq import MuQFeatureExtractor
from src.model.audio.feature_extractor_implementations.sslam import (
    SSLAMFeatureExtractor,
)
from src.model.audio.feature_extractor_implementations.w2vbert import (
    W2VBERT2FeatureExtractor,
)
from src.model.audio.feature_extractor_implementations.whisper import (
    WhisperFeatureExtractor,
)

feature_extractor_model_dict = {
    "w2vbert": W2VBERT2FeatureExtractor,
    "whisper": WhisperFeatureExtractor,
    "sslam": SSLAMFeatureExtractor,
    "muq": MuQFeatureExtractor,
}
