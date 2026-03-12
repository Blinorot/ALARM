from src.model.audio.encoder_implementations.muq import MuQEncoder
from src.model.audio.encoder_implementations.sslam import SSLAMEncoder
from src.model.audio.encoder_implementations.w2vbert import W2VBERT2Encoder
from src.model.audio.encoder_implementations.whisper import WhisperEncoder

encoder_model_dict = {
    "w2vbert": W2VBERT2Encoder,
    "whisper": WhisperEncoder,
    "sslam": SSLAMEncoder,
    "muq": MuQEncoder,
}
