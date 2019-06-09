from typing import NamedTuple


class VoiceParam(NamedTuple):
    sample_rate: int = 22050
    top_db: float = None
    pad_second: float = 0.0


class AcousticFeatureParam(NamedTuple):
    frame_period: int = 5
    order: int = 8
    alpha: float = 0.466
    f0_estimating_method: str = 'harvest'  # dio / harvest


class Param(NamedTuple):
    voice_param: VoiceParam = VoiceParam()
    acoustic_feature_param: AcousticFeatureParam = AcousticFeatureParam()
