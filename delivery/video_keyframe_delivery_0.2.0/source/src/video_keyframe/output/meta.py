from dataclasses import asdict


def build_meta(state, config) -> dict:
    meta = asdict(state)
    meta.update(model=config.model.name, sample_fps=config.sampling.sample_fps,
                sample_stride=config.sampling.sample_stride,
                candidate_quantile=config.scoring.candidate_quantile,
                negative_weight=config.scoring.negative_weight,
                min_match_frame_gap=config.postprocess.min_match_frame_gap,
                max_match_count=config.postprocess.max_match_count,
                config=asdict(config))
    return meta
