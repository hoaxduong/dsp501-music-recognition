"""Features shared by the training notebook and the Streamlit match decision."""

from math import log1p

import numpy as np
from scipy.special import expit

FEATURE_NAMES = ("log_top_votes", "log_vote_gap", "log_vote_density")


def match_features(result):
    """Use only the two best vote counts and the query token count."""
    best, runner = result["candidates"][:2]
    top = best["votes"]
    return (
        log1p(top),
        log1p(top) - log1p(runner["votes"]),
        log1p(top / max(1, result["tokens"])),
    )


def match_score(features, model):
    scaled = (np.asarray(features) - model["mean"]) / model["scale"]
    return expit(scaled @ model["coefficients"] + model["intercept"])


def accept_match(result, model):
    score = float(match_score(match_features(result), model))
    accepted = result["candidates"][0]["votes"] > 0 and score >= model["threshold"]
    return accepted, score
