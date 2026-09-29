"""Adaptive rollout and per-person/category reward state for PA training."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .personalization import normalize_person_token


SUPPORTED_ROLLOUT_LENGTHS = (2, 4, 8)
EASY_SATURATION_MEAN = 0.85
EASY_SATURATION_DISPERSION = 0.05
INFORMATIVE_DISPERSION = 0.15
HARD_MEAN = 0.50

INITIAL_EMA = 0.5
EMA_DECAY = 0.9
RAW_MULTIPLIER_MIN = 0.5
RAW_MULTIPLIER_MAX = 1.5
CATEGORIES = ("identity", "action", "clothing", "location", "emotion")


def _finite_float(value, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


@dataclass(frozen=True)
class DynamicRolloutDecision:
    target_count: int
    should_update: bool
    reason: str

    @property
    def needs_more(self) -> bool:
        return self.reason in {"expand", "expand_hard"}


def decide_dynamic_rollout(rewards) -> DynamicRolloutDecision:
    rewards = [_finite_float(reward, "reward") for reward in rewards]
    count = len(rewards)
    if count not in SUPPORTED_ROLLOUT_LENGTHS:
        raise ValueError(f"rewards must have length 2, 4, or 8; got {count}")
    mean = sum(rewards) / count
    dispersion = max(rewards) - min(rewards)

    if count == 2:
        if mean >= EASY_SATURATION_MEAN and dispersion < EASY_SATURATION_DISPERSION:
            return DynamicRolloutDecision(2, False, "easy_saturated")
        if dispersion >= INFORMATIVE_DISPERSION:
            return DynamicRolloutDecision(2, True, "informative")
        return DynamicRolloutDecision(4, False, "expand")
    if count == 4:
        if dispersion >= INFORMATIVE_DISPERSION:
            return DynamicRolloutDecision(4, True, "informative")
        if mean <= HARD_MEAN:
            return DynamicRolloutDecision(8, False, "expand_hard")
        return DynamicRolloutDecision(4, False, "low_dispersion")
    if dispersion >= INFORMATIVE_DISPERSION:
        return DynamicRolloutDecision(8, True, "informative")
    return DynamicRolloutDecision(8, False, "zero_advantage")


@dataclass
class PersonalizedAdaptiveState:
    initial_ema: float = INITIAL_EMA
    decay: float = EMA_DECAY
    raw_multiplier_min: float = RAW_MULTIPLIER_MIN
    raw_multiplier_max: float = RAW_MULTIPLIER_MAX

    def __post_init__(self) -> None:
        self.profiles: dict[str, dict[str, float]] = {}
        self.initial_ema = _finite_float(self.initial_ema, "initial_ema")
        self.decay = _finite_float(self.decay, "decay")
        self.raw_multiplier_min = _finite_float(self.raw_multiplier_min, "raw_multiplier_min")
        self.raw_multiplier_max = _finite_float(self.raw_multiplier_max, "raw_multiplier_max")
        if not 0 <= self.initial_ema <= 1:
            raise ValueError("initial_ema must be between 0 and 1")
        if not 0 <= self.decay <= 1:
            raise ValueError("decay must be between 0 and 1")
        if self.raw_multiplier_min > self.raw_multiplier_max:
            raise ValueError("raw multiplier bounds are invalid")

    def _profile(self, person: str) -> dict[str, float]:
        person = normalize_person_token(person)
        return self.profiles.setdefault(person, {category: self.initial_ema for category in CATEGORIES})

    @staticmethod
    def _check_category(category: str) -> str:
        if category not in CATEGORIES:
            raise ValueError(f"unknown category: {category}")
        return category

    def multipliers(self, person: str) -> dict[str, float]:
        profile = self._profile(person)
        mean_ema = sum(profile.values()) / len(CATEGORIES)
        raw = {
            category: max(self.raw_multiplier_min, min(self.raw_multiplier_max, 1 + (mean_ema - profile[category])))
            for category in CATEGORIES
        }
        mean_raw = sum(raw.values()) / len(CATEGORIES)
        return {category: value / mean_raw for category, value in raw.items()}

    def multiplier(self, person: str, category: str) -> float:
        self._check_category(category)
        return self.multipliers(person)[category]

    def observe(self, person: str, category: str, reward: float) -> None:
        category = self._check_category(category)
        reward = _finite_float(reward, "reward")
        profile = self._profile(person)
        clipped = max(-1.0, min(1.0, reward))
        value = (clipped + 1.0) / 2.0
        profile[category] = self.decay * profile[category] + (1.0 - self.decay) * value

    def observe_many(self, observations) -> None:
        for person, category, reward in observations:
            self.observe(person, category, reward)

    def state_dict(self) -> dict:
        return {
            "initial_ema": self.initial_ema,
            "decay": self.decay,
            "raw_multiplier_min": self.raw_multiplier_min,
            "raw_multiplier_max": self.raw_multiplier_max,
            "profiles": {person: dict(profile) for person, profile in self.profiles.items()},
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "PersonalizedAdaptiveState":
        result = cls(
            initial_ema=state.get("initial_ema", INITIAL_EMA),
            decay=state.get("decay", EMA_DECAY),
            raw_multiplier_min=state.get("raw_multiplier_min", RAW_MULTIPLIER_MIN),
            raw_multiplier_max=state.get("raw_multiplier_max", RAW_MULTIPLIER_MAX),
        )
        for person, profile in state.get("profiles", {}).items():
            if set(profile) != set(CATEGORIES):
                unknown = set(profile) - set(CATEGORIES)
                raise ValueError(f"unknown category: {next(iter(unknown), 'missing')}")
            normalized_person = normalize_person_token(person)
            if normalized_person in result.profiles:
                raise ValueError(f"duplicate normalized person: {normalized_person}")
            loaded_profile = {}
            for category in CATEGORIES:
                value = _finite_float(profile[category], f"profile[{category}]")
                if not 0 <= value <= 1:
                    raise ValueError(f"profile[{category}] must be between 0 and 1")
                loaded_profile[category] = value
            result.profiles[normalized_person] = loaded_profile
        return result
