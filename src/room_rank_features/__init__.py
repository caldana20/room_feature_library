"""Build grouped pandas features for eight hotel room ranking configurations."""

from .parser import normalize_description, parse_description, room_text_without_beds
from .transformer import RoomFeatureTransformer, SCENARIOS, BASE_SCENARIOS, HISTORY_SCENARIOS
from .preferences import TravelerPreferenceTransformer, PREFERENCE_FEATURES, BED_PREFERENCE_FEATURES

__version__ = "4.3.1"
__all__ = [
    "RoomFeatureTransformer", "SCENARIOS", "normalize_description", "parse_description", "room_text_without_beds",
    "TravelerPreferenceTransformer", "PREFERENCE_FEATURES",
    "BASE_SCENARIOS", "HISTORY_SCENARIOS", "BED_PREFERENCE_FEATURES",
]
