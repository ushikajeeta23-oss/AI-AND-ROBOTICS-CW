"""Report-aligned, DISPLAY-ONLY handling recommendations.

Part A proposes physical mass, width, reach, station, ownership and grip checks.
Part B does not measure these and must never issue robot motor commands.
"""
from dataclasses import dataclass
from .config import DEFAULT_THRESHOLD, MAX_ITEM_MASS_G, GRIP_MIN_MM, GRIP_MAX_MM, WORKSPACE_REACH_MM

DESTINATIONS = {
    'bin': 'report at the station', 'book': 'report at the station',
    'bottle': 'approved refreshments tray', 'chair': 'report at the station',
    'keyboard': 'report at the station', 'mouse': 'approved accessories tray',
    'mug': 'report at the station', 'notebook': 'report at the station',
    'phone': 'report at the station', 'stapler': 'approved stationery tray',
}
CANDIDATE = {'bottle', 'mouse', 'stapler'}

@dataclass(frozen=True)
class Decision:
    action: str
    destination: str
    reason: str


def decide(item: str, confidence: float, threshold: float = DEFAULT_THRESHOLD) -> Decision:
    if item not in DESTINATIONS:
        return Decision('REVIEW', 'none', 'This class is outside the ten-class project.')
    if not 0 <= confidence <= 1 or confidence < threshold:
        return Decision('REVIEW', DESTINATIONS[item],
                        f'Item recognition score is below {threshold:.2f}; a person must check it.')
    if item not in CANDIDATE:
        return Decision('REPORT_ONLY', DESTINATIONS[item],
                        'The initial design does not approve this category for automatic handling.')
    qualification = {
        'bottle': 'Must be an EMPTY PLASTIC bottle, confirmed separately.',
        'mouse': 'Must be UNPLUGGED and authorized for handling.',
        'stapler': 'Must be a small stapler approved for this robot.',
    }[item]
    return Decision('CHECK_BEFORE_LIFT', DESTINATIONS[item],
                    qualification + ' Additional independent checks required: '
                    f'known measured weight <= {MAX_ITEM_MASS_G} g; grip width '
                    f'{GRIP_MIN_MM}-{GRIP_MAX_MM} mm; tray and object within '
                    f'{WORKSPACE_REACH_MM} mm of arm base; clear route and secure trial lift. '
                    'An image classifier cannot verify these conditions.')
