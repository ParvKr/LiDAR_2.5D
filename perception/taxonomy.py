from enum import IntEnum


class SemanticClass(IntEnum):
    """
    SemanticKITTI learning taxonomy.

    Class 0 is ignored during training/evaluation.
    Classes 1-19 are the semantic classes used by the SemanticKITTI semantic segmentation benchmark.
    """

    UNKNOWN = 0

    CAR = 1
    BICYCLE = 2
    MOTORCYCLE = 3
    TRUCK = 4
    OTHER_VEHICLE = 5

    PERSON = 6
    BICYCLIST = 7
    MOTORCYCLIST = 8

    ROAD = 9
    PARKING = 10
    SIDEWALK = 11
    OTHER_GROUND = 12

    BUILDING = 13
    FENCE = 14

    VEGETATION = 15
    TRUNK = 16
    TERRAIN = 17

    POLE = 18
    TRAFFIC_SIGN = 19


NUM_CLASSES = 20

LEARNING_MAP = {
        0: 0,
        1: 0,
        10: 1,
        11: 2,
        13: 5,
        15: 3,
        16: 5,
        18: 4,
        20: 5,
        30: 6,
        31: 7,
        32: 8,
        40: 9,
        44: 10,
        48: 11,
        49: 12,
        50: 13,
        51: 14,
        52: 0,
        60: 9,
        70: 15,
        71: 16,
        72: 17,
        80: 18,
        81: 19,
        99: 0,
        252: 1,
        253: 7,
        254: 6,
        255: 8,
        256: 5,
        257: 5,
        258: 4,
        259: 5,
}

def extract_semantic_id(raw_label: int) -> int:
    """
    Extract the 16-bit semantic ID from a raw SemanticKITTI labels.
    SemanticKITTI stores semantic and instance information together in a 32-bit labels.
    """
    return int(raw_label) & 0xFFFF


def map_semantic_kitti_label(raw_label: int) -> SemanticClass:
    """
    Convert a raw SemanticKITTI labels into the project's 19-class learning taxonomy.
    """
    semantic_id = extract_semantic_id(raw_label)
    return SemanticClass(
        LEARNING_MAP.get(semantic_id, 0)
    )
