from LiDAR_25D.perception.taxonomy import (
    SemanticClass,
    extract_semantic_id,
    map_semantic_kitti_label,
)


def test_extract_semantic_id():
    raw_label = (42 << 16) | 10

    assert extract_semantic_id(raw_label) == 10


def test_map_raw_car_label():
    raw_label = (17 << 16) | 10

    result = map_semantic_kitti_label(raw_label)

    assert result == SemanticClass.CAR


def test_map_moving_car_label():
    raw_label = (25 << 16) | 252

    result = map_semantic_kitti_label(raw_label)

    assert result == SemanticClass.CAR


def test_unknown_label():
    raw_label = (10 << 16) | 999

    result = map_semantic_kitti_label(raw_label)

    assert result == SemanticClass.UNKNOWN