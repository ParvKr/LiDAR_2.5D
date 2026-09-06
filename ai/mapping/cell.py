from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class MapCell:
    """
    Represents a single cell in the 2.5D environment map.

    A cell accumulates observations from multiple LiDAR points.
    """

    point_count: int = 0
    occupied: bool = False

    semantic_class: Optional[int] = None
    semantic_confidence: float = 0.0

    dynamic_probability: float = 0.0
    last_timestamp: Optional[float] = None

    _height_sum: float = field(default=0.0, repr=False)
    _height_squared_sum: float = field(default=0.0, repr=False)
    _semantic_counts: Dict[int, int] = field(default_factory=dict, repr=False)

    @property
    def height(self) -> Optional[float]:
        """Return the mean observed height."""

        if self.point_count == 0:
            return None

        return self._height_sum / self.point_count

    @property
    def height_variance(self) -> Optional[float]:
        """Return the variance of observed heights."""

        if self.point_count == 0:
            return None

        mean = self.height
        variance = (
            self._height_squared_sum / self.point_count
            - mean ** 2
        )

        return max(0.0, variance)

    def add_observation(
        self,
        height: float,
        timestamp: Optional[float] = None,
        semantic_class: Optional[int] = None,
    ) -> None:
        """
        Add one LiDAR observation to this cell.
        """

        self._height_sum += float(height)
        self._height_squared_sum += float(height) ** 2

        self.point_count += 1
        self.occupied = True

        if timestamp is not None:
            self.last_timestamp = timestamp

        if semantic_class is not None:
            semantic_class = int(semantic_class)
            count = self._semantic_counts.get(semantic_class, 0) + 1
            self._semantic_counts[semantic_class] = count
            self.semantic_class = max(
                self._semantic_counts, key=self._semantic_counts.get
            )
            self.semantic_confidence = (
                self._semantic_counts[self.semantic_class] / self.point_count
            )
