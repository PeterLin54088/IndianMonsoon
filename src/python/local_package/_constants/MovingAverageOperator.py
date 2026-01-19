from local_package._constants.Strict_Read_Only import StrictReadOnlyMeta
from typing import Final


class MovingAverageOperator(metaclass=StrictReadOnlyMeta):
    WINDOW_SIZE: Final[int] = 28
