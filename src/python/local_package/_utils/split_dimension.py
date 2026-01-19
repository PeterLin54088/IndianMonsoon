import numpy as np

def split_dimension(
    array: np.ndarray, axis: int = 0, factors: tuple = (43, 365)
) -> np.ndarray:
    original_size = array.shape[axis]
    new_size = np.prod(factors)
    if original_size != new_size:
        raise ValueError(
            f"Cannot split dimension {axis} of size {original_size} into factors {factors} "
            f"with product {new_size}."
        )

    new_shape = (
        list(array.shape[:axis]) + list(factors) + list(array.shape[axis + 1 :])
    )
    reshaped_array = array.reshape(new_shape)
    return reshaped_array