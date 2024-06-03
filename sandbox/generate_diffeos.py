import jax.numpy as jnp
from itertools import product

def generate_matrices_jax(values: jnp.array,
                          dim: int = 2, 
                          norm_cutoff: float = 2) -> list[jnp.array]:
    """
    Generate A_mn matirx for diffeos
    
    Args:
        values (jnp.array): Array of possible entires to try
            Example `values = jnp.linspace(0,0.5, 8)`
        dim (jnp.array): Length & width of matrix. In basis of sin(...)
        norm_cutoff: Removes matricies w/ a higher L1 norm.
    """

    # Create all possible combinations of 4 elements
    all_combinations = jnp.array(list(product(values, repeat=4)))

    # Compute the sum of each combination and filter
    sums = jnp.sum(all_combinations, axis=1)
    valid_combinations = all_combinations[sums < norm_cutoff]

    # Reshape the valid combinations to form 2x2 matrices
    matrices = valid_combinations.reshape(-1, dim, dim)

    return matrices