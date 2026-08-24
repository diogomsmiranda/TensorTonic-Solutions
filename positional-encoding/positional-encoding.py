import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    if seq_len <= 0 or d_model <= 0:
        return None
    
    pe = np.zeros((seq_len,d_model), dtype=float)
    even_dim_indices = np.arange(0,d_model,2)
    odd_dim_indices = np.arange(1,d_model,2)
    pos_indices = np.arange(seq_len)[:, np.newaxis]

    angle_calc = pos_indices / np.power(base,(even_dim_indices)/d_model)
    
    pe[:, even_dim_indices] = np.sin(angle_calc[:, :len(even_dim_indices)])
    pe[:, odd_dim_indices] = np.cos(angle_calc[:, :len(odd_dim_indices)])
    return pe