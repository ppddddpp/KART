from dataclasses import dataclass

@dataclass
class KARTConfig:
    in_features: int
    out_features: int
    Q: int = 8
    D: int = 4
    R: int = 4
    K: int = 4
    M: int = 4
    degree: int = 3

    basis_type: str = 'fourier'
    act_type: str = 'silu'
    shift_type: str = 'deterministic'
    use_domain_bound: bool = True
    init_std: float = 1.0
    
    def __post_init__(self):
        assert self.in_features > 0, "in_features must be greater than 0"
        assert self.out_features > 0, "out_features must be greater than 0"
        assert self.Q > 0 and self.D > 0 and self.R > 0, "Parameters Q, D, R must be greater than 0"
        assert self.K > 0 and self.M > 0, "Parameters K, M must be greater than 0"

        valid_basis = ['fourier', 'b_spline', 'polynomial', 'linear']
        assert self.basis_type in valid_basis, f"basis_type must be one of {valid_basis}"

        valid_shifts = ['deterministic', 'learned']
        assert self.shift_type in valid_shifts, f"shift_type must be one of {valid_shifts}"