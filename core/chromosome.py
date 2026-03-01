import numpy as np
from typing import List, Tuple

class Chromosome:
    def __init__(self, bounds: List[Tuple[float, float]], precision: int, bits: np.ndarray = None):
        self.bounds = bounds
        self.precision = precision
        self.num_variables = len(bounds)
        
        self.m_list = self._calculate_bits_needed()
        
        self.total_bits = sum(self.m_list)
        
        if bits is not None:
            if len(bits) != self.total_bits:
                raise ValueError("Incorrect length of required bits!")
            self.bits = np.array(bits, dtype=np.int8)
        else:
            self.bits = np.random.randint(2, size=self.total_bits, dtype=np.int8)
            
        self.fitness = None
        
    def _calculate_bits_needed(self) -> List[int]:
        m_list = []
        for a, b in self.bounds:
            length = (b - a) * (10 ** self.precision)
            m = int(np.ceil(np.log2(length + 1)))
            m_list.append(m)
        return m_list

    def get_decoded_values(self) -> List[float]:
        decoded_vars = []
        current_idx = 0
        
        for i, (a, b) in enumerate(self.bounds):
            m = self.m_list[i]
            binary_segment = self.bits[current_idx : current_idx + m]
            current_idx += m
            
            powers_of_two = 2 ** np.arange(m - 1, -1, -1)
            decimal_val = np.sum(binary_segment * powers_of_two)
            
            real_val = a + decimal_val * (b - a) / ((2 ** m) - 1)
            
            decoded_vars.append(real_val)
            
        return decoded_vars
        
    def clone(self):
        return Chromosome(self.bounds, self.precision, bits=self.bits.copy())
