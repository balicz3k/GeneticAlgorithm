from typing import List

def sphere_function(variables: List[float]) -> float:
    return sum(x ** 2 for x in variables)

def hyperellipsoid_function(variables: List[float]) -> float:
    total_sum = 0.0
    for i in range(len(variables)):
        total_sum += sum(variables[j] ** 2 for j in range(i + 1))
    return total_sum

# Rejestr funkcji, który zostanie przetłumaczony na rozwijaną listę (Dropdown) w GUI
AVAILABLE_FUNCTIONS = {
    "Sphere": sphere_function,
    "Hyperellipsoid": hyperellipsoid_function
}
