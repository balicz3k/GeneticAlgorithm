from typing import List

def sphere_function(variables: List[float]) -> float:
    return sum(x ** 2 for x in variables)

def hyperellipsoid_function(variables: List[float]) -> float:
    total_sum = 0.0
    for i in range(len(variables)):
        total_sum += sum(variables[j] ** 2 for j in range(i + 1))
    return total_sum

def martin_and_gaddy_function(variables: List[float]) -> float:
    if(len(variables) != 2):
        raise ValueError("Martin & Gaddy function requires exactly 2 variables")
    return (variables[0] - variables[1]) ** 2 + ((variables[0] + variables[1] - 10) / 3) ** 2

AVAILABLE_FUNCTIONS = {
    "Sphere": sphere_function,
    "Hyperellipsoid": hyperellipsoid_function,
    "Martin & Gaddy": martin_and_gaddy_function,
}
