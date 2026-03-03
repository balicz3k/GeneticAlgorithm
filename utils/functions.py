import math
from typing import List

def sphere_function(variables: List[float]) -> float:
    return sum(x ** 2 for x in variables)

def hyperellipsoid_function(variables: List[float]) -> float:
    total_sum = 0.0
    for i in range(len(variables)):
        total_sum += sum(variables[j] ** 2 for j in range(i + 1))
    return total_sum

def schwefel_function(variables: List[float]) -> float:
    n = len(variables)
    return 418.9829 * n - sum(x * math.sin(math.sqrt(abs(x))) for x in variables)

def ackley_function(variables: List[float]) -> float:
    n = len(variables)
    a, b, c = 20, 0.2, 2 * math.pi
    sum_sq = sum(x ** 2 for x in variables)
    sum_cos = sum(math.cos(c * x) for x in variables)
    term1 = -a * math.exp(-b * math.sqrt(sum_sq / n))
    term2 = -math.exp(sum_cos / n)
    return term1 + term2 + a + math.e

def michalewicz_function(variables: List[float]) -> float:
    m = 10
    total = 0.0
    for i, x in enumerate(variables):
        total += math.sin(x) * (math.sin(x**2 * (i + 1) / math.pi) ** (2 * m))
    return -total

def rastrigin_function(variables: List[float]) -> float:
    n = len(variables)
    return 10 * n + sum(x**2 - 10 * math.cos(2 * math.pi * x) for x in variables)

def rosenbrock_function(variables: List[float]) -> float:
    n = len(variables)
    return sum(100 * (variables[i+1] - variables[i]**2)**2 + (variables[i] - 1)**2 for i in range(n - 1))

def dejong3_function(variables: List[float]) -> float:
    return sum(math.floor(x) for x in variables)

def dejong5_function(variables: List[float]) -> float:
    if len(variables) != 2:
        raise ValueError("De Jong 5 function requires exactly 2 variables")
    
    a1 = [-32, -16, 0, 16, 32] * 5
    a2 = []
    for val in [-32, -16, 0, 16, 32]:
        a2.extend([val]*5)
        
    term_sum = sum(1.0 / (i + 1 + (variables[0] - a1[i])**6 + (variables[1] - a2[i])**6) for i in range(25))
    return (0.002 + term_sum)**(-1)

def martin_and_gaddy_function(variables: List[float]) -> float:
    if len(variables) != 2:
        raise ValueError("Martin & Gaddy function requires exactly 2 variables")
    return (variables[0] - variables[1]) ** 2 + ((variables[0] + variables[1] - 10) / 3) ** 2

def griewank_function(variables: List[float]) -> float:
    sum_term = sum(x**2 / 4000.0 for x in variables)
    prod_term = 1.0
    for i, x in enumerate(variables):
        prod_term *= math.cos(x / math.sqrt(i + 1))
    return sum_term - prod_term + 1.0

def easom_function(variables: List[float]) -> float:
    if len(variables) != 2:
        raise ValueError("Easom function requires exactly 2 variables")
    x1, x2 = variables
    return -math.cos(x1) * math.cos(x2) * math.exp(-(x1 - math.pi)**2 - (x2 - math.pi)**2)

def goldstein_price_function(variables: List[float]) -> float:
    if len(variables) != 2:
        raise ValueError("Goldstein and Price function requires exactly 2 variables")
    x, y = variables
    term1 = 1 + ((x + y + 1)**2) * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
    term2 = 30 + ((2*x - 3*y)**2) * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
    return term1 * term2

def picheny_goldstein_price_function(variables: List[float]) -> float:
    if len(variables) != 2:
        raise ValueError("Picheny, Goldstein & Price requires exactly 2 variables")
    gp_val = goldstein_price_function(variables)
    if gp_val <= 0: return float('inf')  
    return (1.0 / 2.427) * (math.log10(gp_val) - 8.693)

def styblinski_tang_function(variables: List[float]) -> float:
    return 0.5 * sum(x**4 - 16*x**2 + 5*x for x in variables)

def mccormick_function(variables: List[float]) -> float:
    if len(variables) != 2:
        raise ValueError("McCormick function requires exactly 2 variables")
    x, y = variables
    return math.sin(x + y) + (x - y)**2 - 1.5*x + 2.5*y + 1

def rana_function(variables: List[float]) -> float:
    n = len(variables)
    res = 0.0
    for i in range(n - 1):
        x_i = variables[i]
        x_next = variables[i+1]
        t1 = math.sqrt(abs(x_next + x_i + 1))
        t2 = math.sqrt(abs(x_next - x_i + 1))
        res += x_i * math.cos(t1) * math.sin(t2) + (1 + x_next) * math.sin(t1) * math.cos(t2)
    return res

def eggholder_function(variables: List[float]) -> float:
    n = len(variables)
    res = 0.0
    for i in range(n - 1):
        x = variables[i]
        y = variables[i+1]
        res += -(y + 47) * math.sin(math.sqrt(abs(y + x/2 + 47))) - x * math.sin(math.sqrt(abs(x - (y + 47))))
    return res

def keane_function(variables: List[float]) -> float:
    n = len(variables)
    sum_cos4 = sum(math.cos(x)**4 for x in variables)
    
    prod_cos2 = 1.0
    for x in variables:
        prod_cos2 *= math.cos(x)**2
        
    num = abs(sum_cos4 - prod_cos2)
    den = math.sqrt(sum( (i+1)*(variables[i]**2) for i in range(n) ))
    if den == 0:
        return 0.0 
    return -num / den

def schaffer2_function(variables: List[float]) -> float:
    if len(variables) != 2:
        raise ValueError("Schaffer 2 function requires exactly 2 variables")
    x, y = variables
    sq = x**2 + y**2
    return 0.5 + (math.sin(math.sqrt(sq))**2 - 0.5) / ((1 + 0.001*sq)**2)

def himmelblau_function(variables: List[float]) -> float:
    if len(variables) != 2:
        raise ValueError("Himmelblau function requires exactly 2 variables")
    x, y = variables
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def pits_and_holes_function(variables: List[float]) -> float:
    if len(variables) != 2:
        raise ValueError("Pits and Holes function requires exactly 2 variables")
    x, y = variables
    optima = [
        (0,0,0.0303), (20,0,0.0284), (0,20,0.0268), 
        (-20,0,0.0331), (0,-20,0.0406), (10,10,0.0796), 
        (-10,-10,0.2387), (-10,10,0.1528), (10,-10,0.2153)
    ]
    val = 0.0
    for ox, oy, depth in optima:
        dist_sq = (x - ox)**2 + (y - oy)**2
        val -= depth * math.exp(-0.25 * dist_sq)
    return val

AVAILABLE_FUNCTIONS = {
    "1. Hypersphere": sphere_function,
    "2. Hyperellipsoid": hyperellipsoid_function,
    "3. Schwefel": schwefel_function,
    "4. Ackley": ackley_function,
    "5. Michalewicz": michalewicz_function,
    "6. Rastrigin": rastrigin_function,
    "7. Rosenbrock": rosenbrock_function,
    "8. De Jong 3": dejong3_function,
    "9. De Jong 5": dejong5_function,
    "10. M & G": martin_and_gaddy_function,
    "11. Griewank": griewank_function,
    "12. Easom": easom_function,
    "13. G & P": goldstein_price_function,
    "14. P, G & P": picheny_goldstein_price_function,
    "15. S & T": styblinski_tang_function,
    "16. McCormick": mccormick_function,
    "17. Rana": rana_function,
    "18. Egg Holder": eggholder_function,
    "19. Keane": keane_function,
    "20. Schaffer 2": schaffer2_function,
    "21. Himmelblau": himmelblau_function,
    "22. Pits & Holes": pits_and_holes_function,
}
