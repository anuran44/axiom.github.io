import sympy as sp
import random

def generate_ode_dataset(num_samples=1000):
    print(f"Generating {num_samples} mathematical equations...")
    x = sp.Symbol('x')
    C1 = sp.Symbol('C1')
    
    dataset = []
    funcs = [x, x**2, sp.sin(x), sp.cos(x), sp.exp(x)]
    
    for _ in range(num_samples):
        f_x = random.choice(funcs) + random.choice([1, 2]) * random.choice(funcs)
        solution = f_x + C1
        dy_dx = sp.diff(f_x, x)
        
        ode_str = f"dy/dx = {str(dy_dx)}"
        sol_str = f"y = {str(solution)}"
        dataset.append((ode_str, sol_str))
        
    return dataset

if __name__ == "__main__":
    data = generate_ode_dataset(5)
    for ode, sol in data:
        print(f"Input: {ode:25} | Target: {sol}")