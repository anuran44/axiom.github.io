import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

class SymbolicODESolver:
    def __init__(self):
        self.x = sp.Symbol('x')
        self.y = sp.Function('y')(self.x)

    def parse_input(self, eq_str):
        s = eq_str.strip().replace("^", "**") 
        s = s.replace("dy/dx", "DerivOne")
        s = s.replace("y''", "DerivTwo")
        s = s.replace("y'", "DerivOne")

        if '=' in s:
            lhs_str, rhs_str = s.split('=', 1)
        else:
            lhs_str, rhs_str = s, '0'

        context = {
            'x': self.x, 
            'y': self.y,
            'DerivOne': self.y.diff(self.x), 
            'DerivTwo': self.y.diff(self.x, 2),
            'exp': sp.exp, 'e': sp.E, 'sin': sp.sin, 'cos': sp.cos
        }

        transformations = standard_transformations + (implicit_multiplication_application,)
        lhs = parse_expr(lhs_str, local_dict=context, transformations=transformations)
        rhs = parse_expr(rhs_str, local_dict=context, transformations=transformations)
        return sp.Eq(lhs, rhs)

    def solve_human_readable(self, eq_str):
        try:
            ode = self.parse_input(eq_str)
            hints = sp.classify_ode(ode, self.y)
            method = hints[0].replace('_', ' ').title() if hints else "Standard Integration"
            solution = sp.dsolve(ode, self.y)

            ode_latex = sp.latex(ode)
            sol_latex = sp.latex(solution)
            integral_latex = sp.latex(sp.Integral(ode.lhs - ode.rhs, self.x))

            steps = [
                "### 🔍 Phase 1: Pattern Identification",
                f"**Mathematical Classification:** `{method}`",
                f"$$ {ode_latex} $$",
                "---",
                "### ⚙️ Phase 2: Symbolic Integration",
                f"$$ {integral_latex} = C $$",
                "---",
                "### ✨ Phase 3: Final Explicit Solution",
                f"$$ {sol_latex} $$"
            ]

            if self.verify_solution(ode, solution):
                steps.append("\n\n✅ **System Verification:** Solution back-propagation returns the original equality natively.")
            else:
                steps.append("\n\n⚠️ **System Verification:** Automatic differentiation flag raised.")

            return "\n".join(steps)
        except Exception as e:
            return f"❌ **Syntax Error:** My parser encountered an issue. Details: {str(e)}"

    def verify_solution(self, ode, solution):
        try:
            lhs = ode.lhs.subs(self.y, solution.rhs).doit()
            rhs = ode.rhs.subs(self.y, solution.rhs).doit()
            return sp.simplify(lhs - rhs) == 0
        except:
            return False