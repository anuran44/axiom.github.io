import torch
import os
from src.symbolic_engine import SymbolicODESolver

class NeuroSymbolicPipeline:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.math_engine = SymbolicODESolver()
        
        weight_path = "weights/model.pth"
        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
            self.model.eval()
            self.model_loaded = True
        else:
            self.model_loaded = False

    def process_query(self, user_input):
        steps = []
        if self.model_loaded:
            steps.append("  [NEURAL SYSTEM] Neural weights loaded. Mathematical Transformer online.")
        else:
            steps.append("  [NEURAL SYSTEM] Warning: Running in Pure Symbolic mode (No weights found).")

        steps.append(self.math_engine.solve_human_readable(user_input))
        return "\n".join(steps)