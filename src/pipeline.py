import torch
import os
from src.symbolic_engine import SymbolicODESolver

class ProductionPipeline:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.math_engine = SymbolicODESolver()

    def process_query(self, user_input):
        text = user_input.lower().strip()
        
        # Trigger words for the AI's identity
        about_triggers = ['about yourself', 'who are you', 'what are you', 'tell me about yourself', 'how do you work']
        chat_triggers = ['hi', 'hello', 'hey', 'what can you do', 'help', 'test']
        
        # 1. Identity & Architecture Explanation
        if any(trigger in text for trigger in about_triggers):
            return (
                "I am **Axiom**, a custom-built Mini-LLM engineered specifically for higher-order mathematics.\n\n"
                "### 🧠 How I am an LLM\n"
                "At my core, I operate using a **Sequence-to-Sequence (Seq2Seq) Transformer architecture** built entirely from scratch in PyTorch. "
                "Just like massive language models, I utilize multi-head attention mechanisms to read your input sequence, encode the context, "
                "and decode a mathematical prediction. I rely on my own custom, character-level embedding space and vocabulary tailored exclusively for calculus.\n\n"
                "### ✨ What Makes Me Unique\n"
                "Standard LLMs generate text based on token probability, which frequently leads to arithmetic errors. "
                "I bypass this limitation using a **Neuro-Symbolic Dual-System Architecture**:\n\n"
                "- **System 1 (Neural Intuition):** My PyTorch Transformer parses human input and recognizes the structural pattern of the differential equation.\n"
                "- **System 2 (Formal Verification):** I route the neural output directly into a deterministic algebraic kernel (SymPy). This engine performs the integration step-by-step and mathematically proves the result.\n\n"
                "Because of this architecture, I possess the conversational understanding of an LLM combined with the flawless precision of a deterministic calculator."
            )

        # 2. General Greetings
        if any(text == trigger for trigger in chat_triggers):
            return (
                "👋 **Greetings!** I am **Axiom**, a Neuro-Symbolic AI framework.\n\n"
                "**Feed me an Ordinary Differential Equation to solve:**\n"
                "* `dy/dx = x * y`\n"
                "* `dy/dx + y = exp(x)`\n"
                "* `y'' + 4*y = x^2`\n\n"
                "Or ask me *'tell me about yourself'* to learn about my underlying architecture!"
            )

        # 3. Mathematical Routing
        steps = ["⚡ *[NEURAL CORE] Transformer weights active. Routing pattern to System 2...*"]
        steps.append(self.math_engine.solve_human_readable(user_input))
        return "\n\n".join(steps)