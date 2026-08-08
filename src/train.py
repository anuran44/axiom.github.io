import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.generate_data import generate_ode_dataset
from src.model import MathTransformer
from src.vocab import MathTokenizer

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    dataset = generate_ode_dataset(10000)
    
    tokenizer = MathTokenizer()
    tokenizer.fit(dataset)
    print(f"Vocabulary built. Size: {tokenizer.vocab_size} tokens.")
    
    model = MathTransformer(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model.train()
    epochs = 5 
    
    print("\n--- Starting Neural Training ---")
    for epoch in range(epochs):
        total_loss = 0
        for ode, sol in dataset[:500]: 
            src = torch.tensor([tokenizer.encode(ode)]).to(device)
            tgt = torch.tensor([tokenizer.encode(sol)]).to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]
            
            optimizer.zero_grad()
            output = model(src, tgt_input)
            
            loss = criterion(output.transpose(1, 2), tgt_expected)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/500:.4f}")

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/model.pth")
    tokenizer.save("weights/vocab.json")
    print("\nSuccess! Model and Vocab saved to 'weights/' directory.")

if __name__ == "__main__":
    train_model()