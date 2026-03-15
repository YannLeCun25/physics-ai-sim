from pinn_solver import PINN
import torch

def main():
    model = PINN()
    # Dummy coordinates (x, t)
    x = torch.tensor([[0.5]], dtype=torch.float32)
    t = torch.tensor([[0.1]], dtype=torch.float32)
    output = model(x, t)
    print(f"Physics-AI Simulation Result: {output.item()}")

if __name__ == "__main__":
    main()
