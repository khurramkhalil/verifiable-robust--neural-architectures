import argparse
import yaml
import torch
import os
import sys
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.pgd_training import train_pgd_at
from src.verification.abcrown_interface import ABCROWNVerifier
from src.primitives.lipschitz_layers import compute_architectural_lipschitz_bound
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Changed from F.avg_pool2d for ONNX compatibility
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)  # Use layer instead of functional
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet20():
    return ResNet(BasicBlock, [3, 3, 3])


from dotenv import load_dotenv
import wandb

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50) # Standard is often 200, use 50 for faster baseline
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Init WandB
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "Verifiable-Robust-NAS-Paper"),
        entity=os.getenv("WANDB_ENTITY"),
        name=f"baseline_resnet20_seed_{args.seed}",
        tags=["baseline", "resnet20"],
        config=args
    )
    
    print(f"--- Standard Robust Baseline (ResNet-20) ---")
    
    model = ResNet20()
    model = model.to(device)
    
    # 2. Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])
    
    os.makedirs('./data', exist_ok=True)
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    if args.debug:
         from itertools import islice
         train_loader = islice(train_loader, 2)
         args.epochs = 1
    
    # 3. PGD Training
    print(f"Starting PGD Training ({args.epochs} epochs)...")
    model, history = train_pgd_at(
        model, train_loader, val_loader, 
        epochs=args.epochs,
        epsilon=8/255, 
        device=device
    )
    
    # Save
    os.makedirs('results/baselines', exist_ok=True)
    save_path = f"results/baselines/resnet20_seed_{args.seed}.pt"
    torch.save(model.state_dict(), save_path)
    wandb.save(save_path)
    
    # 4. Verified Accuracy
    print("Starting Verification...")
    verifier = ABCROWNVerifier(verifier_path='alpha-beta-CROWN')
    
    # Export ONNX
    onnx_path = f"results/baselines/resnet20_seed_{args.seed}.onnx"
    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model.cpu(), dummy, onnx_path, verbose=False)
    model.to(device)
    
    epsilons = [2/255, 4/255, 8/255]
    results = {}
    
    for eps in epsilons:
        eps_str = f"{int(eps*255)}/255"
        print(f"  - eps={eps_str}...", end="", flush=True)
        
        # Verify
        res = verifier.verify(model, (3,32,32), epsilon=eps, num_samples=100 if not args.debug else 10)
        results[eps_str] = res['certified_accuracy']
        print(f" Acc={res['certified_accuracy']:.2%}")
        
    wandb.log({
        "certified_acc_2_255": results['2/255'],
        "certified_acc_4_255": results['4/255'],
        "certified_acc_8_255": results['8/255']
    })
    
    # --- Logging Update: Save Standardized CSV ---
    final_acc = history['val_acc'][-1] * 100.0 # Convert to percentage
    try:
        lip = compute_architectural_lipschitz_bound(model)
    except:
        lip = 0.0
        
    table_data = [[
        "ResNet-20 Baseline", 
        -1, 
        lip, 
        0.0, 
        final_acc, 
        results['2/255'], 
        results['4/255'], 
        results['8/255']
    ]]
    
    columns = ["Method", "Arch Index", "Lipschitz", "SynFlow", "Test Acc", "Cert Acc (2/255)", "Cert Acc (4/255)", "Cert Acc (8/255)"]
    df = pd.DataFrame(table_data, columns=columns)
    
    csv_path = f"results/baselines/resnet20_seed_{args.seed}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    wandb.save(csv_path)
    
    print("Baseline Complete.")
    wandb.finish()

if __name__ == '__main__':
    main()
