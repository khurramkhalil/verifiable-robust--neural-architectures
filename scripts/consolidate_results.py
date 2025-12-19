import pandas as pd
import glob
import os
import argparse
import wandb
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()
    
    # 1. Init WandB for Upload
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "Verifiable-Robust-NAS-Paper"),
        entity=os.getenv("WANDB_ENTITY"),
        name=f"consolidation_seed_{args.seed}",
        job_type="consolidation"
    )
    
    print("--- Consolidating Results ---")
    
    all_dfs = []
    
    # Pattern 1: NSGA-II Results (Constrained & Unconstrained)
    nsga_files = glob.glob(os.path.join(args.results_dir, 'nsga2', f'certified_results_seed_{args.seed}*.csv'))
    for f in nsga_files:
        print(f"Found NSGA-II file: {f}")
        try:
            df = pd.read_csv(f)
            # Ensure Method column exists (backward compat logic)
            if 'Method' not in df.columns:
                if 'unconstrained' in f:
                    df['Method'] = 'Unconstrained Search'
                else:
                    df['Method'] = 'Constrained Search'
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    # Pattern 2: Baseline Results
    baseline_files = glob.glob(os.path.join(args.results_dir, 'baselines', f'*seed_{args.seed}.csv'))
    for f in baseline_files:
        print(f"Found Baseline file: {f}")
        try:
            df = pd.read_csv(f)
            # Ensure Method column (ResNet logic applied placeholders in script, but double check)
            if 'Method' not in df.columns:
                 # Infer from filename
                 if 'resnet' in f: df['Method'] = 'ResNet-20 Baseline'
                 elif 'random' in f: df['Method'] = 'Random Baseline'
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not all_dfs:
        print("No result CSVs found. Exiting.")
        return

    # 2. Merge
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # Standardize Column Names (just in case)
    # Expected: ["Method", "Arch Index", "Lipschitz", "SynFlow", "Test Acc", "Cert Acc (2/255)", "Cert Acc (4/255)", "Cert Acc (8/255)"]
    # Check for minor variations if any
    
    master_path = os.path.join(args.results_dir, f'MASTER_RESULTS_RUN_{args.seed}.csv')
    master_df.to_csv(master_path, index=False)
    print(f"✅ Master CSV saved to: {master_path}")
    print(master_df.groupby('Method').size())
    
    # 3. Upload to WandB
    wandb.log({"master_results": wandb.Table(dataframe=master_df)})
    wandb.save(master_path)
    
    # 4. Upload to Hugging Face
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            print("Uploading to Hugging Face...")
            login(token=hf_token)
            api = HfApi()
            username = api.whoami()['name']
            repo_id = f"{username}/verifiable-nas-cifar10-verified"
            api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="dataset") # Use dataset repo? Or model repo? User said "upload... so we have alternate source". 
            # Previous scripts used "model" repo. Let's stick to the same one to keep it simple, or make a new dataset one.
            # Usually better to separate. But user might expect it in the same place. 
            # Let's use the existing model repo for simplicity unless specified.
            
            api.upload_file(
                path_or_fileobj=master_path,
                path_in_repo=f"MASTER_RESULTS_RUN_{args.seed}.csv",
                repo_id=repo_id,
                repo_type="model" 
            )
            print("✅ Uploaded to Hugging Face.")
        except Exception as e:
            print(f"HF Upload failed: {e}")
            
    wandb.finish()

if __name__ == '__main__':
    main()
