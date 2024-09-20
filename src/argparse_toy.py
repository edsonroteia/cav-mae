import argparse

def main():
    parser = argparse.ArgumentParser(description='Toy example for argument parsing')
    
    parser.add_argument('--dataset', type=str, choices=['audioset', 'vggsound'], 
                        help='Dataset to use for retrieval')
    parser.add_argument('--model_type', type=str, choices=['sync_pretrain', 'pretrain', 'finetune'], 
                        help='Type of model to use')
    parser.add_argument('--strategy', type=str, 
                        help='Strategy for aggregation')
    parser.add_argument('--directions', type=str, nargs='+', 
                        help='Directions for evaluation')
    parser.add_argument('--nums_samples', type=int, nargs='+', 
                        help='Number of samples to test')

    args = parser.parse_args()

    # Print out the parsed arguments
    print(f"Dataset: {args.dataset}")
    print(f"Model Type: {args.model_type}")
    print(f"Strategy: {args.strategy}")
    print(f"Directions: {args.directions}")
    print(f"Number of Samples: {args.nums_samples}")

if __name__ == "__main__":
    main()
