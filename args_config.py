import argparse

def parse_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Configuration Parameters of the FairGame Evaluation")

    # Model name argument (short: -m, long: --model)
    parser.add_argument(
        "-m", "--model",
        dest="model_name",
        type=str,
        default="deepseekv3",
        help="Model Name (default: deepseekv3)"
    )
    parser.add_argument(
        "-dt", "--data_type",
        dest="data_type",
        type=str,
        default="real",
        help="Data Type, you can chose 'real' or 'virtual' (default: 'real')"
    )

    return parser.parse_args()

# Test the argument parser if run directly
if __name__ == "__main__":
    args = parse_args()
    print("[Debug] Parsed arguments:")
    print(f"Model: {args.model_name}, Learning Rate: {args.lr}")
