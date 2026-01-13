import argparse
import time
from synrxnval import SynRxnVal

def main():
    parser = argparse.ArgumentParser(description="Validate synthetic SMILES reactions.")
    parser.add_argument("--input", required=True, help="Path to input parquet or tuple file.")
    parser.add_argument("--t2_model", required=True, help="Path to reagent prediction model (T2).")
    parser.add_argument("--t3_model", required=True, help="Path to forward prediction model (T3).")
    parser.add_argument("--out_dir", default="./outputs", help="Output directory for results.")
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--chunk_id", type=str, default=None)
    parser.add_argument("--time", action="store_true", help="Measure and print total runtime.")
    args = parser.parse_args()

    syn = SynRxnVal(
        input_dataset_path=args.input,
        T2_path=args.t2_model,
        T3_path=args.t3_model,
        beam_size=args.beam_size,
    )
    
    if args.time:
        start = time.perf_counter()
    
    syn.main(out_dir=args.out_dir, chunk_id=args.chunk_id)

    if args.time:
        end = time.perf_counter()
        elapsed = end - start

        print(f"\n Total runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} min)")


if __name__ == "__main__":
    main()
