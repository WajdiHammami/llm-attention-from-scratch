import json
import argparse


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Convert JSONL to TXT")
    argparser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    argparser.add_argument("--output", type=str, required=True, help="Path to output TXT file")
    args = argparser.parse_args()
    print(f"Converting {args.input} to {args.output}...")

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            data = json.loads(line)
            fout.write(data["text"] + "\n")
    print("Conversion completed. Output saved to", args.output)