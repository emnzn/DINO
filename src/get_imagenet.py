import os

from dotenv import load_dotenv
from datasets import load_dataset


def main():
    load_dotenv(os.path.join("..", ".env"))

    hf_token = os.getenv("HF_TOKEN")
    cache_dir = os.path.join("..", "data")
    os.environ["HF_HOME"] = cache_dir
    
    load_dataset(
        "ILSVRC/imagenet-1k",
        num_proc=8, 
        token=hf_token,
        cache_dir=cache_dir, 
        )

if __name__ == "__main__":
    main()