import datasets
import random
from datasets import concatenate_datasets, Dataset


def create_mixed_dataset():
    print("Loading datasets...")

    # Load ShareGPT dataset
    print("Loading ShareGPT dataset...")
    sharegpt_dataset = datasets.load_dataset(
        "json", data_files="ShareGPT_V4.3_unfiltered_cleaned_split.json", split="train"
    )
    print(f"ShareGPT dataset size: {len(sharegpt_dataset)}")

    # Load UltraChat dataset
    print("Loading UltraChat dataset...")
    ultrachat_dataset = datasets.load_dataset("parquet", data_dir="ultrachat_200k", split="train_sft")
    print(f"UltraChat dataset size: {len(ultrachat_dataset)}")

    # Load OpenThoughts2-1M dataset
    print("Loading OpenThoughts2-1M dataset...")
    openthoughts_dataset = datasets.load_dataset(
        "parquet", data_dir="OpenThoughts2-1M", split="train"
    )
    print(f"OpenThoughts2-1M dataset size: {len(openthoughts_dataset)}")

    # Sample from OpenThoughts2-1M
    target_openthoughts_size = 250000
    if len(openthoughts_dataset) > target_openthoughts_size:
        print(f"Sampling {target_openthoughts_size} examples from OpenThoughts2-1M...")
        # Set seed for reproducibility
        openthoughts_dataset = openthoughts_dataset.shuffle(seed=42)
        openthoughts_dataset = openthoughts_dataset.select(range(target_openthoughts_size))

    # Convert all datasets to OpenThoughts format by creating new datasets
    print("Converting datasets to OpenThoughts format...")

    # Convert ShareGPT dataset
    print("Converting ShareGPT dataset...")
    sharegpt_data = []
    for example in sharegpt_dataset:
        conversations = []
        for conv in example["conversations"]:
            # Convert human -> user, gpt -> assistant
            from_role = "user" if conv["from"] == "human" else "assistant"
            conversation_turn = {"from": from_role, "value": conv["value"]}
            conversations.append(conversation_turn)

        sharegpt_data.append({"conversations": conversations, "source": "ShareGPT"})

    # Create new dataset from the converted data
    sharegpt_dataset_new = Dataset.from_list(sharegpt_data)

    # Convert UltraChat dataset
    print("Converting UltraChat dataset...")
    ultrachat_data = []
    for example in ultrachat_dataset:
        conversations = []
        for msg in example["messages"]:
            conversation_turn = {"from": "user" if msg["role"] == "user" else "assistant", "value": msg["content"]}
            conversations.append(conversation_turn)

        ultrachat_data.append({"conversations": conversations, "source": "UltraChat"})

    # Create new dataset from the converted data
    ultrachat_dataset_new = Dataset.from_list(ultrachat_data)

    # Convert OpenThoughts2-1M dataset
    print("Converting OpenThoughts2-1M dataset...")
    openthoughts_data = []
    for example in openthoughts_dataset:
        conversations = []
        for conv in example["conversations"]:
            conversation_turn = {"from": conv["from"], "value": conv["value"]}
            conversations.append(conversation_turn)

        openthoughts_data.append({"conversations": conversations, "source": "OpenThoughts2"})

    # Create new dataset from the converted data
    openthoughts_dataset_new = Dataset.from_list(openthoughts_data)

    print("Dataset sizes after processing:")
    print(f"ShareGPT: {len(sharegpt_dataset_new)}")
    print(f"UltraChat: {len(ultrachat_dataset_new)}")
    print(f"OpenThoughts2-1M (sampled): {len(openthoughts_dataset_new)}")

    print("Dataset features:")
    print(f"ShareGPT features: {sharegpt_dataset_new.features}")
    print(f"UltraChat features: {ultrachat_dataset_new.features}")
    print(f"OpenThoughts2 features: {openthoughts_dataset_new.features}")

    # Combine datasets
    print("Combining datasets...")
    mixed_dataset = concatenate_datasets([sharegpt_dataset_new, ultrachat_dataset_new, openthoughts_dataset_new])

    print(f"Eagle-mix dataset total size: {len(mixed_dataset)}")

    # Shuffle the combined dataset
    print("Shuffling combined dataset...")
    mixed_dataset = mixed_dataset.shuffle(seed=42)

    # Save the mixed dataset
    output_dir = "~/dataset/eagle-mix"
    print(f"Saving eagle-mix dataset to {output_dir}...")
    mixed_dataset.save_to_disk(output_dir)

    print("Eagle-mix dataset creation completed!")

    # Print some statistics
    print("\nDataset composition:")
    source_counts = {}
    for example in mixed_dataset:
        source = example["source"]
        source_counts[source] = source_counts.get(source, 0) + 1

    for source, count in source_counts.items():
        percentage = (count / len(mixed_dataset)) * 100
        print(f"{source}: {count:,} samples ({percentage:.1f}%)")

    print(f"\nTotal: {len(mixed_dataset):,} samples")

    # Show sample from each source
    print("\nSample examples:")
    for source in source_counts.keys():
        print(f"\n--- {source} sample ---")
        sample = [ex for ex in mixed_dataset if ex["source"] == source][0]
        print(f"Keys: {list(sample.keys())}")
        print(f"Conversations length: {len(sample['conversations'])}")
        print(f"First conversation turn: {sample['conversations'][0]}")
        print(f"Source: {sample['source']}")


if __name__ == "__main__":
    create_mixed_dataset()
