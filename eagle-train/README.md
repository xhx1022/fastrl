<h1 style="text-align: center;">Eagle-Train</h1>

# Installation

```bash
conda create --name eagle python=3.10
conda activate eagle
pip install -e .
```

Install flash_attn
```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```


# Step 1: (Optional) Create Mixed DataSet

Create a dataset for Eagle warm-up training by mixing multiple datasets. You can customize the mixing ratio and dataset source.

```bash
python dataset/create_mixed_dataset.py
```


Download processed dataset example:

```bash
huggingface-cli download Qinghao/eagle-mix --repo-type dataset --local-dir /path/to/your/directory
```


# Step 2: (Eagle-3 Only) Generate Frequency Mapping

Generate `d2t` and `t2d` mapping for the given tokenizer and mixed dataset.

```bash
python freq_map/generate_freq.py
```

# Step 3: Cache Hidden States

Cache hidden states for offline training. We support both Eagle-2 and Eagle-3 modes.

```bash
srun -J datagen -N 1 --exclusive bash scripts/datagen_eagle2.sh
```


# Step 4: Train

You can customize the training setting in `scripts/train_eagle2.sh` and `scripts/train_eagle3.sh`. For example, the pre-set epoch number is a large value, you can stop the training when the model converges.

```bash
srun -J eagle -N 1 --exclusive bash scripts/train_eagle2.sh
```


