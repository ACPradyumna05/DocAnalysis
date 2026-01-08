# Make sure folder exists
New-Item -ItemType Directory -Force -Path "C:\models\mistral-7b"

Set-Location "C:\models\mistral-7b"

function DL {
    param([string]$url, [string]$name)

    aria2c `
        --continue=true `
        --max-connection-per-server=16 `
        --split=16 `
        --min-split-size=2M `
        --timeout=120 `
        --retry-wait=5 `
        --max-tries=0 `
        --file-allocation=none `
        -o $name `
        $url
}

DL "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/model-00001-of-00002.safetensors" "model-00001-of-00002.safetensors"
DL "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/model-00002-of-00002.safetensors" "model-00002-of-00002.safetensors"
DL "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/model.safetensors.index.json" "model.safetensors.index.json"
DL "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/config.json" "config.json"
DL "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.json" "tokenizer.json"
DL "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer_config.json" "tokenizer_config.json"
DL "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/special_tokens_map.json" "special_tokens_map.json"
DL "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/generation_config.json" "generation_config.json"
DL "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/vocab.json" "vocab.json"
DL "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/merges.txt" "merges.txt"
