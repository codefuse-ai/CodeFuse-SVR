export PYTHONPATH=$PYTHONPATH:$(pwd)

INPUT_DATA="/path/to/input_data.json"
IMAGE_DIR="/path/to/image_dir"
OUTPUT_DIR="/path/to/output_dir"

# 0. download image
python src/utils/download_image.py --input_data="${INPUT_DATA}" --image_dir="${IMAGE_DIR}"

# 1. full-run
export OPENAI_API_KEY="agent-api-key"
export VLM_API_KEY="vlm-api-key"

python main.py full-run \
    --input_data="${INPUT_DATA}" --output_dir="${OUTPUT_DIR}" --image_dir="${IMAGE_DIR}" \
    --repo_path /path/to/repo \
    --vlm_model Qwen2-VL-7B-Instruct --vlm_url https://xxxx.com/v1/ \
    --max_workers 3 --model_name Kimi-K2-Instruct-0905 --base_url https://xxxxx.com/v1/ --temperature 0.0