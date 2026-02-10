import argparse
import glob
import json
import re
import os
import traceback
from typing import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import mimetypes
import io
from jinja2 import Template
from PIL import Image
from urllib.parse import urlparse
from prompt.image_prompt import VLMPrompt
from src.utils.llm_client import send_chat_completion
from src.utils.image_utils import download_image_from_url, pil_image_to_base64


def parse_deleted_blocks(patch_content: str) -> Dict[str, List[List[str]]]:
    """
    Parse diff, extract filenames and consecutive deletion blocks

    Returns:
        Dict: {filename: [[consecutive deletion block 1], [consecutive deletion block 2], ...]}
    """
    result = {}
    current_file = None
    current_block = []

    lines = patch_content.split('\n')

    for line in lines:
        # Extract modified filename
        if line.startswith('--- a/'):
            # Save the last block of previous file
            if current_file and current_block:
                if current_file not in result:
                    result[current_file] = []
                result[current_file].append(current_block)
                current_block = []

        # Extract new filename
            current_file = line[6:]  # Remove "--- a/"

        elif line.startswith('+++ b/'):
        # Can also extract filename from here (modified filename)
            # Usually same as --- a/, skip here
            continue

        # Extract deleted lines (starting with - but not ---)
        elif line.startswith('-') and not line.startswith('---'):
        # Remove leading - sign
            deleted_line = line[1:]
            current_block.append(deleted_line)

        # Encounter non-deletion line, indicating current consecutive block ends
        else:
            if current_block:
                if current_file not in result:
                    result[current_file] = []
                result[current_file].append(current_block)
                current_block = []

        # Save the last block
    if current_file and current_block:
        if current_file not in result:
            result[current_file] = []
        result[current_file].append(current_block)

    return result


class ImageCodeLocalization:

    def __init__(self, model_name: str, base_url: str):
        self.vlm_model = model_name
        self.base_url = base_url
        self.api_key = os.getenv("VLM_API_KEY", "")
        self.pattern = r"<result>(.*?)</result>"

    @staticmethod
    def convert_coordinate(bbox: List[int], coef: List[float]):
        x, y, width, height = bbox
        x_ratio, y_ratio = coef

        # 2. Apply scaling ratio to each coordinate of bbox
        x1 = x * x_ratio
        y1 = y * y_ratio
        _w = width * x_ratio
        _h = height * y_ratio
        return [int(x1), int(y1), int(_w), int(_h)]

    def parse_response(self, resp: str):
        match = re.search(self.pattern, resp)
        if not match:
            return None

        result = json.loads(match.group(1))
        print("bbox result: ", result)
        return result


    @staticmethod
    def localize_code(patch_file: str) -> Dict[str, List[List[str]]]:

        with open(patch_file, "r") as infile:
            patch_content = infile.read()

        return parse_deleted_blocks(patch_content)


    def localize_image(self, problem_statement: str, code_snips: str, image_path: str, instance_dir: str, min_resolution: int = 500*500):

        image = Image.open(image_path)
        width, height = image.size
        if width * height <= min_resolution:
            return None

        # img_str = pil_image_to_base64(image)
        is_gif = image_path.lower().endswith('.gif') or 'gif' in image_path.lower()
           
        if is_gif:
            # Handle GIF - extract only first frame
            img = Image.open(image_path)
            # Convert to RGB mode and extract first frame
            img = img.convert('RGB')
            output_buffer = io.BytesIO()
            img.save(output_buffer, format='JPEG')
            img_str = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
            img.close()
            img_str = f"data:image/jpeg;base64,{img_str}"
        else:
            # Handle static images
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                img_str = base64.b64encode(img_data).decode('utf-8')
                
                # Get correct MIME type based on file extension
                mime_type, _ = mimetypes.guess_type(image_path)
                if not mime_type or not mime_type.startswith('image/'):
                    mime_type = 'image/jpeg'
                img_str = f"data:{mime_type};base64,{img_str}"

        print("image_str: ", img_str[:30])
        print("image size: ", width, height)

        user_prompt = Template(VLMPrompt).render({
            "resolution": f"{width}x{height}",
            "problem_statement": problem_statement,
            "code_snips": code_snips
        })

        resp = send_chat_completion(
                api_key=self.api_key,
                system_prompt="You are a helpful assistant. ",
                user_prompt=user_prompt,
                image_url=img_str,
                model_name=self.vlm_model,
                base_url=self.base_url
            )["choices"][0]["message"]["content"]
        print("model response: ", resp)
        bbox = self.parse_response(resp)
        if bbox is not None:
            bbox = self.convert_coordinate(bbox, [width/1000, height/1000])
            x1, y1, _w, _h = bbox
            cropped_image = image.crop((x1, y1, x1+_w, y1+_h))
        else:
            return None

        image_name = os.path.basename(image_path)
        cropped_path = os.path.join(instance_dir, f"cropped_{image_name}")
        cropped_image.save(cropped_path)
        print(f"cropped image saved: ", cropped_path)
        return cropped_path

    def process_instance(self, instance_dir: str, image_dir: str):
        # step 0. when this instance cannot be resolved and image larger than 500*500,

        # self.instance_preprocess()
        instance_id = os.path.basename(instance_dir)

        # step 1. use pycfuse to localize bug related code
        patch_file = glob.glob(os.path.join(instance_dir, "*.patch"))
        if not patch_file:
            return None
        code_snips = self.localize_code(patch_file[0])  # kimi patch
        # step 2. use vlm to crop input image
        instance_file = glob.glob(os.path.join(instance_dir, "*.json"))
        if not instance_file:
            return None

        with open(instance_file[0], "r") as infile:
            doc = json.load(infile)

        problem_statement = doc["problem_statement"]
        image_url = json.loads(doc["image_assets"]).get("problem_statement", [])[0]
        parsed_url = urlparse(image_url)
        filename = os.path.basename(parsed_url.path)
        image_file = os.path.join(image_dir, f"{instance_id}_{filename}")
        cropped_path = self.localize_image(problem_statement, json.dumps(code_snips, indent=4), image_file, instance_dir)
        return cropped_path


    def process_batch(self, data_path: str, result_path: str, image_dir: str, max_workers: int = 4):
        with open(data_path, "r") as infile:
            data_dict = json.load(infile)

        # Collect instances to be processed
        instance_dirs = []
        for instance_id in os.listdir(result_path):
            if not os.path.isdir(os.path.join(result_path, instance_id)):
                continue
            if instance_id not in data_dict:
                continue
            instance_dirs.append(os.path.join(result_path, instance_id))

        cropped_instance = []

        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_instance = {
                executor.submit(self.process_instance, instance_dir, image_dir): instance_dir 
                for instance_dir in instance_dirs
            }
            
            # Collect results
            for future in as_completed(future_to_instance):
                instance_dir = future_to_instance[future]
                instance_id = os.path.basename(instance_dir)
                try:
                    cropped_path = future.result()
                    if cropped_path is not None:
                        cropped_instance.append(instance_id)
                        print(f"✓ Successfully processed instance: {instance_id}")
                    else:
                        print(f"- Skipped instance: {instance_id}")
                except Exception as e:
                    print(f"✗ Failed to process instance {instance_id}: {e}")
                    traceback.print_exc()

        # Save results
        with open(os.path.join(result_path, "swebench_image_cropped_instance.json"), "w") as outf:
            outf.write(json.dumps(cropped_instance))
            
        print(f"Total processed {len(instance_dirs)} instances, successfully cropped {len(cropped_instance)} instances")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--repo_path", default="data/swe_bench_mm/repos")
    parser.add_argument("--model_name", default="Qwen3-VL-235B-A22B-Instruct", required=True, help="VLM")
    parser.add_argument("--base_url", required=True, help="Base URL for API")
    args = parser.parse_args()


    fialed_data_path = os.path.join(args.result_path, "all_validation_failed_instance.json")
    localization = ImageCodeLocalization(args.model_name, args.base_url)
    localization.process_batch(fialed_data_path, args.result_path, args.image_dir)


if __name__ == "__main__":
    # Main execution code
    main()