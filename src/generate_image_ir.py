import argparse
import os
import json
import glob
import sys
import traceback
import time
import base64
import mimetypes
import requests
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import tqdm
from PIL import Image
from torch.optim.optimizer import required

from prompt.svr_prompt import SVR_PROMPT
from src.utils.llm_client import send_chat_completion
from src.utils.image_utils import image_url_to_base64, image_to_base64
from urllib.parse import urlparse

def process_image_url_to_base64(img_url, image_dir: str = "", instance_id: str = ""):
    """Process image URL and return base64 encoded string. Return None if local file does not exist"""
    try:
        # Determine if img_url is URL or local file path
        parsed_url = urlparse(img_url)
        
        if parsed_url.scheme in ('http', 'https', 'ftp', 'ftps'):
            # If it's a URL, process as before: parse filename from URL and build local path
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = f"{instance_id}.jpg"
            img_path = os.path.join(image_dir, f"{instance_id}_{filename}")
        else:
            # If it's a local file, use the path directly
            img_path = img_url

        # Check if local file exists, skip if not
        if not os.path.exists(img_path):
            print(f"Local file does not exist, skipping: {img_path}")
            return None

        is_gif = img_path.lower().endswith('.gif') or 'gif' in img_path.lower()
           
        if is_gif:
            # Handle GIF - extract only first frame
            img = Image.open(img_path)
            # Convert to RGB mode and extract first frame
            img = img.convert('RGB')
            output_buffer = io.BytesIO()
            img.save(output_buffer, format='JPEG')
            img_str = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
            img.close()
            return f"data:image/jpeg;base64,{img_str}"
        else:
            # Handle static images
            with open(img_path, 'rb') as img_file:
                img_data = img_file.read()
                img_str = base64.b64encode(img_data).decode('utf-8')
                
                # Get correct MIME type based on file extension
                mime_type, _ = mimetypes.guess_type(img_path)
                if not mime_type or not mime_type.startswith('image/'):
                    mime_type = 'image/jpeg'
                
                return f"data:{mime_type};base64,{img_str}"

    except Exception as e:
        print(f"Failed to process image URL: {img_url} - {str(e)}")
        return None


class GenerateIR:

    def __init__(self, model_name: str, base_url: str):
        self.vlm_model = model_name
        self.base_url = base_url + f"chat/completions" if base_url.endswith("/") else base_url + "/chat/completions"
        self.api_key = os.getenv("VLM_API_KEY", "")

    def generate_ir(self, path_or_url: str):
        img_str = path_or_url

        resp = send_chat_completion(
            api_key=self.api_key,
            model_name=self.vlm_model,
            base_url=self.base_url,
            system_prompt=SVR_PROMPT["system_prompt"],
            user_prompt=SVR_PROMPT["user_prompt"],
            image_url=img_str
        )["choices"][0]["message"]["content"]
        return resp

    def _process_single_instance(self, instance: Dict[str, Any], output_dir: str) -> bool:
        """Internal method to process a single instance"""
        try:
            with open(instance["instance_file"], "r") as infile:
                doc = json.load(infile)

            img_base64 = process_image_url_to_base64(instance["cropped_image"])
            if img_base64 is None:
                return False
            image_caption = self.generate_ir(img_base64)
            doc["sub_graph_caption"] = image_caption

            log_dir = os.path.join(output_dir, instance["instance_id"])
            os.makedirs(log_dir, exist_ok=True)

            with open(os.path.join(log_dir, f"subgraph_{instance['instance_id']}.json"), "w") as outf:
                outf.write(json.dumps(doc, indent=4, ensure_ascii=False))

            return True
        except Exception as e:
            print(f"Error processing instance {instance['instance_id']}: {e}")
            traceback.print_exc()
            return False

    def process_batch(self, result_path: str, input_data: str, output_dir: str, image_dir: str, max_workers: int = 4):
        with open(input_data, "r") as infile:
            cropped_instance = json.load(infile)
        print("total cropped instance list: ", len(cropped_instance))

        processed_instance = os.listdir(output_dir)
        print(f"{len(processed_instance)} has been processed. ")

        # Collect instance information that needs to be processed
        instances_to_process = []
        if next(os.scandir(result_path), None) is None:
            res_dict = {}
            
            with open(input_data, "r") as infile:
                all_data = json.load(infile)
            
            # Collect all images that need to be processed
            tasks = []
            skipped_count = 0
            for problem_id, data in all_data.items():
                image_list = json.loads(data["image_assets"])["problem_statement"]
                for _img_url in image_list:
                    _img_str = process_image_url_to_base64(_img_url, image_dir, problem_id)
                    if _img_str is not None:
                        tasks.append((problem_id, _img_str))
                    else:
                        skipped_count += 1
            
            if skipped_count > 0:
                print(f"Skipped {skipped_count} non-existent image files")

            # tasks = tasks[:3]
    
            print(f"Total images to process: {len(tasks)}")

            bar = tqdm.tqdm(total=len(tasks), ncols=96)
            
            # Use multi-threading for processing
            
            def process_single_image(problem_id_img_tuple):
                problem_id, img_str = problem_id_img_tuple
                try:
                    resp = self.generate_ir(img_str)
                    return problem_id, resp
                except Exception as e:
                    print(f"Error processing image: {e}")
                    traceback.print_exc()
                    return problem_id, None
            
            # Group and store results by problem_id
            temp_results = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_task = {executor.submit(process_single_image, task): task for task in tasks}
                
                for future in as_completed(future_to_task):
                    problem_id, resp = future.result()
                    if resp is not None:
                        if problem_id not in temp_results:
                            temp_results[problem_id] = []
                        temp_results[problem_id].append(resp)
                    bar.update()
            
            # Merge results while maintaining original order
            for problem_id, data in all_data.items():
                res_dict[problem_id] = data
                if problem_id in temp_results:
                    res_dict[problem_id]['image_caption'] = temp_results[problem_id]
                else:
                    res_dict[problem_id]['image_caption'] = []

            with open(os.path.join(output_dir, "image_ir_data.json"), "w") as outf:
                outf.write(json.dumps(res_dict, indent=4, ensure_ascii=False))
        else:
            for instance_id in os.listdir(result_path):
                if instance_id not in cropped_instance:
                    continue

                if result_path != output_dir and instance_id in processed_instance:
                    continue
                else:
                    instance_dir = os.path.join(result_path, instance_id)
                    sub_file = glob.glob(os.path.join(instance_dir, "subgraph_*.json"))
                    if sub_file:
                        continue
                instance_file = glob.glob(os.path.join(instance_dir, "*.json"))
                if not instance_file:
                    continue

                cropped_image = glob.glob(os.path.join(instance_dir, "cropped*.png"))
                if not cropped_image:
                    continue
                
                instances_to_process.append({
                    "instance_id": instance_id,
                    "instance_dir": instance_dir,
                    "instance_file": instance_file[0],
                    "cropped_image": cropped_image[0]
                })

            generated = []
            failed = []
            if len(instances_to_process) > 3:
                instances_to_process = instances_to_process[:3]

            # Use thread pool for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_instance = {
                    executor.submit(self._process_single_instance, instance, output_dir): instance
                    for instance in instances_to_process
                }
                
                # Collect results
                for future in as_completed(future_to_instance):
                    instance = future_to_instance[future]
                    try:
                        success = future.result()
                        if success:
                            generated.append(instance["instance_id"])
                            print(f"✓ Successfully processed instance: {instance['instance_id']}")
                        else:
                            failed.append(instance["instance_id"])
                            print(f"- Skipped instance: {instance['instance_id']}")
                    except Exception as e:
                        failed.append(instance["instance_id"])
                        print(f"✗ Failed to process instance {instance['instance_id']}: {e}")
                        traceback.print_exc()

            print(f"Total processed {len(instances_to_process)} instances")
            print(f"Successfully generated image ir: {len(generated)} instances")
            print(f"Failed to process: {len(failed)} instances")

            # Merge all saved docs into a single json file
            self._merge_all_docs(output_dir)

    def _merge_all_docs(self, output_dir: str):
        """Merge all saved docs into a single json file"""
        all_docs = {}
        
        print("Starting to merge all docs into a single json file...")
        
        # Traverse all subdirectories under output_dir
        for instance_id in os.listdir(output_dir):
            instance_dir = os.path.join(output_dir, instance_id)
            
            # Check if it's a directory
            if not os.path.isdir(instance_dir):
                continue
                
            # Find subgraph_*.json files
            subgraph_files = glob.glob(os.path.join(instance_dir, "subgraph_*.json"))
            
            for subgraph_file in subgraph_files:
                try:
                    with open(subgraph_file, "r", encoding="utf-8") as f:
                        doc = json.load(f)
                         
                    # Use instance_id as key, or extract from filename
                    file_name = os.path.basename(subgraph_file)
                    key = file_name.replace("subgraph_", "").replace(".json", "")
                    
                    all_docs[key] = doc
                    
                except Exception as e:
                    print(f"Failed to read file {subgraph_file}: {e}")
        
        # Save merged json file
        merged_file_path = os.path.join(output_dir, "all_subgraphs_merged.json")
        with open(merged_file_path, "w", encoding="utf-8") as f:
            json.dump(all_docs, f, indent=4, ensure_ascii=False)
            
        print(f"Merge completed! Total file saved to: {merged_file_path}")
        print(f"Total merged {len(all_docs)} docs")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="Qwen3-VL-235B-A22B-Instruct", required=True, help="VLM")
    parser.add_argument("--base_url", required=True, help="Base URL for API")
    parser.add_argument("--result_path")
    parser.add_argument("--max_workers", type=int, default=4, help='Max workers')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    generate_ir = GenerateIR(args.model_name, args.base_url)
    generate_ir.process_batch(args.result_path, args.input_data, args.output_dir, args.image_dir, max_workers=1)


if __name__ == "__main__":
    main()