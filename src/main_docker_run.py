#!/usr/bin/env python3
"""
SWE Bench MM Main Program

Refactored main program providing clear command-line interface and modular code structure.

Usage:
    python main.py --data_path <data_path> --output_dir <output_dir>

Args:
    data_path: Input JSON data file path
    output_dir: Output results directory
    model_name: Model name to use
    repo_path: Repository path

Example:
    python main.py --data_path data/test.json --output_dir results/
"""
import json
import os
import subprocess
import argparse
from typing import *
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import glob
import shutil
import time

from tqdm import tqdm
from jinja2 import Template

from prompt.image_prompt import SubGraphPrompt
from src.utils.logger import logger
from src.utils.config_manager import ConfigManager
from src.run_cmd.run_cfuse import COMMAND

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

class PatchGeneration:
    """Batch process multiple documents"""

    def __init__(self, model_name: str, base_url: str, max_workers: int = 4, temperature: float = 0.0):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = OPENAI_API_KEY

        self.max_workers = max_workers
        self.progress_lock = threading.Lock()
        self.temperature = temperature


    @staticmethod
    def get_processed_instances(output_dir: str) -> List[str]:
        if not os.path.exists(output_dir):
            return []

        processed = []
        for file_path in os.listdir(output_dir):
            if os.path.join(os.path.join(output_dir, file_path)):
        # Check if there are .patch suffix files in the directory
                patch_files = list(glob.glob(os.path.join(output_dir, file_path, "*.patch")))
                if patch_files:
                    processed.append(file_path)
        return processed

    def execute_single_task(self, doc: Dict, repo_path: str, output_dir: str, pbar: tqdm) -> Dict:
        """Execute a single task"""
        instance_id = doc["instance_id"]
        try:
            log_dir = os.path.join(output_dir, instance_id)
            os.makedirs(log_dir, exist_ok=True)

            problem = doc["problem_statement"]
            img_prompt = doc.get("image_caption", "")
            sub_image_prompt = doc.get("sub_graph_caption", "")
            repo_name = doc["repo"].split("/")[-1]
            repo_dir = os.path.join(repo_path, repo_name)
            
            user_prompt = Template(SubGraphPrompt).render({
                "problem_statement": problem,
                "image_captioning": img_prompt,
                "sub_image_caption": sub_image_prompt
            })

            prompt_file = os.path.join(log_dir, f"user_prompt_{instance_id}.txt")
            with open(prompt_file, "w") as out_file:
                out_file.write(user_prompt)

            patch_list = []
            # Clear possible git lock files
            lock_file = os.path.join(repo_dir, '.git', 'index.lock')
            if os.path.exists(lock_file):
                os.remove(lock_file)
                logger.info(f"Removed git lock file: {lock_file}")

            patch_file = os.path.join(log_dir, f"res_patch_{instance_id}.patch")
            script_path = os.path.join(log_dir, f"{instance_id}_script.sh")
            code_command = Template(COMMAND).render(
                {
                    "repo_dir": os.path.abspath(repo_dir),
                    "commit_id": doc["base_commit"],
                    "model_name": self.model_name,
                    "prompt_file": os.path.abspath(prompt_file),
                    "log_dir": os.path.abspath(log_dir),
                    "patch_file": os.path.abspath(patch_file),
                    "base_url": self.base_url,
                    "api_key": self.api_key,
                    "temperature": self.temperature
                }
            )
            with open(script_path, "w") as outf:
                outf.write(code_command)

            logger.info(f"command: {code_command}")

            result = subprocess.run(f"sh {script_path}", capture_output=True, text=True, bufsize=0, shell=True)
            if result.returncode != 0:
                logger.error(f"Error processing {instance_id}: exit_code={result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                doc["error"] = result.stderr
            else:
                resp = result.stdout.strip()
                patch_list.append(resp)
                logger.info(f"results: {resp}")

            doc["fix_patch"] = patch_list
            logger.info(f"Successfully processed {instance_id}")

            with open(os.path.join(log_dir, f"resp_{instance_id}.json"), "w") as out_res:
                out_res.write(json.dumps(doc, indent=4, ensure_ascii=False))

            # Update progress bar
            with self.progress_lock:
                pbar.update(1)

            return doc

        except Exception as e:
            logger.error(f"Exception processing {instance_id}: {str(e)}")
            traceback.print_exc()
            doc["error"] = str(e)

            # Update progress bar
            with self.progress_lock:
                pbar.update(1)

            return doc

    def process_repo_group(self, docs: List[Dict], repo_path: str, output_dir: str, pbar: tqdm) -> List[Dict]:
        """Process all tasks for a single repo group"""
        results = []
        for doc in docs:
            try:
                result = self.execute_single_task(doc, repo_path, output_dir, pbar)
                results.append(result)
            except:
                print(f"error when processing {repo_path}")
                traceback.print_exc()
        return results

    def test_main(self, data_path: str, output_dir: str, repo_path: str):
        with open(data_path, "r") as infile:
            data_dict: Dict = json.load(infile)

        repo_groups: Dict[str, List[Dict]] = {}
        for instance_id, doc in data_dict.items():
            repo = doc["repo"]
            if repo not in repo_groups:
                repo_groups[repo] = []
            repo_groups[repo].append(doc)

        bar = tqdm(total=len(repo_groups), ncols=96)
        for repo_name, docs in repo_groups.items():
            self.process_repo_group(docs, repo_path, output_dir, bar)
            exit()


    def process_batch(
            self,
            data_path: str,
            output_dir: str,
            repo_path: str,
            copy_repo: bool = False
    ) -> None:
        # Copy repo_path to parent directory
        if copy_repo:
            parent_dir = os.path.dirname(os.path.abspath(repo_path))
            timestamp = str(int(time.time()))
            copied_repo_path = os.path.join(parent_dir, f"swebench_m_test_repos_copy_{timestamp}")

            logger.info(f"Copying repo from {repo_path} to {copied_repo_path}")
            shutil.copytree(repo_path, copied_repo_path)
            logger.info(f"Repo copied successfully to {copied_repo_path}")

        # Use copied directory as new repo_path
            repo_path = copied_repo_path

        # Load data
        with open(data_path, "r") as infile:
            data_dict: Dict = json.load(infile)

        # Get processed instances
        processed_instances = set(self.get_processed_instances(output_dir))
        print(f"{len(processed_instances)} items have been processed")

        remaining_data = data_dict
        if not remaining_data:
            print("All instances have been processed")
            return

        # Group by repo
        # remaining_data = {
        #     k: v for i, (k, v) in enumerate(remaining_data.items())
        #     if i < 3
        # }

        repo_groups: Dict[str, List[Dict]] = {}
        for instance_id, doc in remaining_data.items():
            repo = doc["repo"]
            if repo not in repo_groups:
                repo_groups[repo] = []
            repo_groups[repo].append(doc)

        print(f"Found {len(repo_groups)} different repositories to process")
        for repo, docs in repo_groups.items():
            print(f"  {repo}: {len(docs)} instances")

        # Create progress ba
        total_tasks = len(remaining_data)
        with tqdm(total=total_tasks, ncols=96, desc="Overall Progress") as pbar:

            # Use thread pool to concurrently execute tasks for different repos
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_repo = {}

                # Submit tasks for each repo group
                for repo_name, docs in repo_groups.items():
                    future = executor.submit(
                        self.process_repo_group,
                        docs,
                        repo_path,
                        output_dir,
                        pbar
                    )
                    future_to_repo[future] = repo_name

                # Wait for all tasks to complete
                all_results = []
                for future in as_completed(future_to_repo):
                    repo_name = future_to_repo[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                        logger.info(f"Completed processing repository: {repo_name}")
                    except Exception as e:
                        logger.error(f"Error processing repository {repo_name}: {str(e)}")
                        traceback.print_exc()

        logger.info(f"Processing completed. Total processed: {len(all_results)}")



def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="SWE Bench MM Batch Processing Program")
    parser.add_argument("--data_path", type=str, required=True, help="Input JSON data file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output results directory")
    parser.add_argument("--model_name", default="Kimi-K2-Instruct-0905", required=True, help="Model name")
    parser.add_argument("--base_url", required=True, help="Base URL for API")
    parser.add_argument("--repo_path", type=str, help="Repository base path")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum concurrent threads")
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--copy_repo", action="store_true", help="Whether to copy repo directory before starting")

    args = parser.parse_args()

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Save experiment configuration
    config_file = ConfigManager.save_experiment_config(args, output_dir)
    logger.info(f"Experiment configuration saved to: {config_file}")

    data_path = args.data_path

    logger.info(f"Starting processing with data: {data_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Repo path: {args.repo_path}")

    # Create batch processor and run
    processor = PatchGeneration(args.model_name, args.base_url, args.max_workers, args.temperature)
    processor.process_batch(data_path, output_dir, args.repo_path, copy_repo=args.copy_repo)

    logger.info("Processing completed successfully")



if __name__ == "__main__":
    main()