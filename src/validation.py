import json
import os
import argparse
import traceback
from typing import *
import glob
import subprocess
from jinja2 import Template
from prompt.test_prompt import TestPrompt
from src.utils.logger import logger

class Validation:
    def __init__(self):
        pass

    @staticmethod
    def check_for_empty_result(instance_dir: str):
        resp_data = glob.glob(os.path.join(instance_dir, "*.json"))
        patch_data = glob.glob(os.path.join(instance_dir, "*.patch"))

        if not resp_data or not patch_data:
            return None

        with open(patch_data[0], "r") as infile:
            patch = infile.read()

        if not patch:
            return None

        return resp_data[0], patch_data[0]

    @staticmethod
    def check_for_model_failed(resp_data: str):
        with open(resp_data, "r") as infile:
            res_dict = json.load(infile)

        res_data = res_dict["fix_patch"][0]
        if "context_length_exceeded" in res_data.lower():
            return None

        if "Error:" in res_data:
            return None

        return res_dict

    def process_instance(self, instance_dir: str):
        ###### step 1. check for empty result
        empty_patch = self.check_for_empty_result(instance_dir)
        if empty_patch is None:
            return {
                "reason": "empty patch.",
                "instance_dir": instance_dir
            }
        else:
            resp_data, patch_data = empty_patch

        ####### step 2. check for model failed
        res_dict = self.check_for_model_failed(resp_data)
        if res_dict is None:
            return {
                "reason": "model failed. ",
                "instance_dir": instance_dir
            }

        # step 3. check for patch
        patch_size = os.path.getsize(patch_data) / 1024 / 1024
        if patch_size > 15:
            return {
                "reason": "Patch exceed limitation. ",
                "instance_dir": instance_dir
            }

        return None

    def filtering_result(self, data_path: str, result_path: str):
        with open(data_path, "r") as infile:
            instance_dict = json.load(infile)
        logger.info(f"{len(instance_dict)} need to be process")

        error_instance = {}
        crt_instance = {}
        for instance_id in os.listdir(result_path):
            if not os.path.isdir(os.path.join(result_path, instance_id)):
                continue

            if instance_id not in instance_dict:
                continue

            try:
                instance_dir = os.path.join(result_path, instance_id)
                # logger.info("instance dir: ", instance_dir)
                valid_res = self.process_instance(instance_dir)
                # If already processed, proceed to validation step
                if valid_res is not None:
                    error_instance[instance_id] = instance_dict.pop(instance_id)  # Temporarily ignore the reason.
                else:
                    crt_instance[instance_id] = instance_dict.pop(instance_id)
            except Exception as e:
                traceback.print_exc()
                # exit()
                error_instance[instance_id] = instance_dict.pop(instance_id)

        for k, v in instance_dict.items():
            error_instance[k] = v

        print("checked result: ", len(error_instance))
        failed_path = os.path.join(result_path, "model_response_validation_failed.json")
        with open(failed_path, "w") as outf:
            outf.write(json.dumps(error_instance, indent=4, ensure_ascii=False))

        crt_path = os.path.join(result_path, "model_response_validation_success.json")
        with open(crt_path, "w") as outf:
            outf.write(json.dumps(crt_instance, indent=4, ensure_ascii=False))

        logger.info(f"validation done, {len(error_instance)} cannot pass the validation")
        logger.info(f"{len(crt_instance)} has been validated to be correct.")

        return failed_path, crt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--result_path", required=True)
    args = parser.parse_args()

    validation = Validation()
    validation.filtering_result(args.data_path, args.result_path)


if __name__ == "__main__":
    # Main execution code
    main()
