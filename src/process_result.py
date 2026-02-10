import argparse
import collections
import json
import os
import subprocess
import glob
import traceback

import pandas as pd


def convert_list_to_dict(data_path: str, output_dir: str):
    data_dict = {}

    with open(data_path, "r") as infile:
        data_list = json.load(infile)

    for doc in data_list:
        data_dict[doc["instance_id"]] = doc

    with open(os.path.join(output_dir, "swe_bench_mm_prompt_v3_dict.json"), "w") as outf:
        outf.write(json.dumps(data_dict, indent=4, ensure_ascii=False))


def process_git_diff(result_path: str, model_name: str = "SVR"):
    res_dict = {}
    for instance_id in os.listdir(result_path):
        res_patch = list(glob.glob(os.path.join(result_path, instance_id, "*.patch")))
        if res_patch:
            res_patch = res_patch[0]
        else:
            continue
        try:
            with open(res_patch, "r") as infile:
                patch = infile.read()
                res_dict[instance_id] = {
                    "model_name_or_path": model_name,
                    "model_patch": patch
                }
        except:
            print(f"Error when processing: {instance_id}")
            traceback.print_exc()
            res_dict[instance_id] = {
                "model_name_or_path": model_name,
                "model_patch": ""
            }

    # basename = result_path.split("/")[-1]
    with open(os.path.join(result_path, f"{model_name}_result_path.json") , "w") as outf:
        outf.write(json.dumps(res_dict, indent=4, ensure_ascii=False))

    print(f"Total result={len(res_dict)}")


def cal_image_error(result_path: str):
    count = 0
    tot = len(os.listdir(result_path))
    for res_file in os.listdir(result_path):
        instance_id = res_file
        jsonfile = os.path.join(result_path, res_file, f"resp_{instance_id}.json")
        if not os.path.exists(jsonfile):
            continue
        with open(jsonfile, "r") as infile:
            doc = json.load(infile)

        for caption in doc["image_caption"]:
            if caption.startswith("Error"):
                count += 1
                break

    print(f"Total image error instance: {count}")
    print(f"image error ratio: {count / tot}")


def cal_repo_error(result_path: str):
    stats = {}
    metric = ["total", "resolved", "unresolved", "res_ratio", "un_ratio"]

    for instance_id in os.listdir(os.path.join(result_path, "resolved_ids")):
        repo_name = instance_id.split("__")[0]
        if repo_name not in stats:
            stats[repo_name] = collections.defaultdict(int)
        stats[repo_name]["resolved_ids"] += 1
        stats[repo_name]["total"] += 1


    for instance_id in os.listdir(os.path.join(result_path, "unresolved_ids")):
        repo_name = instance_id.split("__")[0]
        if repo_name not in stats:
            stats[repo_name] = collections.defaultdict(float)
        stats[repo_name]["unresolved_ids"] += 1
        stats[repo_name]["total"] += 1

    for k, v in stats.items():
        stats[k]["res_ratio"] = stats[k]["resolved_ids"]/stats[k]["total"]
        stats[k]["un_ratio"] = stats[k]["unresolved_ids"] / stats[k]["total"]

    print(f"stats: {json.dumps(stats, indent=4)}")

    df = pd.DataFrame.from_dict(stats)
    print(df.round(2))
    df.to_excel(os.path.join(result_path, "result_stats.xlsx"))


def patch_error(result_dir):

    count = 0
    tot = len(os.listdir(result_dir))
    invalid_patch = []
    for filepath in os.listdir(result_dir):
        patch_file = os.path.join(result_dir, filepath, f"res_patch_{filepath}.patch")

        if not os.path.exists(patch_file):
            print("patch file: ", patch_file)
            invalid_patch.append(filepath)
            count += 1
            continue
        command = f'git apply --check {patch_file} 2 > /dev/null && echo "Valid patch" || echo "Invalid patch"'
        result = subprocess.run(command, capture_output=True, text=True, bufsize=0, shell=True)
        result = result.stdout.strip()
        # print(result)
        if "invalid" in result.lower():
            count += 1
            invalid_patch.append(filepath)

        # break

    print(f"invalid patch ratio: {count / tot}")
    print(f"total invalid patch count: {count}")

    with open(os.path.join(result_dir, "invalid_patch.json"), "w") as outf:
        outf.write(json.dumps(invalid_patch, indent=4))


def process_result_with_feedback(repo_data_file: str, result_path: str, model_name: str = "SVR"):
    with open(repo_data_file, "r") as infile:
        repo_data = json.load(infile)

    for instance_id in os.listdir(result_path):
        # if instance_id in repo_data:
        #     continue

        res_patch = list(glob.glob(os.path.join(result_path, instance_id, "*.patch")))
        if res_patch:
            res_patch = res_patch[0]
        else:
            continue
        try:
            with open(res_patch, "r") as infile:
                patch = infile.read()
                repo_data[instance_id] = {
                    "model_name_or_path": model_name,
                    "model_patch": patch
                }
        except:
            print(f"Error when processing: {instance_id}")
            traceback.print_exc()
            repo_data[instance_id] = {
                "model_name_or_path": model_name,
                "model_patch": patch
            }

    # parent_dir = os.path.dirname(repo_data_file)
    with open(os.path.join(result_path, f"{model_name}_result.json"), "w") as outf:
        outf.write(json.dumps(repo_data, indent=4, ensure_ascii=False))

    print(f"total result: {len(repo_data)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", required=True)
    parser.add_argument("--repo_file")
    parser.add_argument("--model_name", default="SVR")
    args = parser.parse_args()

    if args.repo_file is not None:
        process_result_with_feedback(args.repo_file, args.result_path)
    else:
        process_git_diff(args.result_path)

    print("process result done.")


if __name__ == "__main__":
    # Main execution code
    main()