import os
import json
import re
import sys
import argparse
from typing import List, Dict, Any

from xml.etree.ElementTree import indent


def extract_test_result(llm_test_result):
    # Use regular expression to match content within <result> tags
    pattern = r'<result>\s*([^<]+)\s*</result>'
    match = re.search(pattern, llm_test_result, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()
    else:
        return None


def read_resp_json_files(result_path: str) -> Dict[str, Any]:
    """
    Traverse all subdirectories under result_path, read json files starting with resp_ in test_log directory
    
    Args:
        result_path: Root directory path to traverse
        
    Returns:
        List[Dict[str, Any]]: List of all json file contents starting with resp_
    """
    failed_instance = {}

    # Traverse all subdirectories under result_path
    for instance_id in os.listdir(result_path):
        item_path = os.path.join(result_path, instance_id)
        
        # Only process directories
        if os.path.isdir(item_path):
            test_log_path = os.path.join(item_path, "test_log")
            # Check if test_log directory exists
            if os.path.exists(test_log_path) and os.path.isdir(test_log_path):
                # Traverse files in test_log directory
                resp_file = os.path.join(test_log_path, f"resp_{instance_id}.json")
                if not os.path.exists(resp_file):
                    print(f"cannot find resp: {instance_id}")
                    continue
                with open(resp_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    if "llm_test_result" not in data:
                        continue
                    llm_resp = data["llm_test_result"]
                    validation_result = extract_test_result(llm_resp)
                    if validation_result is None:
                        failed_instance[instance_id] = data
                    elif "success" not in validation_result.strip().lower():
                        failed_instance[instance_id] = data
                    elif not data['validation_result']:
                        failed_instance[instance_id] = data
                    else:
                        print(f"{instance_id} validation result: {validation_result}")
                        continue
            else:
                print(f"{instance_id} test log not exist")

    print("total failed instance: ", len(failed_instance))

    return failed_instance


def merge_result(agent_val_path: str, rule_val_path: str):
    with open(rule_val_path, "r") as infile:
        rule_val_result = json.load(infile)

    with open(agent_val_path, "r") as infile:
        agent_result = json.load(infile)

    for k in rule_val_result:
        if k not in agent_result:
            agent_result[k] = rule_val_result[k]

    return agent_result


def process_val(result_path):
    data = read_resp_json_files(result_path)

    agent_path = os.path.join(result_path, "agent_validation_failed_instance.json")
    with open(agent_path, "w") as outf:
        outf.write(json.dumps(data, indent=4, ensure_ascii=False))

    rule_path = os.path.join(result_path, "model_response_validation_failed.json")
    all_result = merge_result(agent_path, rule_path)
    
    with open(os.path.join(result_path, "all_validation_failed_instance.json"), "w") as outf:
            outf.write(json.dumps(all_result, indent=4, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", required=True)
    args = parser.parse_args()

    process_val(args.result_path)
    

if __name__ == "__main__":
    main()