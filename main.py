import argparse
import json
import os
import sys

from src.generate_image_ir import GenerateIR
from src.main_docker_run import PatchGeneration
from src.validation import Validation
from src.agent_validation import AgentBaseValidation
from src.process_validation import process_val
from src.image_repo_localization import ImageCodeLocalization
from src.process_result import process_git_diff


def run_svrepair(
        input_data: str,
        output_dir: str,
        repo_path: str,
        project_name: str = "svrepair",
        vlm_model: str = "Qwen3-VL-235B-A22B-Instruct",
        vlm_url: str = "https://xxx/v1",
        image_dir: str = None,
        model_name: str = "Kimi-K2-Instruct-0905",
        base_url: str = "https://xxx/v1",
        temperature: float = 0,
        copy_repo: bool = False,
        max_workers: int = 4
):
    # step 1. generate image ir
    image_ir = GenerateIR(vlm_model, vlm_url)
    image_ir.process_batch(output_dir, input_data, output_dir, image_dir, max_workers=max_workers)
    image_ir_path = os.path.join(output_dir, "image_ir_data.json")
    print(f"Image IR generated: {image_ir_path}")

    # step 2. generate patch
    patch_gen = PatchGeneration(model_name, base_url, max_workers, temperature)
    patch_gen.process_batch(image_ir_path, output_dir, repo_path=repo_path, copy_repo=copy_repo)
    print(f"Patches generated: {output_dir}")

    # step 3. validation
    # rule-base validation
    validation = Validation()
    failed_path, _ = validation.filtering_result(image_ir_path, output_dir)
    # agent-base validation
    agent_val = AgentBaseValidation(repo_path=repo_path, model_name=model_name, base_url=base_url, max_workers=max_workers)
    agent_val.process_batch(failed_path, output_dir, repo_path=repo_path, copy_repo=copy_repo)
    process_val(output_dir)
    
    # step 4. localization
    localizer = ImageCodeLocalization(vlm_model, vlm_url)
    fialed_data_path = os.path.join(output_dir, "all_validation_failed_instance.json")
    localizer.process_batch(fialed_data_path, output_dir, image_dir)

    # step 5. redo patch generation
    image_ir.process_batch(output_dir, fialed_data_path, output_dir, image_dir, max_workers=max_workers)
    subimage_ir_path = os.path.join(output_dir, "all_subgraphs_merged.json")
    print(f"Subimage IR generated: {subimage_ir_path}")
    patch_gen.process_batch(subimage_ir_path, output_dir, repo_path=repo_path, copy_repo=copy_repo)
    print(f"Patches generated again: {output_dir}")

    # step 6. process result
    process_git_diff(output_dir, project_name)


def main():
    parser = argparse.ArgumentParser(description="SVRepair command line tool")
    subparsers = parser.add_subparsers(dest='cmd', help='Available commands')
    
    # generate-image-ir command
    gen_ir_parser = subparsers.add_parser('generate-image-ir', help='Generate image IR')
    gen_ir_parser.add_argument("--model_name", default="Kimi-K2-Instruct-0905", required=True, help="Model name")
    gen_ir_parser.add_argument("--base_url", required=True, help="Base URL for API")
    gen_ir_parser.add_argument("--input_data", required=True, help='Path to input data')
    gen_ir_parser.add_argument("--image_dir", required=True, help='Path to image directory')
    gen_ir_parser.add_argument("--result_path", required=True, help='Path to result data')
    gen_ir_parser.add_argument("--output_dir", required=True, help='Output directory')
    gen_ir_parser.add_argument("--max_workers", type=int, default=4, help='Max workers')
    
    # generate-patch command
    gen_patch_parser = subparsers.add_parser('generate-patch', help='Generate patches')
    gen_patch_parser.add_argument("--image_ir_path", required=True, help='Path to image IR data')
    gen_patch_parser.add_argument("--temperature", default=0, type=float)
    gen_patch_parser.add_argument("--output_dir", required=True, help='Output directory')
    gen_patch_parser.add_argument("--repo_path", required=True, help='Repository path')
    gen_patch_parser.add_argument("--model_name", default="Kimi-K2-Instruct-0905", help='Model name')
    gen_patch_parser.add_argument("--base_url", required=True, help="Base URL for API")
    gen_patch_parser.add_argument("--copy_repo", action="store_true", help='Copy repository')
    
    # validation command
    validation_parser = subparsers.add_parser('validation', help='Validate patches')
    validation_parser.add_argument("--image_ir_path", required=True, help='Path to image IR data')
    validation_parser.add_argument("--result_path", required=True, help='Path to result data')
    validation_parser.add_argument("--output_dir", required=True, help='Output directory')
    validation_parser.add_argument("--model_name", default="Kimi-K2-Instruct-0905", help='Model name')
    validation_parser.add_argument("--base_url", required=True, help="Base URL for API")
    validation_parser.add_argument("--max_workers", type=int, default=4, help='Max workers')
    validation_parser.add_argument("--repo_path", type=str, default="data/swe_bench_mm/repos", help="Repository base path")

    # localization command
    localization_parser = subparsers.add_parser('localization', help='Image localization')
    localization_parser.add_argument("--repo_path", required=True, help='Repository path')
    localization_parser.add_argument("--image_dir", required=True)
    localization_parser.add_argument("--output_dir", required=True, help='Output directory')
    localization_parser.add_argument("--model_name", default="Qwen3-VL-235B-A22B-Instruct", help='Model name')
    localization_parser.add_argument("--base_url", required=True, help="Base URL for API")
    localization_parser.add_argument("--result_path", required=True)
    
    # full-run command (original complete pipeline)
    full_run_parser = subparsers.add_parser('full-run', help='Run complete SVRepair pipeline')
    full_run_parser.add_argument("--input_data", required=True, help='Path to input data')
    full_run_parser.add_argument("--output_dir", required=True, help='Output directory')
    full_run_parser.add_argument("--repo_path", required=True, help='Repository path')
    full_run_parser.add_argument("--image_dir", required=True, help='Path to image directory')
    full_run_parser.add_argument("--vlm_model", default="Qwen3-VL-235B-A22B-Instruct", help='Model name')
    full_run_parser.add_argument("--vlm_url", required=True, help="Base URL for API")
    full_run_parser.add_argument("--model_name", default="Qwen3-VL-235B-A22B-Instruct", help='Model name')
    full_run_parser.add_argument("--base_url", required=True, help="Base URL for API")
    full_run_parser.add_argument("--temperature", default=0, type=float, help='Temperature for model')
    full_run_parser.add_argument("--copy_repo", action="store_true", help='Copy repository')
    full_run_parser.add_argument("--max_workers", type=int, default=4, help='Max workers')
    full_run_parser.add_argument("--project_name", type=str, default="svrepair", help='Project name')
    
    args = parser.parse_args()
    
    if args.cmd is None:
        parser.print_help()
        return
    
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.cmd == 'generate-image-ir':
        image_ir = GenerateIR(args.model_name, args.base_url)
        image_ir.process_batch(args.result_path, args.input_data, args.output_dir, args.image_dir, max_workers=args.max_workers)
        print(f"Image IR generated: {os.path.join(args.output_dir, 'image_ir_data.json')}")
    
    elif args.cmd == 'generate-patch':
        patch_gen = PatchGeneration(args.model_name, args.base_url, args.max_workers, args.temperature)
        patch_gen.process_batch(args.image_ir_path, args.output_dir, repo_path=args.repo_path, copy_repo=args.copy_repo)
        print(f"Patches generated: {args.output_dir}")
    
    elif args.cmd == 'validation':
        validation = Validation()
        failed_path, _ = validation.filtering_result(args.image_ir_path, args.result_path)
        agent_val = AgentBaseValidation(repo_path=args.repo_path, model_name=args.model_name, base_url=args.base_url, max_workers=args.max_workers)
        agent_val.process_batch(failed_path, args.result_path, args.repo_path, copy_repo=args.copy_repo)

        process_val(args.result_path)
    
    elif args.cmd == 'localization':
        localizer = ImageCodeLocalization(args.model_name, args.base_url)
        fialed_data_path = os.path.join(args.result_path, "all_validation_failed_instance.json")
        localizer.process_batch(fialed_data_path, args.result_path, args.image_dir)
    
    elif args.cmd == 'full-run':
        run_svrepair(
            input_data=args.input_data,
            output_dir=args.output_dir,
            repo_path=args.repo_path,
            vlm_model=args.vlm_model,
            vlm_url=args.vlm_url,
            image_dir=args.image_dir,
            model_name=args.model_name,
            base_url=args.base_url,
            temperature=args.temperature,
            copy_repo=args.copy_repo,
            max_workers=args.max_workers,
            project_name=args.project_name
        )

    return 0


if __name__ == "__main__":
    # Main execution code
    sys.exit(main())
