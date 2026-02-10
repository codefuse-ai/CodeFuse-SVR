COMMAND = ("cd {{repo_dir}} && git reset --hard HEAD && git checkout -f {{commit_id}} && "
           "pycfuse --model {{model_name}} --api-key {{api_key}} --base-url {{base_url}} -pp {{prompt_file}} --logs-dir {{log_dir}} --temperature {{temperature}} --yolo && "
           "git -C {{repo_dir}} add . -A && "
           "git -C {{repo_dir}} diff --cached -- '*.js' '*.ts' '*.jsx' '*.tsx' '*.js.snap' '*.scss' '*.md' '*.lua[' ':!*test*' ':!*tests*' ':!*_test.py' > {{patch_file}}")