
models=("Qwen/Qwen2.5-7B-Instruct" "/home/aiscuser/CascadeDistill/saves/deepseek_qwen2.5-7b-Instruct/use_math-verify_filter/deepseek_qwen2.5-7b-Instruct_1kx5_sft_ds3/20251008-225244" "/home/aiscuser/CascadeDistill/saves/deepseek_qwen2.5-7b-Instruct/use_math-verify_filter/deepseek_qwen2.5-7b-Instruct_5kx1_sft_ds3/20251009-001129" "/home/aiscuser/CascadeDistill/saves/deepseek_qwen2.5-7b-Instruct/use_math-verify_filter/deepseek_qwen2.5-7b-Instruct_math-verify-1kx5_sft_ds3/20251009-012917" "/home/aiscuser/CascadeDistill/saves/deepseek_qwen2.5-7b-Instruct/use_math-verify_filter/deepseek_qwen2.5-7b-Instruct_math-verify-5kx1_sft_ds3/20251009-040524")

tasks=("AIME" "AMC" "Olympiad" "hendrycks_math_500" "gsm8k_cot_zeroshot")

max_model_tokens=16000
max_gen_tokens=16000
model_args="tensor_parallel_size=1,data_parallel_size=8,gpu_memory_utilization=0.85,max_model_len=$max_model_tokens,dtype=bfloat16"
output_path="test_math_verify"
batch_size="auto"

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        echo "Running lm_eval with model: $model, task: $task"
        lm_eval --model vllm \
            --model_args pretrained="$model",$model_args \
            --gen_kwargs do_sample=false,temperature=0,max_gen_toks=$max_gen_tokens \
            --tasks "$task" \
            --batch_size "$batch_size" \
            --log_samples \
            --trust_remote_code \
            --output_path "$output_path" \
            --apply_chat_template \


        SANTIZED_MODEL_SAVE_LABEL=$(echo ${model} | sed 's/\//__/g')
        echo ${SANTIZED_MODEL_SAVE_LABEL}
        if [ "$task" != "gsm8k_cot_zeroshot" ]; then
            python math_metric_llm_eval_general.py --directory_path ${output_path}/${SANTIZED_MODEL_SAVE_LABEL} --task ${task}
        elif [ "$task" == "gsm8k_cot_zeroshot" ]; then
            python math_metric_gsm8k.py --directory_path ${output_path}/${SANTIZED_MODEL_SAVE_LABEL} 
        fi


    done
done
