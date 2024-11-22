#--model_args pretrained=microsoft/Phi-3-mini-4k-instruct,parallelize=True \
#--model_args pretrained=microsoft/rho-math-7b-v0.1,parallelize=True \
#--model_args pretrained=deepseek-ai/deepseek-math-7b-rl,parallelize=True \
#--model_args pretrained=meta-llama/Meta-Llama-3.1-8B,parallelize=True \
#--model_args pretrained=RUCKBReasoning/TableLLM-13b,parallelize=True \
#--model_args pretrained=EleutherAI/llemma_7b,parallelize=True \
#microsoft/Phi-3-medium-4k-instruct
#Qwen/Qwen2-72B-Instruct
#mistralai/Mixtral-8x22B-Instruct-v0.1
#meta-llama/Llama-3.2-3B-Instruct
#meta-llama/Llama-3.2-1B-Instruct

DEVICE="ALL"

lm_eval --model vllm \
    --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=3 \
    --tasks num_tab_qa_gittables_100k \
    $([ "$DEVICE" != "ALL" ] && echo "--device cuda:$DEVICE") \
    --batch_size auto:4 \
    --output_path ./num_tab_qa_test_output/output.txt \
    --num_fewshot 2 \
    --trust_remote_code #\
    #--log_samples