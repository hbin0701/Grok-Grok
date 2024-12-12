#!/bin/bash

# Run all tasks simultaneously on different GPUs
# CUDA_VISIBLE_DEVICES=0 python train_grok.py --fn_name Task1 --project_name "Arith-Transfer-Multi" &
# CUDA_VISIBLE_DEVICES=1 python train_grok.py --fn_name Task2 --project_name "Arith-Transfer-Multi" &
# CUDA_VISIBLE_DEVICES=2 python train_grok.py --fn_name Task3 --project_name "Arith-Transfer-Multi" &
# CUDA_VISIBLE_DEVICES=3 python train_grok.py --fn_name Task4 --project_name "Arith-Transfer-Multi" &
# CUDA_VISIBLE_DEVICES=4 python train_grok.py --fn_name Task5 --project_name "Arith-Transfer-Multi" &
# CUDA_VISIBLE_DEVICES=5 python train_grok.py --fn_name Task6 --project_name "Arith-Transfer-Multi" &
# CUDA_VISIBLE_DEVICES=6 python train_grok.py --fn_name Task7 --project_name "Arith-Transfer-Multi" &
# CUDA_VISIBLE_DEVICES=7 python train_grok.py --fn_name Task8 --project_name "Arith-Transfer-Multi" &
# CUDA_VISIBLE_DEVICES=7 python train_grok.py --fn_name Task9 --project_name "Arith-Transfer-Multi" &
# Wait for all processes to complete
# wait

# echo "All training tasks completed"
#!/bin/bash

BASE_DIR="/home/hyeonbin/Arith_transfer"
PROJECT_NAME="Arith-Transfer-Multi"
NUM_TASKS=9

# Function to run all target tasks from a source task checkpoint
run_transfer_from_checkpoint() {
    local source_task=$1
    
    # Create array of target tasks (all tasks except source)
    for target_task in $(seq 1 $NUM_TASKS); do
        if [ $target_task -ne $source_task ]; then
            echo "Training Task${target_task} from Task${source_task} checkpoint..."
            CUDA_VISIBLE_DEVICES=$(( (target_task-1) % 8 )) python train_grok.py \
                --fn_name "Task${target_task}" \
                --project_name "${PROJECT_NAME}" \
                --ckpt "${BASE_DIR}/Task${source_task}/full_run_data.pth" &
        fi
    done
    
    # Wait for all parallel tasks to complete
    wait
    echo "Completed all transfers from Task${source_task}"
    echo "----------------------------------------"
}

# Main execution
echo "Starting transfer learning experiments..."
echo "----------------------------------------"

# Run transfers from each task's checkpoint
for source_task in $(seq 1 $NUM_TASKS); do
    echo "Starting transfers from Task${source_task} checkpoint..."
    run_transfer_from_checkpoint $source_task
done

echo "All transfer learning experiments completed!"