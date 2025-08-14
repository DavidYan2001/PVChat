#!/bin/bash

# 函数：执行命令并检查结果
execute_step() {
    step_num=$1
    env_name=$2
    working_dir=$3
    script_path=$4
    script_args=$5  # 可选的脚本参数

    echo "==================================================="
    echo "Ready to execute Step ${step_num}..."
    echo "Environment: ${env_name}"
    echo "Working Directory: ${working_dir}"
    echo "Script: ${script_path}"
    echo "==================================================="

   # 等待用户确认（设置 60s 超时）
    if ! read -t 60 -p "Press Enter to continue with Step ${step_num}, or type 'skip' to skip this step, or 'exit' to quit: " user_input; then
    echo -e "\nTimeout reached, automatically continuing..."
    user_input=""  # 设置为空字符串，相当于按了Enter
fi
    # 如果超时或什么都没输入，则 user_input 为空字符串，默认继续（相当于按 Enter）
    if [ -z "$user_input" ]; then
        echo "No input detected in 60s, continue..."
    elif [ "$user_input" = "skip" ]; then
        echo "Skipping Step ${step_num}"
        return 0
    elif [ "$user_input" = "exit" ]; then
        echo "Exiting script..."
        exit 0
    fi

    # 激活 conda 环境
    echo "Activating conda environment: ${env_name}"
    source /root/miniconda3/etc/profile.d/conda.sh
    if ! conda activate ${env_name}; then
        echo "Failed to activate ${env_name} environment"
        return 1
    fi

    # 设置 HF_HOME 环境变量
    echo "Setting HF_HOME environment variable"
    export HF_HOME=/root/autodl-tmp/cache/

    # 切换到工作目录
    echo "Changing to directory: ${working_dir}"
    if ! cd ${working_dir}; then
        echo "Failed to change to directory: ${working_dir}"
        return 1
    fi

    # 执行脚本
    echo "Executing script: ${script_path} ${script_args}"
    if [[ ${script_path} == *.py ]]; then
        python ${script_path} ${script_args}
    elif [[ ${script_path} == *.sh ]]; then
        bash ${script_path} ${script_args}
    else
        echo "Unsupported script type: ${script_path}"
        return 1
    fi

    # 检查执行结果
    if [ $? -ne 0 ]; then
        echo "Step ${step_num} failed!"
        read -p "Press Enter to continue with next step, or type 'exit' to quit: " user_input
        if [ "$user_input" = "exit" ]; then
            echo "Exiting script..."
            exit 1
        fi
        return 1
    fi

    echo "Step ${step_num} completed successfully!"
    echo

    # 等待用户确认继续
    read -p "Press Enter to continue to next step, or type 'exit' to quit: " user_input
    if [ "$user_input" = "exit" ]; then
        echo "Exiting script..."
        exit 0
    fi
}

# 主执行流程
main() {
    # Step 1
    execute_step 1 "deepfacelab" "/root/autodl-tmp/yufei/datasets/cekebv-hq" \
        "/root/autodl-tmp/yufei/datasets/cekebv-hq/generate_new_person.py" "--small_dataset"

    # Step 2
    execute_step 2 "deepfacelab" "/root/autodl-tmp/yufei/DeepFaceLab" \
        "/root/autodl-tmp/yufei/DeepFaceLab/extract_and_merge.sh"

    # Step 3
    execute_step 3 "face_quality" "/root/autodl-tmp/yufei/DeepFaceLab" \
        "/root/autodl-tmp/yufei/DeepFaceLab/face_quality_filter.py"

    # Step 4
    execute_step 4 "internvideo" "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality" \
        "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/get_age_gender.py"

    # Step 5
    execute_step 5 "consisid" "/root/autodl-tmp/yufei/ConsisID" \
        "/root/autodl-tmp/yufei/ConsisID/batch_all_generation.py"

    # Step 6
    execute_step 6 "photomaker" "/root/autodl-tmp/yufei/PhotoMaker/inference_scripts" \
        "/root/autodl-tmp/yufei/PhotoMaker/inference_scripts/inference_all_output.py"

    # Step 7
    execute_step 7 "internvideo" "/root/autodl-tmp/yufei/DeepFaceLab" \
        "/root/autodl-tmp/yufei/DeepFaceLab/sync_and_run.py"

    # Step 8
    execute_step 8 "LivePortrait" "/root/autodl-tmp/yufei/LivePortrait" \
        "/root/autodl-tmp/yufei/LivePortrait/inference_all_make.py"

    # Step 9
    execute_step 9 "consisid" "/root/autodl-tmp/yufei/datasets/cekebv-hq" \
        "/root/autodl-tmp/yufei/datasets/cekebv-hq/get_all_video_base.py"

    # Step 10
    execute_step 10 "internvideo" "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality" \
        "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/video_qa_generation_all_video.py"

    # Step 11
    execute_step 11 "internvideo" "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality" \
        "/root/autodl-tmp/yufei/InternVideo/InternVideo2/multi_modality/replace_sks.py" "--sks_name '<Mi>'"

}

# 运行主流程
main

# 检查整体执行结果
if [ $? -eq 0 ]; then
    echo "All steps completed successfully!"
else
    echo "Process failed! Check the logs above for details."
    exit 1
fi