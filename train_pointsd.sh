root_path=$1
export DS_SKIP_CUDA_CHECK=1

#wandb config
#export WANDB_API_KEY=""
#export WANDB_USERNAME=

export project_name="PointSD"
export task_name="" #experiment name
export project_path="."
export code_file="train_pointsd.py" 

export model_dir="" #Path to the stable diffusion checkpoint root
export dataset_dir="" #Path to the pre-training dataset root
export img_dir="" #Path to the image dataset root

export output_dir="./checkpoints/${task_name}"
export log_dir="${project_path}/logs"
export cache_dir="${project_path}/cached/${task_name}"

if [ ! -d $log_dir ]; then
	mkdir $log_dir
fi

if [ ! -d $log_dir/$task_name ]; then
	mkdir $log_dir/$task_name
fi

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

cd $project_path &&
CUDA_VISIBLE_DEVICES=<GPUs> accelerate launch --config_file $project_path/accelerate_configs/zero2_config.yaml --use_deepspeed \
    --zero_stage 2 \
    --gradient_clipping 1.0 \
    --zero3_init_flag false \
    --deepspeed_multinode_launcher standard \
    --mixed_precision no \
    --main_process_port $PORT \
  train_pointsd.py \
  --pretrained_model_name_or_path=$model_dir \
  --train_data_dir=$dataset_dir \
  --img_dir=$img_dir \
  --resolution=64 \
  --num_train_epochs=300 \
  --num_warm_epochs=10 \
  --train_batch_size=64 \
  --scale=0.0 \
  --mixed_precision no \
  --checkpointing_steps=5000 \
  --validation_steps=500 \
  --sample_points_num=1024 \
  --train_timesteps_min=500 \
  --train_timesteps_max=1000 \
  --dataloader_num_workers=8 \
  --learning_rate=1.25e-04 --lr_scheduler="cosine" \
  --seed=0 \
  --num_tokens=4 \
  --perform_mix \
  --depth=3 \
  --stage1_ckpt='' \
  --run_stage='' \
  --tracker_project_name=$project_name \
  --output_dir=$output_dir \
 2>&1 | tee -a $log_dir/$task_name/$HOSTNAME-`date +"%Y-%m-%d_%H_%M_%S"`.log
