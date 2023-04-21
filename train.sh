# Training parameters
num_epochs=40
batch_size=32
lr=5e-4
weight_decay=1e-5
momentum=0.9

betas_lower=0.9
betas_upper=0.999
gamma=0.5
step_size=10
# Saving parameters
ckpt_save_path='./ckpts'
ckpt_prefix='cktp_epoch_'
ckpt_save_freq=10
report_path='./reports'
data_paths='./data/lgg-mri-segmentation/kaggle_3m/'
regex_image_paths='\.\/data\/lgg-mri-segmentation\/kaggle_3m\/.*.*_.*_.*\/.*_(.*_.*_.*)_(\d*)\.tif'

python train.py --batch-size $batch_size \
    --lr $lr \
    --weight-decay $weight_decay \
    --num-epochs $num_epochs \
    --ckpt-save-path $ckpt_save_path \
    --ckpt-prefix $ckpt_prefix \
    --ckpt-save-freq $ckpt_save_freq \
    --report-path $report_path \
    --data-paths $data_paths \
    --regex-image-paths $regex_image_paths \
    --momentum $momentum \
    --betas $betas_lower $betas_upper \
    --gamma $gamma \
    --step-size $step_size
