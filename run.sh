source /gemini/data-1/miniconda3/set_env.sh
conda activate YOSO

cp /gemini/data-1/weights/resnet50-0676ba61.pth /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth

# python -u tools/train.py configs/mask2former/mask2former_r50_fusion_8xb2-160k_ade20k-512x512.py --resume

# python -u tools/train.py configs/mask2former/mask2former_r50_8xb2-fusion-50e_roi.py # --resume

python -u tools/train.py configs/mask2former/mask2former_r50_8xb2-fusion-50e_roi1024.py --resume


