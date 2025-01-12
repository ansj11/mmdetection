

config="mask2former_r50_8xb2-fusion-50e_roi1024"

python tools/test.py configs/mask2former/${config}.py work_dirs/${config}/iter_1005000.pth --show
