# bach
python -u tools/train.py --serial 1 --seed 10 --cfg configs/bach_ft_weakaugs.yaml --lr 0.1 --model_name vit_b16 --freeze_backbone --classifier cls --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/bach_ft_weakaugs.yaml --lr 0.1 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vpt_deep --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/bach_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --adapter adapter --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/bach_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vpt_shallow --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/bach_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vqt --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/bach_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --adapter convpass --opt sgd --weight_decay 0 --cpu_workers 8

# bdctkidney
python -u tools/train.py --serial 1 --seed 10 --cfg configs/bdctkidney_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vpt_deep --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/bdctkidney_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --adapter adapter --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/bdctkidney_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/bdctkidney_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vpt_shallow --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/bdctkidney_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vqt --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/bdctkidney_ft_weakaugs.yaml --lr 0.1 --model_name vit_b16 --freeze_backbone --classifier cls --adapter convpass --opt sgd --weight_decay 0 --cpu_workers 8

# gzcxray
python -u tools/train.py --serial 1 --seed 10 --cfg configs/gzcxray_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/gzcxray_ft_weakaugs.yaml --lr 0.1 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vpt_deep --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/gzcxray_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --adapter adapter --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/gzcxray_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vpt_shallow --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/gzcxray_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vqt --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/gzcxray_ft_weakaugs.yaml --lr 0.1 --model_name vit_b16 --freeze_backbone --classifier cls --adapter convpass --opt sgd --weight_decay 0 --cpu_workers 8

# idriddr
python -u tools/train.py --serial 1 --seed 10 --cfg configs/idriddr_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/idriddr_ft_weakaugs.yaml --lr 0.01 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vpt_deep --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/idriddr_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --adapter adapter --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/idriddr_ft_weakaugs.yaml --lr 0.1 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vpt_shallow --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/idriddr_ft_weakaugs.yaml --lr 0.003 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vqt --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/idriddr_ft_weakaugs.yaml --lr 0.003 --model_name vit_b16 --freeze_backbone --classifier cls --adapter convpass --opt sgd --weight_decay 0 --cpu_workers 8

# idridme
python -u tools/train.py --serial 1 --seed 10 --cfg configs/idridme_ft_weakaugs.yaml --lr 0.003 --model_name vit_b16 --freeze_backbone --classifier cls --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/idridme_ft_weakaugs.yaml --lr 0.01 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vpt_deep --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/idridme_ft_weakaugs.yaml --lr 0.03 --model_name vit_b16 --freeze_backbone --classifier cls --adapter adapter --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/idridme_ft_weakaugs.yaml --lr 0.01 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vpt_shallow --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/idridme_ft_weakaugs.yaml --lr 0.03 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vqt --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/idridme_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --adapter convpass --opt sgd --weight_decay 0 --cpu_workers 8

# lublung
python -u tools/train.py --serial 1 --seed 10 --cfg configs/lublung_ft_weakaugs.yaml --lr 0.01 --model_name vit_b16 --freeze_backbone --classifier cls --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/lublung_ft_weakaugs.yaml --lr 0.03 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vpt_deep --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/lublung_ft_weakaugs.yaml --lr 0.1 --model_name vit_b16 --freeze_backbone --classifier cls --adapter adapter --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/lublung_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vpt_shallow --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/lublung_ft_weakaugs.yaml --lr 0.1 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vqt --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/lublung_ft_weakaugs.yaml --lr 0.1 --model_name vit_b16 --freeze_backbone --classifier cls --adapter convpass --opt sgd --weight_decay 0 --cpu_workers 8

# mhist
python -u tools/train.py --serial 1 --seed 10 --cfg configs/mhist_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/mhist_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vpt_deep --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/mhist_ft_weakaugs.yaml --lr 0.1 --model_name vit_b16 --freeze_backbone --classifier cls --adapter adapter --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/mhist_ft_weakaugs.yaml --lr 0.1 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vpt_shallow --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/mhist_ft_weakaugs.yaml --lr 0.01 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vqt --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/mhist_ft_weakaugs.yaml --lr 0.1 --model_name vit_b16 --freeze_backbone --classifier cls --adapter convpass --opt sgd --weight_decay 0 --cpu_workers 8

# shoct
python -u tools/train.py --serial 1 --seed 10 --cfg configs/shoct_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/shoct_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vpt_deep --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/shoct_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vpt_shallow --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/shoct_ft_weakaugs.yaml --lr 0.1 --model_name vit_b16 --freeze_backbone --classifier cls --adapter adapter --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/shoct_ft_weakaugs.yaml --lr 0.3 --model_name vit_b16 --freeze_backbone --classifier cls --prompt vqt --opt sgd --weight_decay 0 --cpu_workers 8
python -u tools/train.py --serial 1 --seed 10 --cfg configs/shoct_ft_weakaugs.yaml --lr 0.03 --model_name vit_b16 --freeze_backbone --classifier cls --adapter convpass --opt sgd --weight_decay 0 --cpu_workers 8

