# download data from wandb https://wandb.ai/nycu_pcs/ILA/
python download_save_wandb_data.py --serials 0 26 --output_file saw_stage1_lr.csv

# wandb make scripts for stage 2 (seed) from stage 1 (lr)
# python lr_script.py --input_file data\saw_stage1.csv --output_file script_stage2_saw
# python lr_script.py --input_file data\saw_pt_stage1_lr.csv --output_file script_stage2_saw_pt

python download_save_wandb_data.py
python download_save_wandb_data.py --serials 40 41 42 43 --output_file saw_cost.csv

python download_save_wandb_data_feature_metrics.py --serials 31 32


# feature metrics (interpret results based on attn/cka similarities or l2/norms)
# our has higher attention across layers in average, and 
# the deviation across layers is also higher (change across network to focus
# on discriminative regions as image is processed)

# cka similarities for norm2 outputs:
# lower -> higher accuracies
# change across layers (compared to 1st layer): expect it to be gradual
# average change for problematic models is higher, 
# also stdev of change is higher (indicates large jumps in change at certain layers)
# also second order of change (change of change between cka): mean and std 
# higher for problematic models (investigate further, more methods/datasets for future work)

# cka similarities for attention scores:
# higher cka_std_train (higher deviation across layers, indicates variation
# in attention scores between layers: no attention collapse?)
# complementary to figures in results_inference/

#  results_inference/*_31/ and *_32/ folders hold figures for:
# 1) mean and std of attention per layer, cka for norm2 outputs, and 2) cka for attention
python compute_feature_metrics_mean_std.py


# visualization of datasets
python grid_datasets.py
python grid_datasets.py --datasets leaves --save_name datasets_leaves --label_datasets
python grid_datasets.py --serial 421 --datasets soygene soylocal --save_name datasets_ufgir_classes_soygene_soylocal --dataset_classes --font_size_title 13
python grid_datasets.py --serial 421 --datasets cotton soyglobal --save_name datasets_ufgir_classes_cotton_soyglobal --dataset_classes --dataset_classes_labels "Cotton Class 1" "Cotton Class 2" "SoyGlobal Class 1" "SoyGlobal Class 2" --font_size_title 13
python grid_datasets.py --serial 421 --datasets soyageing soyglobal --save_name datasets_ufgir_classes_soyageing_soyglobal --dataset_classes --dataset_classes_labels "SoyAgeing Class 1" "SoyAgeing Class 2" "SoyGlobal Class 1" "SoyGlobal Class 2" --font_size_title 13


# visualization of attention
# Low, Mid, High (-level features)
# for low it looks normal for most
# mid ours still focuses on the leaves but others focus on background
# high: focus/attn on background the leaf is taken over (slight tint, hint)
# shortcut: similar to shortcut learning (https://arxiv.org/abs/2004.07780)
# and the issue with the mark from hospitals from chexnet (https://arxiv.org/abs/1711.05225)
# ours is not as pronounced but attention is still split between bg/leaves
python grid_attention.py --font_size_title 10

# benefits of ila+saw for 
python grid_attention.py --datasets soyageing --models vit_b16_cls_fz_30 vit_b16_cls_fz_30 vit_b16_cls_adapter_adapter_30 vit_b16_ila_dso_cls_adapter_saw_ft_30 --vis_types None rollout_0_4 rollout_0_4 rollout_0_4 --vis_labels Samples ViT Adapter Ours --save_name rollout_0_4_soyageing
python grid_attention.py --datasets soyageing --models vit_b16_cls_fz_30 vit_b16_cls_fz_30 vit_b16_cls_adapter_adapter_30 vit_b16_ila_dso_cls_adapter_saw_ft_30 --vis_types None rollout_4_8 rollout_4_8 rollout_4_8 --vis_labels Samples ViT Adapter Ours --save_name rollout_4_8_soyageing
python grid_attention.py --datasets soyageing --models vit_b16_cls_fz_30 vit_b16_cls_fz_30 vit_b16_cls_adapter_adapter_30 vit_b16_ila_dso_cls_adapter_saw_ft_30 --vis_types None rollout_8_12 rollout_8_12 rollout_8_12 --vis_labels Samples ViT Adapter Ours --save_name rollout_8_12_soyageing

# benefits of saw + ft
python grid_attention.py --datasets soyageing --models vit_b16_cls_fz_30 vit_b16_cls_fz_30 vit_b16_cls_adapter_adapter_30 vit_b16_ila_dso_cls_adapter_ila_30 vit_b16_ila_dso_cls_adapter_saw_30 vit_b16_ila_dso_cls_adapter_saw_ft_30 --vis_types None rollout_4_8 rollout_4_8 rollout_4_8 rollout_4_8 rollout_4_8 --vis_labels Samples ViT Adapter ILA SAW SAW+FT --save_name rollout_4_8_soyageing_all
python grid_attention.py --datasets soyageing --models vit_b16_cls_fz_30 vit_b16_cls_fz_30 vit_b16_cls_adapter_adapter_30 vit_b16_ila_dso_cls_adapter_ila_30 vit_b16_ila_dso_cls_adapter_saw_30 vit_b16_ila_dso_cls_adapter_saw_ft_30 --vis_types None rollout_8_12 rollout_8_12 rollout_8_12 rollout_8_12 rollout_8_12 --vis_labels Samples ViT Adapter ILA SAW SAW+FT --save_name rollout_8_12_soyageing_all

# same but only saw+ft (no saw only)
python grid_attention.py --datasets soyageing --models vit_b16_cls_fz_30 vit_b16_cls_fz_30 vit_b16_cls_adapter_adapter_30 vit_b16_ila_dso_cls_adapter_ila_30 vit_b16_ila_dso_cls_adapter_saw_ft_30 --vis_types None rollout_4_8 rollout_4_8 rollout_4_8 rollout_4_8 --vis_labels Samples ViT Adapter ILA SAW --save_name rollout_4_8_soyageing_most
python grid_attention.py --datasets soyageing --models vit_b16_cls_fz_30 vit_b16_cls_fz_30 vit_b16_cls_adapter_adapter_30 vit_b16_ila_dso_cls_adapter_ila_30 vit_b16_ila_dso_cls_adapter_saw_ft_30 --vis_types None rollout_8_12 rollout_8_12 rollout_8_12 rollout_8_12 --vis_labels Samples ViT Adapter ILA SAW --save_name rollout_8_12_soyageing_most


# motivation behind dual attention rollout (vs rollout and single layer attention)
python grid_attention.py --datasets leaves --models vit_b16_cls_fz_30 vit_b16_cls_fz_30 vit_b16_cls_fz_30 vit_b16_cls_fz_30 --vis_types None rollout_0_4 rollout_1_3 rollout_2_4 --vis_labels "Samples" "1-4" "2-3" "3-4" --save_name rollout_early_leaves_fz
python grid_attention.py --datasets leaves --models vit_b16_cls_fz_30 vit_b16_cls_fz_30 vit_b16_cls_fz_30 vit_b16_cls_fz_30 vit_b16_cls_fz_30 vit_b16_cls_fz_30 vit_b16_cls_fz_30 vit_b16_cls_fz_30 --vis_types None rollout_0_4 rollout_1_3 rollout_2_4 attention_0 attention_1 attention_2 attention_3 --vis_labels "Samples" "1-4" "2-3" "3-4" "1" "2" "3" "4" --save_name rollout_attention_early_leaves_fz

# compare early rollout for og and after mid training
python grid_attention.py --datasets leaves --models vit_b16_cls_fz_30 vit_b16_cls_fz_30 vit_b16_ila_dso_cls_adapter_saw_30 vit_b16_cls_fz_30 vit_b16_ila_dso_cls_adapter_saw_30 vit_b16_cls_fz_30 vit_b16_ila_dso_cls_adapter_saw_30 --vis_types None rollout_0_4 rollout_0_4 rollout_1_3 rollout_1_3 rollout_2_4 rollout_2_4 --vis_labels "Samples" "ViT 1-4" "SAW 1-4" "ViT 2-3" "SAW 2-3" "ViT 3-4" "SAW 3-4" --save_name rollout_early_leaves_fz_saw

# compare rollout in general for og and after mid training
python grid_attention.py --datasets leaves --models vit_b16_cls_fz_30 vit_b16_cls_fz_30 vit_b16_ila_dso_cls_adapter_saw_30 vit_b16_cls_fz_30 vit_b16_ila_dso_cls_adapter_saw_30 vit_b16_cls_fz_30 vit_b16_ila_dso_cls_adapter_saw_30 --vis_types None rollout_0_4 rollout_0_4 rollout_4_8 rollout_4_8 rollout_8_12 rollout_8_12 --vis_labels "Samples" "ViT 1-4" "SAW 1-4" "ViT 4-8" "SAW 4-8" "ViT 8-12" "SAW 8-12" --save_name rollout_all_leaves_fz_saw



# accuracies
python summarize_acc.py


# cost
# flops recorded during training are at eval mode so same as inference
# so flops_train == flops_inference in this case; future may change
python summarize_cost.py

# combine into fgirft_backbones_vit_stage2.csv
# update acc from data/csdnet_table.csv
python stack_two_df.py --input_df_1 FGIRFT\results_all\cost\cost_ufgir.csv --input_df_2 results_all\cost\cost.csv --output_file cost_combined.csv



# figures

python plot_acc_flops_params.py
python plot_acc_flops_params.py --add_text_ours_only --font_size 11 --output_file acc_vs_flops_vs_params_log_text_all

python plot_acc_flops_params.py --keep_serials 3 101 --x_var_name tp_batched --output_file acc_vs_tp_batched_vs_params_448 --log_scale_x --x_label "Inference Throughput (Images/s)" --title "Accuracy over 10 UFGIR Datasets vs Inference Throughput \n with Size Proportional to Task-Specific Parameters" --add_text_ours_only


# plots of acc vs throughput


# make plots using tably 
python tably.py data\stats_datasets.csv

python tably.py results\tables\acc_1_pivoted_mean_std_latex_cotton_soy.csv --no-escape
python tably.py results\tables\acc_1_pivoted_mean_std_latex_soyageing_subsets.csv --no-escape

python tably.py results\tables\acc_3_pivoted_mean_std_latex_cotton_soy.csv --no-escape
python tably.py results\tables\acc_3_pivoted_mean_std_latex_soyageing_subsets.csv --no-escape

python tably.py results\tables\acc_cost.csv --no-escape

python tably.py results\tables\ablation_rsds.csv --no-escape

python tably.py results\tables\ablation_ds_kernel_size.csv --no-escape
python tably.py results\tables\ablation_ds_loc.csv --no-escape

python tably.py results\tables\ablation_backbones_components.csv --no-escape
python tably.py results\tables\ablation_saw_variations.csv --no-escape

python tably.py results\tables\feature_metrics_cotton.csv --no-escape

# pseucodode for model in pseudocode.py: can change to white bg
# then insert into word: insert -> object -> opendocument text -> copy -> save with paint.net
