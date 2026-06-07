# Experiments
## TD3 Local
```bash
# BASELINE RUN
conda activate lead_carla_fork
export CARLA_ROOT=/home/erikhbj/Documents/master/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_local_residual_td3_baseline_2env.env \
bash rl_finetuning/train_tfv6_td3_parallel.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_TD3_LOCAL_RESIDUAL_8_baseline_T1213 \
  --debug 0 --debug_viz --debug_viz_every_n 100000 --debug_viz_burst_len 3000 # --activate_cnn_heatmap
```

## TD3 VC06
### UTD4 actor0,5
```bash
# VC06 needs to resume
# GPU0
source ~/.bashrc
conda activate lead_carla_fork
export CARLA_ROOT=/data/work/erikhbj/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_vc06_residual_td3_baseline_gpu0_4env_UTD4_ac1.env \
bash rl_finetuning/train_tfv6_td3_parallel.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_TD3_VC06_RESIDUAL_7_baseline_UTD4_actor1  \
  --debug 0 --debug_viz --debug_viz_every_n 100000 --debug_viz_burst_len 3000
```

### Speedhistory
```bash
# VC06 needs to resume
# GPU1
source ~/.bashrc
conda activate lead_carla_fork
export CARLA_ROOT=/data/work/erikhbj/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_vc06_residual_td3_baseline_gpu1_4env_speedhistory.env \
bash rl_finetuning/train_tfv6_td3_parallel.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_TD3_VC06_RESIDUAL_6_baseline_4env_speedhistory \
  --debug 0 --debug_viz --debug_viz_every_n 100000 --debug_viz_burst_len 2000
```

## TD3 idun baseline all improvements
### TD3 idun baseline all improvements Left Turn
```bash
# Idun job 24584634 and 24597074 (failed repeated startups) and 24602234
sbatch --job-name=TD3_idun_baseline_allimprovements \
  --export=ALL,EXP_NAME=TD3_idun_baseline_allimprovements,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_IMPROVED_nstep3_nowarm_terminalhint35_regular002_speedhist_criticstopsignprivil_gradclip50.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```

And with harsh terminal hint 50
```bash
# Idun job 24604131
sbatch --job-name=TD3_idun_baseline_allimprovements_terminal50 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_allimprovements_terminal50,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_IMPROVED2_nstep3_nowarm_terminalhint500_regular002_speedhist_criticstopsignprivil_gradclip50_sparsestylereward.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```

And with harsh terminal hint 15
```bash
# Idun job 24621368
sbatch --job-name=TD3_idun_baseline_allimprovements_terminal15 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_allimprovements_terminal15,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_IMPROVED2_nstep3_nowarm_terminalhint150_regular002_speedhist_criticstopsignprivil_gradclip50_sparsestylereward.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
And with terminal hint 7.5
```bash
# Idun job 24639381 and then 24689917
sbatch --job-name=TD3_idun_baseline_allimprovements_terminal7dot5 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_allimprovements_terminal7dot5,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_IMPROVED2_nstep3_nowarm_terminalhint75_regular002_speedhist_criticstopsignprivil_gradclip50_sparsestylereward.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
And with actor depth 2:
```bash
# Idun job 24698300
sbatch --job-name=TD3_idun_baseline_allimprovements_actor2layer \
  --export=ALL,EXP_NAME=TD3_idun_baseline_allimprovements_actor2layer,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_IMPROVED_nstep3_nowarm_terminalhint35_regular002_speedhist_criticstopsignprivil_gradclip50_actor2layer.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
+weather:
```bash
# Idun job 24700350
sbatch --job-name=TD3_idun_baseline_allimprovements_actor2layer_weather \
  --export=ALL,EXP_NAME=TD3_idun_baseline_allimprovements_actor2layer_weather,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_IMPROVED_nstep3_nowarm_terminalhint35_regular002_speedhist_criticstopsignprivil_gradclip50_actor2layer_weather.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```

actor depth 2 and hidden size 1024
```bash
# Idun job 24700347
sbatch --job-name=TD3_idun_baseline_allimprovements_actor2layer_hidden1024 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_allimprovements_actor2layer_hidden1024,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_IMPROVED_nstep3_nowarm_terminalhint35_regular002_speedhist_criticstopsignprivil_gradclip50_actor2layer_hidden1024.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
+weather:
```bash
# Idun job 24700352
sbatch --job-name=TD3_idun_baseline_allimprovements_actor2layer_hidden1024_weather \
  --export=ALL,EXP_NAME=TD3_idun_baseline_allimprovements_actor2layer_hidden1024_weather,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_IMPROVED_nstep3_nowarm_terminalhint35_regular002_speedhist_criticstopsignprivil_gradclip50_actor2layer_hidden1024_weather.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
actor2layer hidden 1024 and adaptiveavgpool
```bash
# Idun job 24700353
sbatch --job-name=TD3_idun_baseline_allimprovements_actor2layer_hidden1024_adaptiveavgpool_weather \
  --export=ALL,EXP_NAME=TD3_idun_baseline_allimprovements_actor2layer_hidden1024_adaptiveavgpool_weather,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_IMPROVED_nstep3_nowarm_terminalhint35_regular002_speedhist_criticstopsignprivil_gradclip50_actor2layer_hidden1024_adaptiveavgpool_weather.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```


### TD3 idun baseline all improvements 4LT with weather.
Regular
```bash
# Idun job 24639464
sbatch --job-name=TD3_idun_baseline_allimprovements_B2D4LT_WEATHER \
  --export=ALL,EXP_NAME=TD3_idun_baseline_IMPROVED_B2D4LT_WEATHER,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_IMPROVED_B2D4LT_WEATHER.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
And then with terminal hint 15:
```bash
# Idun job 24639465 (WRONG in config) re-run: 24685954
sbatch --job-name=TD3_idun_baseline_allimprovements_B2D4LT_WEATHER_term150 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_IMPROVED_B2D4LT_WEATHER_term150,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_IMPROVED_B2D4LT_WEATHER_terminalhint150.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
And then with terminal hint 7.5:
```bash
# Idun job 24639507 (WRONG in config), re-run: 24685944
sbatch --job-name=TD3_idun_baseline_allimprovements_B2D4LT_WEATHER_term75 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_IMPROVED_B2D4LT_WEATHER_term75,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_IMPROVED_B2D4LT_WEATHER_terminalhint75.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
Try with the adaptive avg pool.
```bash
# Idun job 24672415
sbatch --job-name=TD3_idun_baseline_allimprovements_B2D4LT_WEATHER_avgpool32 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_IMPROVED_B2D4LT_WEATHER_avgpool32,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_IMPROVED_B2D4LT_WEATHER_avgpool32.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
Now with actor2layer
```bash
# Idun job 24698302
sbatch --job-name=TD3_idun_baseline_allimprovements_B2D4LT_WEATHER_actor2layer \
  --export=ALL,EXP_NAME=TD3_idun_baseline_IMPROVED_B2D4LT_WEATHER_actor2layer,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_IMPROVED_B2D4LT_WEATHER_actor2layer.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```


### TD3 idun baseline all improvements All intersection
```bash
# Idun job 24589971
sbatch --job-name=TD3_idun_baseline_allimprovements_allintersection \
  --export=ALL,EXP_NAME=TD3_idun_baseline_allimprovements_allintersection,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_IMPROVED_ALLINTERSECTION.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```


## TD3 idun baseline experiments
### n-step returns +-> no penalty warmup +-> terminalhint 3,5
```bash
# Idun job 24379137
sbatch --job-name=TD3_idun_baseline_nstep3 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_nstep3,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_nstep3.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
# Idun job 24379138
sbatch --job-name=TD3_idun_baseline_nstep3_nopenaltywarm \
  --export=ALL,EXP_NAME=TD3_idun_baseline_nstep3_nopenaltywarm,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_nstep3_nowarm.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
# Idun job 24379139
sbatch --job-name=TD3_idun_baseline_nstep3_nopenaltywarm_terminalhint35 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_nstep3_nopenaltywarm_terminalhint35,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_nstep3_nowarm_terminalhint35.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```

### Regularization term
```bash
# coef 0.1
# Idun job 24377929
sbatch --job-name=TD3_idun_baseline_regularization \
  --export=ALL,EXP_NAME=TD3_idun_baseline_regularization,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_bcregularization.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
```bash
# coef 0.02
# Idun job 24398958
sbatch --job-name=TD3_idun_baseline_regularization002 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_regularization002,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_bcregularization002.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```


### Speed history
```bash
# Idun job 24400087
sbatch --job-name=TD3_idun_speedhistory \
  --export=ALL,EXP_NAME=TD3_idun_baseline_speedhistory,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_speedhistory.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```

### UTD2a0.5
```bash
# Idun job 24400088
sbatch --job-name=TD3_idun_baseline_utd2a05 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_utd2a05,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_utd2a05.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```

### UTD4a1
```bash
# Idun job 24400090
sbatch --job-name=TD3_idun_baseline_utd4a1 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_utd4a1,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_utd4a1.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
### Base policy deterministic run
```bash
# Idun job 24377931
sbatch --job-name=TD3_idun_basepolicy_noexplore \
  --export=ALL,EXP_NAME=TD3_idun_basepolicy_noexplore,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASEPOLICY_NOEXPLORE.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
### Baseline run
```bash
# Idun job 24400224
sbatch --job-name=TD3_idun_baseline \
  --export=ALL,EXP_NAME=TD3_idun_baseline,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
### Baseline run with weather
```bash
# Idun job 24699893
sbatch --job-name=TD3_idun_baseline_weather \
  --export=ALL,EXP_NAME=TD3_idun_baseline_weather,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_weather.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```

### Critic privil stop signs
```bash
# Idun job 24521052 and 24560198
sbatch --job-name=TD3_idun_baseline_criticstopsignprivil \
  --export=ALL,EXP_NAME=TD3_idun_baseline_criticstopsignprivil,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_criticstopsignprivil.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
### TTC reward
```bash
# Idun job 24521054 and then 24560186
sbatch --job-name=TD3_idun_baseline_ttc \
  --export=ALL,EXP_NAME=TD3_idun_baseline_ttc,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_ttc.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```

### Grad clip 50
```bash
# Idun job 24541060 and then 24584537
sbatch --job-name=TD3_idun_baseline_gradclip50 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_gradclip50,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_gradclip50.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```

### gamma 0999
```bash
# Idun job 24541061
sbatch --job-name=TD3_idun_baseline_gamma0999 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_gamma0999,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_gamma0999.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
### penalty 50 and 15 and 7.5
penalty 50
```bash
# Idun job 24604132
sbatch --job-name=TD3_idun_baseline_terminalpen50_nopenaltywarm \
  --export=ALL,EXP_NAME=TD3_idun_baseline_terminalpen50_nopenaltywarm,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_terminalpen50_nopenaltywarm.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
penalty 15
```bash
# Idun job 24621373
sbatch --job-name=TD3_idun_baseline_terminalpen15_nopenaltywarm \
  --export=ALL,EXP_NAME=TD3_idun_baseline_terminalpen15_nopenaltywarm,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_terminalpen15_nopenaltywarm.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
penalty 7.5
```bash
# Idun job 24639390
sbatch --job-name=TD3_idun_baseline_terminalpen7dot5_nopenaltywarm \
  --export=ALL,EXP_NAME=TD3_idun_baseline_terminalpen7dot5_nopenaltywarm,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_terminalpen7dot5_nopenaltywarm.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```

### Actor hidden layers depth 2
```bash
# Idun job 24698299
sbatch --job-name=TD3_idun_baseline_actor2layer \
  --export=ALL,EXP_NAME=TD3_idun_baseline_actor2layer,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_actor2layer.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
And with width 1024 as well:
```bash
# Idun job 24700346
sbatch --job-name=TD3_idun_baseline_actor2layer_hidden1024 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_actor2layer_hidden1024,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_actor2layer_hidden1024.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```


### Route activated
```bash
# Idun job 24382372 and then 24406797  (with debug viz every 200k)
sbatch --job-name=TD3_idun_basepolicy_withroute \
  --export=ALL,EXP_NAME=TD3_idun_basepolicy_withroute,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_withroute.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
```bash
# LOCAL 5090 RUN
conda activate lead_carla_fork
export CARLA_ROOT=/home/erikhbj/Documents/master/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_local_residual_td3_baseline_2env_withroute.env \
bash rl_finetuning/train_tfv6_td3_parallel.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_TD3_LOCAL_RESIDUAL_8_baseline_T1213_withroute \
  --debug 0 --debug_viz --debug_viz_every_n 100000 --debug_viz_burst_len 2000 # --activate_cnn_heatmap
```

## Transformer
```bash
# VC06 gpu0
source ~/.bashrc
conda activate lead_carla_fork
export CARLA_ROOT=/data/work/erikhbj/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_vc06_residual_td3_baseline_gpu0_transformer_UTD4a1_regularization001_criticprivstopsign.env \
bash rl_finetuning/train_tfv6_td3_parallel.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_TD3_VC06_RESIDUAL_8_transformer_UTD4a1_speedhistory_regularization_criticprivstopsign  \
  --debug 0 --debug_viz --debug_viz_every_n 100000 --debug_viz_burst_len 2000
```
```bash
# VC06 gpu1
source ~/.bashrc
conda activate lead_carla_fork
export CARLA_ROOT=/data/work/erikhbj/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_vc06_residual_td3_baseline_gpu1_transformer_concatcoeftoqhead.env \
bash rl_finetuning/train_tfv6_td3_parallel.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_TD3_VC06_RESIDUAL_8_transformer_concatcoefactiontoqhead_speedhistory  \
  --debug 0 --debug_viz --debug_viz_every_n 100000 --debug_viz_burst_len 2000
```


## CNN size ablations
### Adaptive avgpool2d23
```bash
# Idun job 24447077 and 24520841
sbatch --job-name=TD3_idun_baseline_CNN_adaptiveavgpool2d23 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_CNN_adaptiveavgpool2d23,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_CNN_adaptiveavgpool2d23.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
### Adaptive headhidden1024
```bash
# Idun job 24447081
sbatch --job-name=TD3_idun_baseline_CNN_headhidden1024 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_CNN_headhidden1024,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_CNN_headhidden1024.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
### Adaptive CNN_stridedpool
```bash
# Idun job 24447080
sbatch --job-name=TD3_idun_baseline_CNN_stridedpool \
  --export=ALL,EXP_NAME=TD3_idun_baseline_CNN_stridedpool,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_CNN_stridedpool.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
### Adaptive CNN_width128
```bash
# Idun job 24447078
sbatch --job-name=TD3_idun_baseline_CNN_width128 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_CNN_width128,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_CNN_width128.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
### MIX: avgpool2d23 and width128
```bash
# Idun job 24520866
sbatch --job-name=TD3_idun_baseline_CNN_avgpool2d23_width128 \
  --export=ALL,EXP_NAME=TD3_idun_baseline_CNN_avgpool2d23_width128,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_BASELINE_CNN_avgpool2d23_width128.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```

## VC06 CNN
```bash
# VC06 gpu0
source ~/.bashrc
conda activate lead_carla_fork
export CARLA_ROOT=/data/work/erikhbj/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_vc06_residual_td3_baseline_gpu0_CNN_stridedpool_convwidth128.env \
bash rl_finetuning/train_tfv6_td3_parallel.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_TD3_VC06_RESIDUAL_9_CNN_stridedpool_convwidth128  \
  --debug 0 --debug_viz --debug_viz_every_n 200000 --debug_viz_burst_len 2000
```
```bash
# VC06 gpu1
source ~/.bashrc
conda activate lead_carla_fork
export CARLA_ROOT=/data/work/erikhbj/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_vc06_residual_td3_baseline_gpu1_CNN_adaptiveavgpool2d23_convwidth128.env \
bash rl_finetuning/train_tfv6_td3_parallel.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_TD3_VC06_RESIDUAL_9_CNN_adaptiveavgpool2d23_convwidth128  \
  --debug 0 --debug_viz --debug_viz_every_n 100000 --debug_viz_burst_len 2000
```



## VC06 Reward
Improved+reward
```bash
# VC06 gpu0
source ~/.bashrc
conda activate lead_carla_fork
export CARLA_ROOT=/data/work/erikhbj/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_vc06g0_residual_td3_IMPROVED2_nstep3_nowarm_terminalhint500_regular002_speedhist_criticstopsignprivil_gradclip50_sparsestylereward.env \
bash rl_finetuning/train_tfv6_td3_parallel.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_TD3_VC06_RESIDUAL_10_terminalhint50  \
  --debug 0 --debug_viz --debug_viz_every_n 500000 --debug_viz_burst_len 2000
```
Just reward
```bash
# VC06 gpu1
source ~/.bashrc
conda activate lead_carla_fork
export CARLA_ROOT=/data/work/erikhbj/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_vc06g1_residual_td3_BASELINE_terminalpen50_nopenaltywarm.env \
bash rl_finetuning/train_tfv6_td3_parallel.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_TD3_VC06_RESIDUAL_10_baseline_terminalhint50  \
  --debug 0 --debug_viz --debug_viz_every_n 500000 --debug_viz_burst_len 2000
```
### Hint 25
```bash
# VC06 gpu0
source ~/.bashrc
conda activate lead_carla_fork
export CARLA_ROOT=/data/work/erikhbj/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_vc06g0_residual_td3_IMPROVED2_nstep3_nowarm_terminalhint250_regular002_speedhist_criticstopsignprivil_gradclip50_sparsestylereward.env \
bash rl_finetuning/train_tfv6_td3_parallel.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_TD3_VC06_RESIDUAL_10_improved_terminalhint25  \
  --debug 0 --debug_viz --debug_viz_every_n 500000 --debug_viz_burst_len 2000
```

### Hint 10
```bash
# VC06 gpu1
source ~/.bashrc
conda activate lead_carla_fork
export CARLA_ROOT=/data/work/erikhbj/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_vc06g1_residual_td3_IMPROVED2_nstep3_nowarm_terminalhint100_regular002_speedhist_criticstopsignprivil_gradclip50_sparsestylereward.env \
bash rl_finetuning/train_tfv6_td3_parallel.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_TD3_VC06_RESIDUAL_10_improved_terminalhint10  \
  --debug 0 --debug_viz --debug_viz_every_n 500000 --debug_viz_burst_len 2000
```

## VC06 both GPUS
### Actor 2 layers.
```bash
source ~/.bashrc
conda activate lead_carla_fork
export CARLA_ROOT=/data/work/erikhbj/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_vc06_residual_baseline_improvements_all4lt_weather_actor2layer.env \
bash rl_finetuning/train_tfv6_td3_parallel.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_TD3_VC06_RESIDUAL_baseline_improvements_all4lt_weather_actor2layer  \
  --debug 0 --debug_viz --debug_viz_every_n 500000 --debug_viz_burst_len 2000
```
```bash
source ~/.bashrc
conda activate lead_carla_fork
export CARLA_ROOT=/data/work/erikhbj/master-spring-ad-lead/3rd_party/fork_export_t1213_fixed/LinuxNoEditor
unset PYTHONPATH
RUN_CONFIG_FILE=rl_finetuning/configs/train_vc06_residual_baseline_improvements_all4lt_weather_actor2layer_hidden1024.env \
bash rl_finetuning/train_tfv6_td3_parallel.sh \
  outputs/checkpoints/tfv6_resnet34 \
  outputs/rl_logs \
  TFV6_TD3_VC06_RESIDUAL_baseline_improvements_all4lt_weather_actor2layer_hidden1024  \
  --debug 0 --debug_viz --debug_viz_every_n 500000 --debug_viz_burst_len 2000
```


## VC06 verify

## Idun allintersection baseline
```bash
# Idun job 24465931
sbatch --job-name=TD3_idun_allintersection_baseline \
  --export=ALL,EXP_NAME=TD3_idun_allintersection_baseline,RUN_CONFIG_FILE=rl_finetuning/configs/train_idun_residual_td3_ALLINTERSECTION_BASELINE.env \
  rl_finetuning/train_idun_residual_td3_BASELINE.slurm
```
