# All Yellow 

`results/all_yellow/`

- PPORMLearning: `python /home/ubuntu/rm-marl/baselines/ppo/new_stack/ppo.py --rm --rm_learning`
- PPORM `python /home/ubuntu/rm-marl/baselines/ppo/new_stack/ppo.py --rm`
- PPORMLearningProp `python /home/ubuntu/rm-marl/baselines/ppo/new_stack/ppo.py --rm --rm_learning --prop`
- PPO `python /home/ubuntu/rm-marl/baselines/ppo/new_stack/ppo.py`  !TODO
- PPOLSTMProp `python /home/ubuntu/rm-marl/baselines/ppo/new_stack/ppo.py --use_lstm --use_labels`
- PPOLSTM `python /home/ubuntu/rm-marl/baselines/ppo/new_stack/ppo.py --use_lstm`

- PPORMLearningTransfer1 `python /home/ubuntu/rm-marl/baselines/ppo/new_stack/ppo.py --rm --rm_learning --restore`
- PPORMLearningTransfer2 `python /home/ubuntu/rm-marl/baselines/ppo/new_stack/ppo.py --rm --restore`
- PPORMLearningTransfer3 `python /home/ubuntu/rm-marl/baselines/ppo/new_stack/ppo.py --rm --restore --policies_to_train=rm_state_u0`


tensorboard --logdir_spec=no_learning:~/ray_results/PPORM_2024-09-08_08-14-54/PPORM_rm-marl_RMEnv_6db93_00000_0_2024-09-08_08-14-54,full_learning:/home/ubuntu/ray_results/PPORMLearning_2024-09-17_20-21-17/PPORMLearning_rm-marl_RMEnv_65353_00000_0_2024-09-17_20-21-18

tensorboard --logdir_spec=no_learning:/home/ubuntu/ray_results/PPORM_2024-09-20_15-11-13/PPORM_rm-marl_RMEnv_9366f_00000_0_2024-09-20_15-11-13,full_learning:/home/ubuntu/ray_results/PPORMLearning_2024-09-17_20-21-17/PPORMLearning_rm-marl_RMEnv_65353_00000_0_2024-09-17_20-21-18


# 2 Yellow
tensorboard --logdir_spec=rm:/home/ubuntu/ray_results/PPORM_2024-09-28_14-10-31/PPORM_rm-marl_RMEnv_6c1ec_00000_0_2024-09-28_14-10-32,rm_learning:/home/ubuntu/ray_results/PPORMLearning_2024-09-27_16-56-13/PPORMLearning_rm-marl_RMEnv_674be_00000_0_2024-09-27_16-56-13,rm_learning_prop:/home/ubuntu/ray_results/PPORMLearning_2024-09-28_19-51-26/PPORMLearning_rm-marl_RMEnv_0c4e5_00000_0_2024-09-28_19-51-27,crm:/home/ubuntu/ray_results/PPORMLearning_2024-09-30_21-00-54/PPORMLearning_rm-marl_RMEnv_15410_00000_0_2024-09-30_21-00-54

# 4 Yellow
tensorboard --logdir_spec=rm_learning_2:/home/ubuntu/ray_results/PPORMLearning_2024-09-27_16-56-13/PPORMLearning_rm-marl_RMEnv_674be_00000_0_2024-09-27_16-56-13,rm_learning_4:/home/ubuntu/ray_results/PPORMLearning_2024-09-30_15-06-44/PPORMLearning_rm-marl_RMEnv_9b2fe_00000_0_2024-09-30_15-06-44,rm_learning_8:/home/ubuntu/ray_results/PPORMLearning_2024-10-02_01-46-32/PPORMLearning_rm-marl_RMEnv_26737_00000_0_2024-10-02_01-46-32,restore_4:/home/ubuntu/ray_results/PPORM_2024-09-29_16-51-57/PPORM_rm-marl_RMEnv_23712_00000_0_2024-09-29_16-51-57,restore_4_all_policies:/home/ubuntu/ray_results/PPORM_2024-09-30_07-48-06/PPORM_rm-marl_RMEnv_543a2_00000_0_2024-09-30_07-48-06,restore_8:/home/ubuntu/ray_results/PPORM_2024-10-01_18-26-11/PPORM_rm-marl_RMEnv_a290a_00000_0_2024-10-01_18-26-11,restore_8_all_policies:/home/ubuntu/ray_results/PPORM_2024-10-04_07-07-53/PPORM_rm-marl_RMEnv_60051_00000_0_2024-10-04_07-07-54 --port 6007

/home/ubuntu/ray_results/PPORM_2024-10-04_07-07-53/PPORM_rm-marl_RMEnv_60051_00000_0_2024-10-04_07-07-54

tensorboard --logdir_spec=v5_rm_learning:/home/ubuntu/ray_results/PPORMLearning_2024-10-05_17-12-53/PPORMLearning_rm-marl_RMEnv_0ed63_00000_0_2024-10-05_17-12-53,v5-ppo:/home/ubuntu/ray_results/PPO_2024-10-07_21-48-16/PPO_rm-marl_FOLMultiRoom-AllRoom-v5_dbc6a_00000_0_2024-10-07_21-48-16,v3-rm-learning-fixed-obj:/home/ubuntu/ray_results/PPORMLearning_2024-10-08_20-40-50/PPORMLearning_rm-marl_RMEnv_9adce_00000_0_2024-10-08_20-40-50,v3-rm_learning:/home/ubuntu/ray_results/PPORMLearning_2024-09-27_16-56-13/PPORMLearning_rm-marl_RMEnv_674be_00000_0_2024-09-27_16-56-13,crm-transfer:/home/ubuntu/ray_results/PPORM_2024-10-09_19-27-23/PPORM_rm-marl_RMEnv_826c1_00000_0_2024-10-09_19-27-23

/home/ubuntu/ray_results/PPORMLearning_2024-10-08_20-40-50/PPORMLearning_rm-marl_RMEnv_9adce_00000_0_2024-10-08_20-40-50