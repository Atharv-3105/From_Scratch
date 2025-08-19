class Config:
    exp_name = "DQN_Taxi"
    env_name = "Taxi-v3"
    seed = 42
    
    #Training Parameters
    total_timesteps = int(4e6)
    learning_rate = 2.5e-4
    buffer_size = 20000
    gamma = 0.99
    tau = 1.0
    target_network_freq = 50
    batch_size = 128
    start_e = 1.0
    end_e = 0.05
    exploration_fraction = 0.5
    learning_starts = 1000
    train_frequency = 4
    
    #Logging & Saving
    capture_video = True
    save_model = True
    upload_model = True
    hf_entity = ""
    
    #WandB configuration
    use_wandb = True
    wandb_entity = ""
    wandb_project = "DQN_Taxi"