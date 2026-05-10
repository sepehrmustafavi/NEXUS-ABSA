import os

class Config:
    # ---------------------------------------------------------
    # 1. General Settings
    # ---------------------------------------------------------
    PROJECT_NAME = "NEXUS_ABSA_Experiment"
    RANDOM_STATE = 42
    
    # ---------------------------------------------------------
    # 2. Kaggle Offline Paths (Mounting the Dataset)
    # ---------------------------------------------------------
    KAGGLE_INPUT_BASE = "/kaggle/input/nexus_absa_kaggle_input"
    MODELS_DIR = os.path.join(KAGGLE_INPUT_BASE, "models")
    CACHE_DIR = os.path.join(KAGGLE_INPUT_BASE, "cache")
    
    CONCEPTNET_CACHE_PATH = os.path.join(CACHE_DIR, "absa_conceptnet_cache.json")
    SENTICNET_CACHE_PATH = os.path.join(CACHE_DIR, "absa_senticnet_cache.json")

    # ---------------------------------------------------------
    # 3. Training & Hyperparameters
    # ---------------------------------------------------------
    LOG_FILE = "absa_experiment_logs.txt"
    RESULTS_FILE = "absa_final_results.json"

    BATCH_SIZE = 8      
    EPOCHS = 4            
    LEARNING_RATE = 2e-5  
    MAX_LEN_SEMEVAL = 128 
    MAX_LEN_MAMS = 256    
    
    # ---------------------------------------------------------
    # 4. Experimental Scenarios List
    # ---------------------------------------------------------
    ROBERTA_PATH = os.path.join(MODELS_DIR, "roberta")
    DEBERTA_PATH = os.path.join(MODELS_DIR, "deberta")
    QWEN_PATH = os.path.join(MODELS_DIR, "qwen3.5-2b")

    SCENARIOS = [
        (1, ROBERTA_PATH, "roberta", "static", "senticnet", "mams"),
        (2, DEBERTA_PATH, "deberta", "static", "senticnet", "mams"),
        (3, QWEN_PATH, "qwen2", "static", "senticnet", "mams"), 

        (4, ROBERTA_PATH, "roberta", "flexible", "senticnet", "mams"),
        (5, DEBERTA_PATH, "deberta", "flexible", "senticnet", "mams"),
        (6, QWEN_PATH, "qwen2", "flexible", "senticnet", "mams"),

        (7, DEBERTA_PATH, "deberta", "flexible", "senticnet", "semeval"),
        (8, QWEN_PATH, "qwen2", "flexible", "senticnet", "semeval"),

        (9, DEBERTA_PATH, "deberta", "flexible", "conceptnet", "mams")
    ]