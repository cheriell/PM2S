 # Experimental configurations during model training.
training_configs = {

    'beat': {
        'learning_rate': 1e-3,
        'dropout': 0.15,
        'max_length': 500,
        'batch_size': 32,
        'gpus': [0,1,2,3],
        'num_workers': 4,
    },

    'quantisation': {
        'learning_rate': 1e-3,
        'dropout': 0.15,
        'max_length': 500,
        'batch_size': 32,
        'gpus': [0,1,2,3],
        'num_workers': 4,
    },

    'hand_part': {
        'learning_rate': 1e-3,
        'dropout': 0.15,
        'max_length': 500,
        'batch_size': 32,
        'gpus': [0,1,2,3],
        'num_workers': 4,
    },

    'key_signature': {
        'learning_rate': 1e-3,
        'dropout': 0.15,
        'max_length': 500,
        'batch_size': 32,
        'gpus': [0,1,2,3],
        'num_workers': 4,
    },
    
    'time_signature': {
        'learning_rate': 1e-3,
        'dropout': 0.15,
        'max_length': 100,
        'batch_size': 512,
        'gpus': [0,1,2,3],
        'num_workers': 4,
    },
    
}