# Random hyperparameter optimization for LuckyLoser model.


def serialize_config(config):
    for k, v in config.items():
        if callable(v):
            config[v] = v.__name__
    return config
