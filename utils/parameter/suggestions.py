import torch.optim as optim

def suggest_categorical(trial, categories, name):
    category = trial.suggest_categorical(name, categories)
    return category

def suggest_optimizer(trial, model, name):
    weight_decay = trial.suggest_loguniform(f'{name}_weight_decay', 1e-10, 1e-3)
    adam_lr = trial.suggest_loguniform(f'{name}_adam_lr', 1e-5, 1e-1)
    optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
    return optimizer

def suggest_uniform_list(trial, low, high, q, size, name):
    ret = [int(trial.suggest_discrete_uniform(f'{name}_{i}', low, high, q))
           for i in range(size)]
    return ret

def suggest_loguniform_list(trial, low, high, size, name):
    ret = [trial.suggest_loguniform(f'{name}_{i}', low, high)
           for i in range(size)]
    return ret

def suggest_uniform(trial, low, high, q, name):
    ret = int(trial.suggest_discrete_uniform(f'{name}', low, high, q))
    return ret

def suggest_loguniform(trial, low, high, name):
    ret = trial.suggest_loguniform(f'{name}', low, high)
    return ret
