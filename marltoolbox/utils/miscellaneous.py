import inspect

from ray.rllib.agents.callbacks import DefaultCallbacks

def sequence_of_fn_wt_same_args(*args, function_list, **kwargs) -> None:
    for fn in function_list:
        fn(*args, **kwargs)


def overwrite_config(dict_: dict, key, value):
    assert isinstance(dict_, dict)
    current_value = dict_
    found = True
    for k in key.split("."):
        if not found:
            print(f'Intermediary key: {k} not found in full key: {key}')
            return
        dict_ = current_value
        if k in current_value.keys():
            current_value = current_value[k]
        else:
            found = False

    if current_value != value:
        if found:
            print(f'Overwriting (key, value): ({key},{current_value}) with value: {value}')
            dict_[k] = value
        else:
            print(f'Adding (key, value): ({key},{value}) in dict.keys: {dict_.keys()}')
            dict_[k] = value


def extract_checkpoints(tune_experiment_analysis):
    all_best_checkpoints_per_trial = [
        tune_experiment_analysis.get_best_checkpoint(trial,
                                                     metric=tune_experiment_analysis.default_metric,
                                                     mode=tune_experiment_analysis.default_mode)
        for trial in tune_experiment_analysis.trials
    ]
    return all_best_checkpoints_per_trial


def merge_callbacks(*callbacks_list):

    callbacks_list = [callback() if inspect.isclass(callback) else callback for callback in callbacks_list]

    class MergeCallBacks(DefaultCallbacks):
        def __getattribute__(self, name):
            super_attr = super().__getattribute__(name)
            if callable(super_attr):
            # if hasattr(super_attr, '__call__'):
                def newfunc(*args, **kwargs):
                    for callbacks in callbacks_list:
                        function = callbacks.__getattribute__(name)
                        function(*args, **kwargs)
                return newfunc
            else:
                return super_attr
    return MergeCallBacks


def use_seed_as_idx(list_to_select_from):
    def get_value(policy_config):
        if isinstance(policy_config["seed"], int):
            print("use_seed_as_idx", policy_config["seed"])
            return list_to_select_from[policy_config["seed"]]
        else:
            print('use_seed_as_idx default to checkpoint 0. config["seed"]:', policy_config["seed"])
            return list_to_select_from[0]
    return get_value


def check_using_tune_class(config):
    return config.get("TuneTrainerClass", None) is not None
