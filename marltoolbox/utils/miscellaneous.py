
def sequence_of_fn_wt_same_args(function_list, *args, **kwargs) -> None:
    for fn in function_list:
        fn(*args, **kwargs)
