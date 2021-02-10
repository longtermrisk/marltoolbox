#
# from marltoolbox.utils import same_and_cross_perf
#
# def init_evaluator():
#     evaluator = same_and_cross_perf.SameAndCrossPlayEvaluator(
#         TrainerClass=None,
#         group_names=["group_name1", "group_name2"],
#         evaluation_config={},
#         stop_config={},
#         exp_name="exp_name",
#         policies_to_train=["None"],
#         policies_to_load_from_checkpoint=[],
#     )
#     return evaluator
#
# def make_fake_analysis_metrics_per_mode(n_modes):
#     fake_analysis_metrics_per_mode = []
#     for i in range(n_modes):
#         mode, group_pair_id, group_pair_name = f"mode_{i}", f"pair_id_{i}", f"pair_name_{i}"
#         available_metrics_list = None
#         fake_analysis_metrics_per_mode.append( (mode, available_metrics_list, group_pair_id, group_pair_name ) )
#
#     return fake_analysis_metrics_per_mode
#
# def test_plot_results():
#     same_cross_play_evaluator = init_evaluator()
#
#     same_cross_play_evaluator.plot_results(analysis_metrics_per_mode=)