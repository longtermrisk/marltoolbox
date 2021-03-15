# Current training configs

# ipython -- train.py --name=test_selfish --enable_cheap_comm --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --response_entropy_reg 0.05
# ipython -- train.py --name=test_prosociality_level_0.5_wo_med --enable_cheap_comm --utterance_entropy_reg=0.05 --proposal_entropy_reg=0.05 --response_entropy_reg=0.05 --scale_before_redist --prosociality_level=0.5
# ipython -- train.py --name=test_prosociality_level_0.3_wo_med --enable_cheap_comm --utterance_entropy_reg=0.05 --proposal_entropy_reg=0.05 --response_entropy_reg=0.05 --scale_before_redist --prosociality_level=0.3

# ipython -- train.py --name=test_selfish_w_med --enable_cheap_comm --enable_arbitrator --arbitrator_main_loss_coeff 6.1e-8 --arbitrator_entropy_reg 3.05e-7 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --response_entropy_reg 0.05
# ipython -- train.py --name=test_prosociality_level_0.5_w_med --enable_cheap_comm --utterance_entropy_reg=0.05 --proposal_entropy_reg=0.05 --response_entropy_reg=0.05 --scale_before_redist --prosociality_level=0.5 --enable_arbitrator --arbitrator_main_loss_coeff 6.1e-8 --arbitrator_entropy_reg 3.05e-7
# ipython -- train.py --name=test_prosociality_level_0.3_w_med --enable_cheap_comm --utterance_entropy_reg=0.05 --proposal_entropy_reg=0.05 --response_entropy_reg=0.05 --scale_before_redist --prosociality_level=0.3 --enable_arbitrator --arbitrator_main_loss_coeff 6.1e-8 --arbitrator_entropy_reg 3.05e-7response_


ipython -- train.py --name=debug --enable_cheap_comm --utterance_entropy_reg=0.05 --proposal_entropy_reg=0.05 --response_entropy_reg=0.05 --scale_before_redist --prosociality_level=0.5 --enable_arbitrator --arbitrator_main_loss_coeff 6.1e-8 --arbitrator_entropy_reg 3.05e-7 --enable_binding_comm --share_utilities --enable_overflow --agents_sgd --arbitrator_sgd --enable_cuda



# Previous training configs

# ipython -- train.py --enable-cuda --disable-prosocial --disable-comms --model-file=models/model.pt

# both proposal and communications channels open, not prosocial
# --model-file=models/comms_prop_no_soc_reduced_ent.pt
# ipython -- train.py --enable-cuda --disable-prosocial --term-entropy-reg 0.05 --utterance-entropy-reg 0.0001 --proposal-entropy-reg 0.005

# communication channel open, proposal channel closed, not prosocial
# --model-file=models/comms_prop_no_soc_reduced_ent.pt
# ipython -- train.py --enable-cuda --disable-proposal --disable-prosocial --term-entropy-reg 0.05 --utterance-entropy-reg 0.0001 --proposal-entropy-reg 0.005

# proposal channel open, communication channel closed, not prosocial
# --model-file=models/comms_prop_no_soc_reduced_ent.pt
# ipython -- train.py --enable-cuda --disable-comms --disable-prosocial --term-entropy-reg 0.05 --utterance-entropy-reg 0.0001 --proposal-entropy-reg 0.005

# prosocial
# --model-file=models/comms_prop_no_soc_reduced_ent.pt
# ipython -- train.py --enable-cuda --term-entropy-reg 0.5 --utterance-entropy-reg 0.0001 --proposal-entropy-reg 0.01

# nothing is open
# ipython -- train.py --enable-cuda --disable-proposal --disable-prosocial --disable-comms --term-entropy-reg 0.05 --utterance-entropy-reg 0.0001 --proposal-entropy-reg 0.005


# only new proposal_repr channel open, orig reg coeffs
# ipython -- train.py --enable-cuda --disable-proposal --disable-prosocial --disable-comms --enable-proposal-repr --term-entropy-reg 0.05 --utterance-entropy-reg 0.0001 --proposal-entropy-reg 0.005

# only new proposal_repr channel open, utterance entropy reg = proposal entropy reg
# ipython -- train.py --enable-cuda --disable-proposal --disable-prosocial --disable-comms --enable-proposal-repr --term-entropy-reg 0.05 --mediator-entropy-reg 0.05 --utterance-entropy-reg 0.005 --proposal-entropy-reg 0.005

# also increase term entropy reg
# ipython -- train.py --enable-cuda --disable-proposal --disable-prosocial --disable-comms --enable-proposal-repr --term-entropy-reg 0.5 --utterance-entropy-reg 0.005 --proposal-entropy-reg 0.005

# all reg coeffs at max values
# ipython -- train.py --enable-cuda --disable-proposal --disable-prosocial --disable-comms --enable-proposal-repr --term-entropy-reg 0.5 --utterance-entropy-reg 0.01 --proposal-entropy-reg 0.01



# proposal and proposal_repr both open, utterance entropy reg = proposal entropy reg
# ipython -- train.py --enable-cuda --disable-prosocial --disable-comms --enable-proposal-repr --term-entropy-reg 0.05 --mediator-entropy-reg 0.05 --utterance-entropy-reg 0.005 --proposal-entropy-reg 0.005

# proposal open, proposal_repr closed
# ipython -- train.py --enable-cuda --disable-prosocial --disable-comms --term-entropy-reg 0.05 --mediator-entropy-reg 0.05 --utterance-entropy-reg 0.005 --proposal-entropy-reg 0.005

# both closed
# ipython -- train.py --enable-cuda --disable-prosocial --disable-proposal --disable-comms --term-entropy-reg 0.05 --mediator-entropy-reg 0.05 --utterance-entropy-reg 0.005 --proposal-entropy-reg 0.005



# only new proposal_repr channel open, mediation enabled
# ipython -- train.py --enable-cuda --disable-proposal --disable-prosocial --disable-comms --enable-proposal-repr --term-entropy-reg 0.05 --mediator-entropy-reg 0.05 --utterance-entropy-reg 0.005 --proposal-entropy-reg 0.005 --enable-mediator
# ipython -- train.py --disable-proposal --disable-prosocial --disable-comms --enable-proposal-repr --term-entropy-reg 0.05 --mediator-entropy-reg 0.05 --mediator-main-loss-coeff 100000 --utterance-entropy-reg 0.005 --proposal-entropy-reg 0.005 --enable-mediator

# ipython -- train.py --name=debug --enable_proposal_repr --enable_mediator --term_entropy_reg 0.05 --mediator_entropy_reg 0.05 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --mediator_main_loss_coeff 100000


# mediation disabled, just for debug
# ipython -- train.py --name=debug --enable_proposal_repr --term_entropy_reg 0.05 --mediator_entropy_reg 0.05 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005





# Trying out some mediation hyperparams

# ipython -- train.py --name=wo --enable_proposal_repr --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05
# ipython -- train.py --name=10000_0.05 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 10000 --mediator_entropy_reg 0.05 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05
# ipython -- train.py --name=1000_0.05 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 1000 --mediator_entropy_reg 0.05 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05
# ipython -- train.py --name=1_0.05 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 1 --mediator_entropy_reg 0.05 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05
# ipython -- train.py --name=100000_0.05 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 100000 --mediator_entropy_reg 0.05 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05

# On a grid

# ipython -- train.py --name=wo --enable_proposal_repr --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05
# ipython -- train.py --name=minus_8 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 0.00000001 --mediator_entropy_reg 0.05 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05
# ipython -- train.py --name=minus_5 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 0.00001 --mediator_entropy_reg 0.05 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05
# ipython -- train.py --name=minus_2 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 0.01 --mediator_entropy_reg 0.05 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05
# ipython -- train.py --name=plus_1 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 10 --mediator_entropy_reg 0.05 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05
# ipython -- train.py --name=plus_4 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 10000 --mediator_entropy_reg 0.05 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05
# ipython -- train.py --name=plus_7 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 10000000 --mediator_entropy_reg 0.05 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05

# ipython -- train.py --name=debug --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 0.05 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05
# ipython -- train.py --name=do_not_scale --enable_proposal_repr --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05


# ipython -- train.py --name=debug --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-9 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05
# ipython -- train.py --name=1e+4_scaling_both --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 1e+4 --mediator_entropy_reg 500 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05
# ipython -- train.py --name=1e+1_scaling_both --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 1e+1 --mediator_entropy_reg 0.05 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05
# ipython -- train.py --name=1e-2_scaling_both_larger_ent_coeffs --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 0.01 --mediator_entropy_reg 0.0005 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=1e+4_scaling_both_larger_ent_coeffs --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 1e+4 --mediator_entropy_reg 500 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=1e+1_scaling_both_larger_ent_coeffs --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 1e+1 --mediator_entropy_reg 0.05 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=disable_arb_ent_reg --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 0 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05
# ipython -- train.py --name=disable_arb_ent_reg_larger_ent_coeffs_redo_account --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 0 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05

# ipython -- train.py --name=random_arbitration --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 0 --mediator_entropy_reg 0.05 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05
# ipython -- train.py --name=random_arbitration_larger_ent_coeffs --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 0 --mediator_entropy_reg 0.05 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05


# ipython -- train.py --name=agents_sgd --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-9 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05 --agents_sgd
# ipython -- train.py --name=agents_med_sgd --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-9 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05 --agents_sgd --mediator_sgd
# ipython -- train.py --name=agents_sgd_larger_ent_coeffs --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-9 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --agents_sgd
# ipython -- train.py --name=agents_med_sgd_larger_ent_coeffs --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-9 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --agents_sgd --mediator_sgd
# ipython -- train.py --name=agents_sgd_larger_ent_coeffs_redo --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --agents_sgd
# ipython -- train.py --name=agents_med_sgd_larger_ent_coeffs_redo --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --agents_sgd --mediator_sgd


# ipython -- train.py --name=share_utilities --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-9 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05 --share_utilities
# ipython -- train.py --name=debug --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-9 --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05 --share_utilities
# ipython -- train.py --name=share_utilities_larger_ent_coeffs_redo_account --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-9 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --share_utilities
# ipython -- train.py --name=share_utilities_larger_ent_coeffs_redo --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --share_utilities

# ipython -- train.py --name=context_concat --enable_proposal_repr --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05 --agents_sgd --mediator_sgd

# ipython -- train.py --name=prosocial --enable_proposal_repr --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05 --prosocial
# ipython -- train.py --name=prosocial_larger_ent_coeffs --enable_proposal_repr --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --prosocial
# ipython -- train.py --name=prosocial_larger_ent_coeffs_redo_account --enable_proposal_repr --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --prosocial
# ipython -- train.py --name=prosocial_larger_ent_coeffs_redo_account_w_arb --enable_proposal_repr --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --prosocial --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-9

# ipython -- train.py --name=fc_context_net_larger_ent_coeffs --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-9 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=fc_context_net_2_layers_larger_ent_coeffs --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-9 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=fc_context_net_larger_ent_coeffs_redo_account --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-9 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05

# ipython -- train.py --name=wo_talk_wo_mediation --utterance_entropy_reg 0.00 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=wo_talk_w_mediation --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 0.00 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=binding --enable_proposal --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-9 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=binding_redo --enable_proposal --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=binding_wo_arb --enable_proposal --enable_proposal_repr --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=binding_wo_arb_redo --enable_proposal --enable_proposal_repr --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=binding_wo_arb_wo_cheap --enable_proposal --utterance_entropy_reg 0.00 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=binding_random_arbitration --enable_proposal --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 0 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05

# python -- train.py --name=redist_after_scaling_redo --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05


# ipython -- train.py --name=main_exp_ent_8 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-8 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=main_exp_ent_7 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=main_exp_ent_6 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-6 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=main_exp_ent_5 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-5 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=main_exp_ent_4 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-4 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=main_exp_ent_3 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-3 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=main_exp_ent_2 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-2 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=main_exp_ent_6e-8 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 6e-8 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=main_exp_ent_1.2e-7 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 1.2e-7 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=main_exp_ent_6e-7 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 6e-7 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=main_exp_ent_1.2e-6 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 1.2e-6 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=main_exp_wo_arb --enable_proposal_repr --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05



# ipython -- train.py --name=paper_coeffs_wo_arb --enable_proposal_repr --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05

# ipython -- train.py --name=paper_coeffs_random_arb --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 0 --mediator_entropy_reg 0.05 --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05

# ipython -- train.py --name=paper_coeffs_med_ent_9 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-9 --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05

# ipython -- train.py --name=paper_coeffs_med_ent_7 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05

# ipython -- train.py --name=paper_coeffs_med_ent_6 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-6 --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05

# ipython -- train.py --name=paper_coeffs_med_ent_0 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 0 --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05

# ipython -- train.py --name=impl_coeffs_wo_arb --enable_proposal_repr --utterance_entropy_reg 1e-4 --proposal_entropy_reg 5e-3 --term_entropy_reg 0.05

# ipython -- train.py --name=impl_coeffs_random_arb --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 0 --mediator_entropy_reg 0.05 --utterance_entropy_reg 1e-4 --proposal_entropy_reg 5e-3 --term_entropy_reg 0.05

# ipython -- train.py --name=impl_coeffs_med_ent_9 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-9 --utterance_entropy_reg 1e-4 --proposal_entropy_reg 5e-3 --term_entropy_reg 0.05

# ipython -- train.py --name=impl_coeffs_med_ent_7 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 1e-4 --proposal_entropy_reg 5e-3 --term_entropy_reg 0.05

# ipython -- train.py --name=impl_coeffs_med_ent_6 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-6 --utterance_entropy_reg 1e-4 --proposal_entropy_reg 5e-3 --term_entropy_reg 0.05

# ipython -- train.py --name=impl_coeffs_med_ent_0 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 0 --utterance_entropy_reg 1e-4 --proposal_entropy_reg 5e-3 --term_entropy_reg 0.05



# ipython -- train.py --name=debug --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-2 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 

# ipython -- train.py --name=overflow_baseline --enable_overflow --enable_proposal_repr --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=overflow_arbitration --enable_overflow --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=overflow_arbitration_share_utilities --enable_overflow --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --share_utilities
# ipython -- train.py --name=overflow_prosocial --enable_overflow --enable_proposal_repr --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --prosocial
# ipython -- train.py --name=overflow_binding --enable_overflow --enable_proposal_repr --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --enable_proposal
# ipython -- train.py --name=overflow_random_arbitration --enable_overflow --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 0 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05

# ipython -- train.py --name=destroy_not_redist_cheap --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=destroy_not_redist_binding --enable_proposal_repr --enable_mediator --enable_proposal --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05

# ipython -- train.py --name=partially_prosocial --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --scale_before_redist
# ipython -- train.py --name=partially_prosocial_wo_med --enable_proposal_repr  --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --scale_before_redist

# ipython -- train.py --name=cross_play_pair0 --enable_proposal_repr --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=cross_play_pair3_partially_prosocial_0.5 --enable_proposal_repr  --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --scale_before_redist
# ipython -- train.py --name=partially_prosocial_0.5_w_med --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --scale_before_redist

# ipython -- train.py --name=cross_play_pair1_fully_prosocial --enable_proposal_repr --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --prosocial

# ipython -- train.py --name=prosocial_binding --enable_proposal_repr --enable_proposal --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05 --prosocial
# ipython -- train.py --name=prosocial_binding_scale_before_redist --enable_proposal_repr --enable_proposal --utterance_entropy_reg 0.005 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05 --prosocial --scale_before_redist
# ipython -- train.py --name=prosocial_binding_higher_reg --enable_proposal_repr --enable_proposal --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --prosocial
# ipython -- train.py --name=prosocial_binding_scale_before_redist_higher_reg --enable_proposal_repr --enable_proposal --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --prosocial --scale_before_redist
# ipython -- train.py --name=prosocial_higher_utt_reg --enable_proposal_repr --utterance_entropy_reg 0.05 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --prosocial


# ipython -- train.py --name=cross_play_pair1_1e-3 --enable_proposal_repr --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05
# ipython -- train.py --name=cross_play_pair1_fully_prosocial_1e-3 --enable_proposal_repr --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --prosocial
# ipython -- train.py --name=cross_play_pair1_fully_prosocial_binding_1e-3 --enable_proposal_repr --enable_proposal --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.005 --term_entropy_reg 0.05 --prosocial

# turn the regime on
# ipython -- train.py --name=cross_play_pair1_partially_prosocial_0.5_1e-3 --enable_proposal_repr  --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --scale_before_redist
# ipython -- train.py --name=cross_play_pair1_partially_prosocial_0.5_w_med_1e-3 --enable_proposal_repr --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-7 --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --scale_before_redist


# different utility types

# ipython -- train.py --name=selfish_1_5_only_vs_1_5_only --enable_proposal_repr --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --utility_type='1_5_only,1_5_only'
# ipython -- train.py --name=selfish_3_4_5_only_vs_3_4_5_only --enable_proposal_repr --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --utility_type='3_4_5_only,3_4_5_only'
# ipython -- train.py --name=selfish_max_on_0_vs_max_on_0 --enable_proposal_repr --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --utility_type='max_on_0,max_on_0'
# ipython -- train.py --name=selfish_min_on_0_vs_min_on_0 --enable_proposal_repr --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --utility_type='min_on_0,min_on_0'
# ipython -- train.py --name=selfish_uniform_vs_1_5_only --enable_proposal_repr --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --utility_type='uniform,1_5_only'
# ipython -- train.py --name=selfish_max_on_0_vs_min_on_0 --enable_proposal_repr --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --utility_type='max_on_0,min_on_0'
# ipython -- train.py --name=selfish_1_5_only_vs_3_4_5_only --enable_proposal_repr --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --utility_type='1_5_only,3_4_5_only'

# ipython -- train.py --name=partially_prosocial_1_5_only_vs_1_5_only --enable_proposal_repr --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --utility_type='1_5_only,1_5_only' --prosociality_level=0.5 --scale_before_redist
# ipython -- train.py --name=partially_prosocial_3_4_5_only_vs_3_4_5_only --enable_proposal_repr --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --utility_type='3_4_5_only,3_4_5_only' --prosociality_level=0.5 --scale_before_redist
# ipython -- train.py --name=partially_prosocial_max_on_0_vs_max_on_0 --enable_proposal_repr --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --utility_type='max_on_0,max_on_0' --prosociality_level=0.5 --scale_before_redist
# ipython -- train.py --name=partially_prosocial_min_on_0_vs_min_on_0 --enable_proposal_repr --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --utility_type='min_on_0,min_on_0' --prosociality_level=0.5 --scale_before_redist
# ipython -- train.py --name=partially_prosocial_uniform_vs_1_5_only --enable_proposal_repr --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --utility_type='uniform,1_5_only' --prosociality_level=0.5 --scale_before_redist
# ipython -- train.py --name=partially_prosocial_max_on_0_vs_min_on_0 --enable_proposal_repr --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --utility_type='max_on_0,min_on_0' --prosociality_level=0.5 --scale_before_redist
# ipython -- train.py --name=partially_prosocial_1_5_only_vs_3_4_5_only --enable_proposal_repr --utterance_entropy_reg 1e-3 --proposal_entropy_reg 0.05 --term_entropy_reg 0.05 --utility_type='1_5_only,3_4_5_only' --prosociality_level=0.5 --scale_before_redist

# ipython -- train.py --name=debug --training_episodes=16 --utterance_entropy_reg=0.05 --proposal_entropy_reg=0.05 --term_entropy_reg=0.05 --scale_before_redist --prosociality_level=0.5 --enable_mediator --mediator_main_loss_coeff 6.1e-8 --mediator_entropy_reg 3.05e-7



