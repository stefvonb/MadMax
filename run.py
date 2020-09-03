#!/usr/bin/env python3

print("INFO:\tImporting modules...")
from madmax import PhysicsError
from madmax.utils import lhe_event_loop, determine_process
from madmax.statistics import BreitWigner, Flat, Model, Gaussian
from madmax.objects import Event, Particle, FourMomentum
import madmax.change_of_variables as cov
import madmax.nlopt_interface as nl

import numpy as np
import lhapdf, math, time, statistics, sys, itertools, pickle
from copy import deepcopy

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument(dest="config_file", help="The config file that has all necessary information")
args = parser.parse_args()

from configparser import ConfigParser
config = ConfigParser()
config.read(args.config_file)

top_mass = 173.0
w_mass = 80.419
top_width = 1.5
w_width = 2.0476
h_mass = 125.0
h_width = 6.382e-3
b_resolution = 15.0/100.0

model = {
    "signal": Model(m_t1=[BreitWigner(top_mass, top_width), top_mass],
              m_t2=[BreitWigner(top_mass, top_width), top_mass],
              m_w1=[BreitWigner(w_mass, w_width), w_mass],
              m_w2=[BreitWigner(w_mass, w_width), w_mass],
              m_h=[BreitWigner(h_mass, 5.0), h_mass],
              b1_E_scale=[Gaussian(1.0, b_resolution), 1.0],
              b2_E_scale=[Gaussian(1.0, b_resolution), 1.0],
              b4_E_scale=[Gaussian(1.0, b_resolution), 1.0]),
    "background": Model(m_t1=[BreitWigner(top_mass, top_width), top_mass],
              m_t2=[BreitWigner(top_mass, top_width), top_mass],
              m_w1=[BreitWigner(w_mass, w_width), w_mass],
              m_w2=[BreitWigner(w_mass, w_width), w_mass],
              b1_E_scale=[Gaussian(1.0, b_resolution), 1.0],
              b2_E_scale=[Gaussian(1.0, b_resolution), 1.0],
              b3_E_scale=[Gaussian(1.0, b_resolution), 1.0],
              b4_E_scale=[Gaussian(1.0, b_resolution), 1.0])
}

parameter_ordering = {"signal": ["m_t1", "m_t2", "m_w1", "m_w2", "m_h", "b1_E_scale", "b2_E_scale", "b4_E_scale"],
                      "background": ["m_t1", "m_t2", "m_w1", "m_w2", "b1_E_scale", "b2_E_scale", "b3_E_scale", "b4_E_scale"]}
dimension = {k: len(parameter_ordering[k]) for k in ["signal", "background"]}

print("INFO:\tLoading event file...")
input_events =  pickle.load(open(config['run']['events_file'], "rb"))

print("INFO:\tInitialising MEs...")
import importlib.util
# Start with signal...
spec = importlib.util.spec_from_file_location("signal_me.allmatrix2py", "{}/allmatrix2py.so".format(config['matrixelements']['signal_me_path']))
signal_me = importlib.util.module_from_spec(spec)
spec.loader.exec_module(signal_me)
signal_me.initialise("{}/param_card.dat".format(config['matrixelements']['signal_me_path']))
signal_me_pdgs_list = signal_me.get_pdg_order()
me_initial_states = {"signal": set(), "background": set()}
me_final_states = {"signal": set(), "background": set()}
for p_l in signal_me_pdgs_list:
    me_initial_states["signal"].add(tuple(p_l[:2]))
    me_final_states["signal"].add(tuple(p_l[2:]))
print("INFO:\tSignal matrix element process: {}".format(determine_process(signal_me)))
# ... then the background
spec = importlib.util.spec_from_file_location("background_me.allmatrix2py", "{}/allmatrix2py.so".format(config['matrixelements']['background_me_path']))
background_me = importlib.util.module_from_spec(spec)
spec.loader.exec_module(background_me)
background_me.initialise("{}/param_card.dat".format(config['matrixelements']['background_me_path']))
background_me_pdgs_list = background_me.get_pdg_order()
for p_l in background_me_pdgs_list:
    me_initial_states["background"].add(tuple(p_l[:2]))
    me_final_states["background"].add(tuple(p_l[2:]))
print("INFO:\tBackground matrix element process: {}".format(determine_process(background_me)))

pdf_name = "NNPDF23_lo_as_0130_qed"
print("INFO:\tInitialising {} PDF...".format(pdf_name))
pdf = lhapdf.mkPDF(pdf_name, 0)
alpha_s = 0.130
Q = 91.1876

key_ordering = {
    "signal": ["l1", "v1", "b/1", "l2", "v2", "b/2", "b/3", "b/4"],
    "background": ["b/1", "l1", "v1", "b/2", "l2", "v2", "b/3", "b/4"]
}

transfer_functions = {"b/1": Gaussian(1.0, b_resolution),
                      "b/2": Gaussian(1.0, b_resolution),
                      "b/3": Gaussian(1.0, b_resolution),
                      "b/4": Gaussian(1.0, b_resolution)}

def matrix_element(event, process, pdgs_list, skip_TF=False):
    p_matrix = event.momentum_matrix_transpose(key_ordering[process])
    if not skip_TF:
        for k, tf in transfer_functions.items():
            tf.mean = event.checkpoint_particles[k].E
            tf.sigma = event.checkpoint_particles[k].E*b_resolution
        tf_values = [tf.f(event.particles[k].E) for k, tf in transfer_functions.items()]

    me_values = []
    for pdgs in pdgs_list:
        f1_x1 = pdf.xfxQ(pdgs[0], event.x1, Q)
        f2_x2 = pdf.xfxQ(pdgs[1], event.x2, Q)
        f1_x2 = pdf.xfxQ(pdgs[0], event.x2, Q)
        f2_x1 = pdf.xfxQ(pdgs[1], event.x1, Q)
        if process == "signal":
            me_value = signal_me.smatrixhel(pdgs, p_matrix, alpha_s, 0.0, -1)*(f1_x1*f2_x2 + f1_x2*f2_x1)
        elif process == "background":
            me_value = background_me.smatrixhel(pdgs, p_matrix, alpha_s, 0.0, -1)*(f1_x1*f2_x2 + f1_x2*f2_x1)
        if not skip_TF:
            for tf_value in tf_values:
                me_value *= tf_value
        me_values.append(me_value)
    return sum(me_values)


def objective_function(x, grad, events, process, pdgs_list):
    largest_me = 0.0
    p = dict(zip(parameter_ordering[process], np.nan_to_num(model[process].cube_to_standard(x, parameter_ordering[process]))))
    # Transfer function for b/1/2/4
    for i in range(len(events)):
        events[i].reset_particles(["b/1", "b/2", "b/3", "b/4"])
        try:
            events[i]["b/1"].scale_energy(p["b1_E_scale"])
            events[i]["b/2"].scale_energy(p["b2_E_scale"])
            if process == "background":
                events[i]["b/3"].scale_energy(p["b3_E_scale"])
            events[i]["b/4"].scale_energy(p["b4_E_scale"])
        except PhysicsError:
            return 0.0
    if process == "signal":
        block_D_solutions = cov.block_D(events, ["b/3", "b/4"], p["m_h"], m1=4.7)
        solutions = cov.class_D(block_D_solutions, ["v1", "v2", "l1", "b/1", "l2", "b/2"], p["m_t1"], p["m_t2"], p["m_w1"], p["m_w2"])
    elif process == "background":
        solutions = cov.class_D(events, ["v1", "v2", "l1", "b/1", "l2", "b/2"], p["m_t1"], p["m_t2"], p["m_w1"], p["m_w2"])
    if len(solutions) > 0:
        for sol in solutions:
            me_value = matrix_element(sol, process, pdgs_list)
            if me_value > largest_me:
                largest_me = me_value
    
    return largest_me

def process_event(input_event):
    lhe_fs_pids = [input_event[k].pid for k in input_event.particles]
    try:
        event_final_state = {k: [list(fs) for fs in me_final_states[k] if sorted(lhe_fs_pids) == sorted(fs)][0] for k in ["signal", "background"]}
        pdgs_list = {k: [list(i) + v for i in me_initial_states[k]] for k, v in event_final_state.items()}
    except IndexError:
        print("ERROR:\tProblem parsing PID lists for ME")
        sys.exit()
    event = deepcopy(input_event)
    event.save_checkpoint()

    optimiser = {"signal": nl.optimiser(algorithm_name, dimension["signal"], ftol_rel=0.01, maxtime=200.0),
                 "background": nl.optimiser(algorithm_name, dimension["background"], ftol_rel=0.01, maxtime=200.0)}

    times = {}
    times["start"] = time.perf_counter()
    xopt, sopt = {}, {}
    for p in ["signal", "background"]:
        if p == "signal":
            permuted_events = event.make_permutations([("b/3", "b/4")])
        else:
            permuted_events = event.make_permutations()
        optimiser[p].set_max_objective(lambda x, grad: objective_function(x, grad, permuted_events, p, pdgs_list[p]))
        optimiser[p].run(model[p].cube_starting_values(parameter_ordering[p]))
        xopt[p] = optimiser[p].xopt
        sopt[p] = model[p].cube_to_standard(optimiser[p].xopt, parameter_ordering[p])
        times[p] = time.perf_counter()
        
    total_time = time.perf_counter() - times["start"]
    
    return round(total_time, 3), nl.NLOPT_EXIT_CODES[optimiser["signal"].last_optimize_result()], round(times["signal"] - times["start"], 3), optimiser["signal"].last_optimum_value(), nl.NLOPT_EXIT_CODES[optimiser["background"].last_optimize_result()], round(times["background"] - times["signal"], 3), optimiser["background"].last_optimum_value()

if __name__ == "__main__":
    num_events = int(config['run']['num_events'])
    algorithm_name = config['run']['algorithm']

    print("INFO:\tMaximising {} events with algorithm: {}...".format(num_events, algorithm_name))

    results_file = open(config['run']['output_file'], 'w')
    results_file.write("total_time,signal_termination_reason,signal_time,signal_maximised_weight,background_termination_reason,background_time,background_maximised_weight\n")
    results_file.close()
    print("INFO:\tWriting results to {}".format(results_file.name))

    with open(config['run']['output_file'], 'a', buffering=1) as f:
        for event in input_events[:num_events]:
            try:
                result = process_event(event)
                f.write(",".join([str(r_i) for r_i in result]))
                f.write("\n")
            except KeyboardInterrupt:
                print("\nINFO:\tProgram interrupted. Exiting gracefully...")
                break

    print("INFO:\tFinished")
