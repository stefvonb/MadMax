from . import FP_TOLERANCE
import numpy as np
from madmax.objects import Event, Particle, FourMomentum

PID_NAMES = {
    1: "u", -1: "u~", 2: "d", -2: "d~", 3: "s", -3: "s~", 4: "c", -4: "c~", 5: "b", -5: "b~", 6: "t", -6: "t~",
    11: "e-", -11: "e+", 12: "ve", -12: "ve~",
    13: "mu-", -13: "mu+", 14: "vmu", -14: "vmu~",
    15: "tau-", -15: "tau+", 16: "vtau", -16: "vtau~",
    21: "g", 22: "y", 23: "Z", 24: "W+", -24: "W-", 25: "h", 32: "Z'", 33: "Z''", 34: "W'+", -34: "W'-", 35: "H", 36: "A", 37: "H+", -37: "H-",
}

def remove_complex(input_array):
    output_array = []
    for element in input_array:
        if abs(element.imag) < FP_TOLERANCE:
            output_array.append(float(element.real))
    return np.array(output_array, dtype=np.float)


def lhe_event_loop(lhe_filename, ordering, sqrt_s=13000):
    with open(lhe_filename, 'r') as lhe_file:
        reading_event = False
        for line in lhe_file:
            if reading_event:
                lhe_event += line
                ls = line.split()
                if len(ls) == 13:
                    if int(ls[1]) == -1:
                        z_mom = float(ls[8])
                        if z_mom > 0.0:
                            event.x1 = z_mom/(event.sqrt_s/2)
                        else:
                            event.x2 = -z_mom/(event.sqrt_s/2)
                    if int(ls[1]) == 1:
                        if abs(int(ls[0])) == 12 or abs(int(ls[0])) == 14:
                            status = 1
                        else:
                            status = 2
                        event.add_particle(ordering[index], Particle(int(ls[0]), status, FourMomentum(float(ls[9]), float(ls[6]), float(ls[7]), float(ls[8]))))
                        index += 1
            if "<event>" in line:
                reading_event = True
                event = Event(sqrt_s)
                lhe_event = ""
                index = 0
            elif "</event>" in line:
                reading_event = False
                lhe_event = lhe_event.replace("\n</event>", "")
                yield event, lhe_event


def particle_grouping(particle_set):
    particle_set = list(particle_set)
    if len(particle_set) == 1:
        return PID_NAMES[particle_set[0]]
    elif len(particle_set) > 2:
        if 21 in particle_set:
            return "p"
    else:
        return "({})".format(",".join([PID_NAMES[pid] for pid in particle_set]))

def determine_process(matrix_element):
    try:
        pdg_order = matrix_element.get_pdg_order()
    except AttributeError:
        print("ERROR:\tCould not determine PDGs of matrix element. Make sure you are passing the appropriate argument.")
        return None

    num_particles = len(pdg_order[0])
    particle_sets = [set() for _ in range(num_particles)]

    for channel in pdg_order:
        for i in range(num_particles):
            particle_sets[i].add(channel[i])

    process_string = "{} {} -> {}".format(particle_grouping(particle_sets[0]), particle_grouping(particle_sets[1]),
                                          " ".join([particle_grouping(ps) for ps in particle_sets[2:]]))
    return process_string
