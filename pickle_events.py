#!/usr/bin/env python3

import pickle
from madmax.utils import lhe_event_loop
from argparse import ArgumentParser

filename = "events_signal_smeared_bres10.lhe"
ordering = ["l1", "v1", "b/1", "l2", "v2", "b/2", "b/3", "b/4"]

output_filename = filename.replace('lhe', 'pkl')
events = []

for n, (event, lhe_e) in enumerate(lhe_event_loop(filename, ordering)):
    events.append(event)

output_file = open(output_filename, 'wb')
pickle.dump(events, output_file)
output_file.close()
