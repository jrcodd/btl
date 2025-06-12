import tensorflow as tf

path_to_events_file = "/media/btl/Jackson/lap_gym/sofa_zoo/sofa_zoo/envs/tissue_dissection/runs/PPO_STATE_2rows_Falsevis_1/events.out.tfevents.1749572672.cnc-rp.1490917.0"

for e in tf.train.summary_iterator(path_to_events_file):
    for v in e.summary.value:
        print(f"Tag: {v.tag}, Step: {e.step}, Value: {v.simple_value}")