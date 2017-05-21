import tensorflow as tf
import numpy as np
import threading
import time

def MyLoop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print(f'Stoping from id: {worker_id}\n')
            coord.request_stop()
        else:
            print(f'Working on id: {worker_id}\n')

        time.sleep(1)


coord = tf.train.Coordinator()

threads = [
    threading.Thread(target=MyLoop, args=(coord, i,)) for i in range(5)
]

for t in threads:
    t.start()

coord.join(threads)