from fenci import Segment


import logging

logging.basicConfig(level=logging.INFO)

seg = Segment()

seg.reset_model(model='simple')

seg.training('../icwb2-data/training', 'msr_training.utf8', with_hmm=True)

seg.save_model(save_hmm=True)


