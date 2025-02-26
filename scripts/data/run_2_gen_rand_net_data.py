from sfeno.datasets.synthesized import dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--network_size", type=int, dest="network_size", action="store", default=100)
pargs = parser.parse_args()

if __name__=='__main__':
    path = 'sfeno/datasets/synthesized'

    dataset.create_and_save_data(path, pargs.network_size)