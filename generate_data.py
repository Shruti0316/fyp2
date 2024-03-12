import os
import argparse
import numpy as np
from tqdm import tqdm

from utils.data_utils import save_dataset
from utils.data_utils import set_seed


def generate_top_data(dataset_size, op_size, prize_type='const', max_length=2, num_depots=1):

    # Coordinates
    depot = np.random.uniform(size=(dataset_size, 3))
    loc = np.random.uniform(size=(dataset_size, op_size, 3))
    energy = np.random.uniform(size=(dataset_size, op_size))
    delay = np.random.uniform(size=(dataset_size, op_size))
    network_lifetime = np.random.uniform(size=(dataset_size, op_size))
    pdr = np.random.uniform(size=(dataset_size, op_size))
    throughput = np.random.uniform(size=(dataset_size, op_size))
    connectivity = np.random.uniform(size=(dataset_size, op_size))
    routing_overhead = np.random.uniform(size=(dataset_size, op_size))

    # Methods taken from Fischetti et al. 1998
    if prize_type == 'const':
        prize = np.ones((dataset_size, op_size))
    elif prize_type == 'unif':
        prize = (1 + np.random.randint(0, 100, size=(dataset_size, op_size))) / 100.
    else:  # Based on distance to depot
        assert prize_type == 'dist'
        prize_ = np.linalg.norm(depot[:, None, :] - loc, axis=-1)
        prize = (1 + (prize_ / prize_.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.

    # Max length should be approximately half of optimal TSP tour, such that half (a bit more) of the nodes can be
    # visited (which is maximally difficult as this has the largest number of possibilities). Recommendations are:
    # 2 for n=20 | 3 for n=50 | 4 for n=100
    max_length = np.full(dataset_size, max_length)  # Capacity, same for whole dataset

    # Output
    output = [depot, loc, prize, max_length, energy, delay, network_lifetime, pdr, throughput, connectivity, routing_overhead]

    # End depot different from start depot
    if num_depots == 2:
        depot2 = np.random.uniform(size=(dataset_size, 3))
        output.append(depot2)

    return list(zip(
        *[item.tolist() for item in output]
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Filename
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset (test, validation...)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")

    # Data
    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem: const, dist, unif or all.")
    parser.add_argument('--max_length', type=float, nargs='+', default=[2],
                        help="Normalized time limit to solve the problem")
    parser.add_argument('--num_depots', type=int, default=1, help="Number of depots. Options are 1 or 2. num_depots=1"
                        "means that the start and end depot are the same. num_depots=2 means that they are different")

    # Misc
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")
    opts = parser.parse_args()
    assert len(opts.max_length) in [1, len(opts.graph_sizes)], \
        "You must indicate one single max_length or one max_length for each graph size"

    # Set seed
    set_seed(opts.seed)

    # For each distribution
    distributions = ['const', 'unif', 'dist'] if opts.data_distribution == 'all' else [opts.data_distribution]
    for distribution in distributions:
        print(f'Reward distribution: {distribution}')

        # For each graph_size
        for i, graph_size in enumerate(tqdm(opts.graph_sizes)):

            # Max length
            max_length = opts.max_length[0] if len(opts.max_length) == 1 else opts.max_length[i]

            # Directory
            data_dir = os.path.join(opts.data_dir, str(opts.num_depots) + 'depots', distribution, str(graph_size))
            os.makedirs(data_dir, exist_ok=True)

            # Filename
            filename = os.path.join(data_dir, "{}_seed{}_L{}.pkl".format(opts.name, opts.seed, max_length))

            # Generate data
            dataset = generate_top_data(opts.dataset_size, graph_size, prize_type=distribution,
                                        max_length=max_length, num_depots=opts.num_depots)

            # Save dataset
            save_dataset(dataset, filename)
    print('Finished')
