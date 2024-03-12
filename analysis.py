import os
import torch
import argparse
import numpy as np
from matplotlib import pyplot as plt

from utils.data_utils import set_seed, str2bool, assign_colors
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork
from utils.functions import load_problem
from utils import load_model
from nets.gpn import GPN
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def arguments(args=None):
    parser = argparse.ArgumentParser(description="Visualize predictions made by some algorithms")
    parser.add_argument('--seed', type=int, default=0, help='Random seed to use')

    # Method
    parser.add_argument('--model', help='Path to load model. Just indicate the directory where epochs are saved or'
                                        'the directory + the specific epoch you want to load. For baselines, indicate'
                                        'the name of the baselines instead (opga, pso, aco)-')

    # Problem
    parser.add_argument('--problem', default='top', help="The problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--data_distribution', type=str, default='const',
                        help='Data distribution to use during training, defaults and options depend on problem')
    parser.add_argument('--num_agents', type=int, default=4, help="Number of agents")
    parser.add_argument('--num_depots', type=int, default=1, help="Number of depots. Options are 1 or 2. num_depots=1"
                        "means that the start and end depot are the same. num_depots=2 means that they are different")
    parser.add_argument('--return2depot', type=str2bool, default=True, help="True for constraint of returning to depot")
    parser.add_argument('--max_length', type=float, default=2, help="Normalized time limit to solve the problem")

    # CPU / GPU
    parser.add_argument('--use_cuda', type=str2bool, default=True, help="True to use CUDA")
    opts = parser.parse_args(args)
    opts.use_cuda = torch.cuda.is_available() and opts.use_cuda

    # Check problem is correct
    assert opts.problem == 'top', "Only the top problem is supported"
    assert opts.num_agents > 0, 'num_agents must be greater than 0'

    # Check baseline is correct for the given problem
    assert opts.model in ('opga', 'aco', 'pso') or os.path.exists(opts.model), \
        'Path to model does not exist. For baselines, the supported baselines for TOP are opga, aco, pso'
    return opts

def reshape_tours(tours, num_agents, end_ids=0):
    new_tours = [[] for _ in range(num_agents)]
    count, check = 0, True
    for node in tours.reshape(-1, order='F'):
        if count >= num_agents:
            break
        if node == end_ids:
            if check:
                count += 1
            check = False
        else:
            new_tours[count].append(node)
            check = True
    return new_tours


def add_depots(tours, num_agents, graph_size, node_energy):
    tours = list(tours)
    for k in range(num_agents):
        tours[k] = np.array(tours[k])
        if len(tours[k]) > 0:
            if tours[k][0] != 0:
                tours[k] = np.concatenate(([0], tours[k]), axis=0)
            if tours[k][-1] != graph_size + 1:
                tours[k] = np.concatenate((tours[k], [graph_size + 1]), axis=0)
        else:
            tours[k] = np.array([0, graph_size + 1])

        energies = [node_energy[node] for node in tours[k] if node != 0 and node != graph_size + 1]

        print('Agent {}: '.format(k + 1), tours[k])
        print(energies)

    return tours


def main(opts):

    # Set seed for reproducibility
    set_seed(opts.seed)

    # Load problem
    problem = load_problem(opts.problem)
    dataset = problem.make_dataset(size=opts.graph_size, num_samples=1, distribution=opts.data_distribution,
                                   max_length=opts.max_length, num_agents=opts.num_agents, num_depots=opts.num_depots)
    
    specific_epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 99]
    
    # Loop over each epoch
    for epoch in specific_epochs:
        epoch_file = f"epoch-{epoch}.pt"
        epoch_path = os.path.join(opts.model, epoch_file)
    
        # Set the device
        device = torch.device("cuda:0" if opts.use_cuda else "cpu")
        
        if os.path.exists(epoch_path):
            # Load model (Transformer, PN, GPN) for evaluation on the chosen device
            model, _ = load_model(epoch_path, num_agents=opts.num_agents)
            model.set_decode_type('greedy')
            model.num_depots = opts.num_depots
            model.num_agents = opts.num_agents
            model.eval()  # Put in evaluation mode to not track gradients
            model.to(device)

            # Calculate tour
            inputs = dataset.data[0]
            for k, v in inputs.items():
                inputs[k] = torch.tensor(v).unsqueeze(0).to(device)
            _, _, tours = model(inputs, return_pi=True)

            # Torch tensors to numpy
            tours = tours.cpu().detach().numpy().squeeze()
            for k, v in inputs.items():
                inputs[k] = v.cpu().detach().numpy().squeeze()

            # Reshape tours list
            tours = reshape_tours(tours, opts.num_agents, end_ids=inputs['loc'].shape[0] + 1)

            # Add depots and print tours and analyse parameters
            tours = add_depots(tours, opts.num_agents, opts.graph_size, inputs['energy'])
        else:
            print(f"Epoch file '{epoch_file}' does not exist.")

if __name__ == "__main__":
    main(arguments())