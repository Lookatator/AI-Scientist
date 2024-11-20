import os
import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(curr_dir, 'src')
sys.path.insert(0, os.path.join(src_dir, 'PyTorch-NEAT'))
sys.path.insert(1, curr_dir)

import random
import shutil
import numpy as np
import torch
import argparse
from dataclasses import dataclass

import neat
from neat.math_util import mean
from neat.reporting import ReporterSet
from neat.population import CompleteExtinctionException
from neat.graphs import required_for_output
from pytorch_neat.cppn import create_cppn
from pytorch_neat.activations import str_to_activation
from pytorch_neat.aggregations import str_to_aggregation

from src.ppo.run import run_ppo
from parallel_evaluator import ParallelEvaluator

from evogym import is_connected, has_actuator, get_full_connectivity, hashable
import evogym.envs  # To register the environments, otherwise they are not available


@dataclass
class PPOConfig:
    verbose_ppo: int = 1
    learning_rate: float = 1e-3 
    n_steps: int = 128
    batch_size: int = 64 
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    ent_coef: float = 0.01
    clip_range: float = 0.1
    total_timesteps: int = 256000 
    log_interval: int = 100 
    n_envs: int = 4 
    n_eval_envs: int = 1 
    n_evals: int = 1 
    eval_interval: int = 1e5 

@dataclass
class ExperimentConfig:
    save_path: str
    exp_name: str = 'test_cppn'
    env_name: str = 'Thrower-v0' 
    pop_size: int = 12 
    structure_shape: tuple = (5, 5)
    max_evaluations: int = 6
    num_cores: int = 12 
    ppo: PPOConfig = PPOConfig()


class Node:
    def __init__(
        self,
        children,
        weights,
        response,
        bias,
        activation,
        aggregation,
        name=None,
        leaves=None,
    ):
        """
        children: list of Nodes
        weights: list of floats
        response: float
        bias: float
        activation: torch function from .activations
        aggregation: torch function from .aggregations
        name: str
        leaves: dict of Leaves
        """
        self.children = children
        self.leaves = leaves
        self.weights = weights
        self.response = response
        self.bias = bias
        self.activation = activation
        self.activation_name = activation
        self.aggregation = aggregation
        self.aggregation_name = aggregation
        self.name = name
        if leaves is not None:
            assert isinstance(leaves, dict)
        self.leaves = leaves
        self.activs = None
        self.is_reset = None

    def __repr__(self):
        header = "Node({}, response={}, bias={}, activation={}, aggregation={})".format(
            self.name,
            self.response,
            self.bias,
            self.activation_name,
            self.aggregation_name,
        )
        child_reprs = []
        for w, child in zip(self.weights, self.children):
            child_reprs.append(
                "    <- {} * ".format(w) + repr(child).replace("\n", "\n    ")
            )
        return header + "\n" + "\n".join(child_reprs)

    def activate(self, xs, shape):
        """
        xs: list of torch tensors
        """
        if not xs:
            return torch.full(shape, self.bias)
        inputs = [w * x for w, x in zip(self.weights, xs)]
        try:
            pre_activs = self.aggregation(inputs)
            activs = self.activation(self.response * pre_activs + self.bias)
            assert activs.shape == shape, "Wrong shape for node {}".format(self.name)
        except Exception:
            raise Exception("Failed to activate node {}".format(self.name))
        return activs

    def get_activs(self, shape):
        if self.activs is None:
            xs = [child.get_activs(shape) for child in self.children]
            self.activs = self.activate(xs, shape)
        return self.activs

    def __call__(self, **inputs):
        assert self.leaves is not None
        assert inputs
        shape = list(inputs.values())[0].shape
        self.reset()
        for name in self.leaves.keys():
            assert (
                inputs[name].shape == shape
            ), "Wrong activs shape for leaf {}, {} != {}".format(
                name, inputs[name].shape, shape
            )
            self.leaves[name].set_activs(inputs[name])
        return self.get_activs(shape)

    def _prereset(self):
        if self.is_reset is None:
            self.is_reset = False
            for child in self.children:
                child._prereset()  # pylint: disable=protected-access

    def _postreset(self):
        if self.is_reset is not None:
            self.is_reset = None
            for child in self.children:
                child._postreset()  # pylint: disable=protected-access

    def _reset(self):
        if not self.is_reset:
            self.is_reset = True
            self.activs = None
            for child in self.children:
                child._reset()  # pylint: disable=protected-access

    def reset(self):
        self._prereset()  # pylint: disable=protected-access
        self._reset()  # pylint: disable=protected-access
        self._postreset()  # pylint: disable=protected-access


class Leaf:
    def __init__(self, name=None):
        self.activs = None
        self.name = name

    def __repr__(self):
        return "Leaf({})".format(self.name)

    def set_activs(self, activs):
        self.activs = activs

    def get_activs(self, shape):
        assert self.activs is not None, "Missing activs for leaf {}".format(self.name)
        assert (
            self.activs.shape == shape
        ), "Wrong activs shape for leaf {}, {} != {}".format(
            self.name, self.activs.shape, shape
        )
        return self.activs

    def _prereset(self):
        pass

    def _postreset(self):
        pass

    def _reset(self):
        self.activs = None

    def reset(self):
        self._reset()


def create_cppn(genome, config, leaf_names, node_names, output_activation=None):

    genome_config = config.genome_config
    required = required_for_output(
        genome_config.input_keys, genome_config.output_keys, genome.connections
    )

    # Gather inputs and expressed connections.
    node_inputs = {i: [] for i in genome_config.output_keys}
    for cg in genome.connections.values():
        if not cg.enabled:
            continue

        i, o = cg.key
        if o not in required and i not in required:
            continue

        if i in genome_config.output_keys:
            continue

        if o not in node_inputs:
            node_inputs[o] = [(i, cg.weight)]
        else:
            node_inputs[o].append((i, cg.weight))

        if i not in node_inputs:
            node_inputs[i] = []

    nodes = {i: Leaf() for i in genome_config.input_keys}

    assert len(leaf_names) == len(genome_config.input_keys)
    leaves = {name: nodes[i] for name, i in zip(leaf_names, genome_config.input_keys)}

    def build_node(idx):
        if idx in nodes:
            return nodes[idx]
        node = genome.nodes[idx]
        conns = node_inputs[idx]
        children = [build_node(i) for i, w in conns]
        weights = [w for i, w in conns]
        if idx in genome_config.output_keys and output_activation is not None:
            activation = output_activation
        else:
            activation = str_to_activation[node.activation]
        aggregation = str_to_aggregation[node.aggregation]
        nodes[idx] = Node(
            children,
            weights,
            node.response,
            node.bias,
            activation,
            aggregation,
            leaves=leaves,
        )
        return nodes[idx]

    for idx in genome_config.output_keys:
        build_node(idx)

    outputs = [nodes[i] for i in genome_config.output_keys]

    for name in leaf_names:
        leaves[name].name = name

    for i, name in zip(genome_config.output_keys, node_names):
        nodes[i].name = name

    return outputs


class Population(neat.Population):
    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None

    def run(self, fitness_function, constraint_function=None, n=None):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided constraint function.
            # If some genomes violate the constraint, generate new genomes and replace them, until all genomes satisfy the constraint.
            if constraint_function is not None:
                genomes = list(self.population.items())
                validity = constraint_function(genomes, self.config, self.generation)
                valid_idx = np.where(validity)[0]
                valid_genomes = np.array(genomes)[valid_idx]
                while len(valid_genomes) < self.config.pop_size:
                    new_population = self.reproduction.create_new(self.config.genome_type,
                                                                    self.config.genome_config,
                                                                    self.config.pop_size - len(valid_genomes))
                    new_genomes = list(new_population.items())
                    validity = constraint_function(new_genomes, self.config, self.generation)
                    valid_idx = np.where(validity)[0]
                    valid_genomes = np.vstack([valid_genomes, np.array(new_genomes)[valid_idx]])

                self.population = dict(valid_genomes)
                self.species.speciate(self.config, self.population, self.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(self.population.items()), self.config, self.generation)

            # Gather and report statistics.
            best = None
            for g in self.population.values():
                if g.fitness is None:
                    raise RuntimeError("Fitness not assigned to genome {}".format(g.key))

                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in self.population.values())
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome


def get_cppn_input(structure_shape):
    x, y = torch.meshgrid(torch.arange(structure_shape[0]), torch.arange(structure_shape[1]))
    x, y = x.flatten(), y.flatten()
    center = (np.array(structure_shape) - 1) / 2
    d = ((x - center[0]) ** 2 + (y - center[1]) ** 2).sqrt()
    return x, y, d

def get_robot_from_genome(genome, config):
    nodes = create_cppn(genome, config, leaf_names=['x', 'y', 'd'], node_names=['empty', 'rigid', 'soft', 'hori', 'vert'])
    structure_shape = config.extra_info['structure_shape']
    x, y, d = get_cppn_input(structure_shape)
    material = []
    for node in nodes:
        material.append(node(x=x, y=y, d=d).numpy())
    material = np.vstack(material).argmax(axis=0)
    robot = material.reshape(structure_shape)
    return robot

def eval_genome_fitness(genome, config, genome_id, generation):
    robot = get_robot_from_genome(genome, config)
    args, env_name = config.extra_info['args'], config.extra_info['env_name']
    
    connectivity = get_full_connectivity(robot)
    save_path_generation = os.path.join(config.extra_info['save_path'], f'generation_{generation}')
    save_path_structure = os.path.join(save_path_generation, 'structure', f'{genome_id}')
    save_path_controller = os.path.join(save_path_generation, 'controller')
    np.savez(save_path_structure, robot, connectivity)

    fitness = run_ppo(
        args, robot, env_name, save_path_controller, f'{genome_id}', connectivity
    )
    return fitness

def eval_genome_constraint(genome, config, genome_id, generation):
    robot = get_robot_from_genome(genome, config)
    validity = is_connected(robot) and has_actuator(robot)
    if validity:
        robot_hash = hashable(robot)
        if robot_hash in config.extra_info['structure_hashes']:
            validity = False
        else:
            config.extra_info['structure_hashes'][robot_hash] = True
    return validity


class SaveResultReporter(neat.BaseReporter):

    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.generation = None

    def start_generation(self, generation):
        self.generation = generation
        save_path_structure = os.path.join(self.save_path, f'generation_{generation}', 'structure')
        save_path_controller = os.path.join(self.save_path, f'generation_{generation}', 'controller')
        os.makedirs(save_path_structure, exist_ok=True)
        os.makedirs(save_path_controller, exist_ok=True)

    def post_evaluate(self, config, population, species, best_genome):
        save_path_ranking = os.path.join(self.save_path, f'generation_{self.generation}', 'output.txt')
        genome_id_list, genome_list = np.arange(len(population)), np.array(list(population.values()))
        sorted_idx = sorted(genome_id_list, key=lambda i: genome_list[i].fitness, reverse=True)
        genome_id_list, genome_list = list(genome_id_list[sorted_idx]), list(genome_list[sorted_idx])
        with open(save_path_ranking, 'w') as f:
            out = ''
            for genome_id, genome in zip(genome_id_list, genome_list):
                out += f'{genome_id}\t\t{genome.fitness}\n'
            f.write(out)

def run_cppn_neat(
    config: ExperimentConfig
):
    exp_name, env_name, pop_size, structure_shape, max_evaluations, num_cores = (
        config.exp_name,
        config.env_name,
        config.pop_size,
        config.structure_shape,
        config.max_evaluations,
        config.num_cores,
    )

    save_path = os.path.join(config.save_path, exp_name)

    try:
        os.makedirs(save_path)
    except FileExistsError:
        print(f'THIS EXPERIMENT ({exp_name}) ALREADY EXISTS')
        print('Override? (y/n): ', end='')
        ans = input()
        if ans.lower() == 'y':
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        else:
            return None, None
        print()

    save_path_metadata = os.path.join(save_path, 'metadata.txt')
    with open(save_path_metadata, 'w') as f:
        f.write(f'POP_SIZE: {pop_size}\n' \
            f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n' \
            f'MAX_EVALUATIONS: {max_evaluations}\n')

    structure_hashes = {}

    config_path = os.path.join(curr_dir, 'neat.cfg')
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
        extra_info={
            'structure_shape': structure_shape,
            'save_path': save_path,
            'structure_hashes': structure_hashes,
            'args': config.ppo, # args for run_ppo
            'env_name': env_name,
        },
        custom_config=[
            ('NEAT', 'pop_size', pop_size),
        ],
    )

    pop = Population(config)
    reporters = [
        neat.StatisticsReporter(),
        neat.StdOutReporter(True),
        SaveResultReporter(save_path),
    ]
    for reporter in reporters:
        pop.add_reporter(reporter)

    evaluator = ParallelEvaluator(num_cores, eval_genome_fitness, eval_genome_constraint)

    pop.run(
        evaluator.evaluate_fitness,
        evaluator.evaluate_constraint,
        n=np.ceil(max_evaluations / pop_size))

    best_robot = get_robot_from_genome(pop.best_genome, config)
    best_fitness = pop.best_genome.fitness
    return best_robot, best_fitness


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CPPN-NEAT with save path specification')
    parser.add_argument('--path', type=str, required=True, help='Path to save experiment data')
    args = parser.parse_args()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    config = ExperimentConfig(save_path=args.path)
    best_robot, best_fitness = run_cppn_neat(config)
    
    print('Best robot:')
    print(best_robot)
    print('Best fitness:', best_fitness)
