import copy
from itertools import count
import math
import os
import sys
import json
from datetime import datetime

curr_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(curr_dir, "src")
sys.path.insert(0, os.path.join(src_dir, "PyTorch-NEAT"))
sys.path.insert(1, curr_dir)

import random
import shutil
import numpy as np
import torch
import torch.functional as F
import argparse
from dataclasses import dataclass
from functools import reduce
from operator import mul

import neat
from neat.config import ConfigParameter, write_pretty_params, DefaultClassConfig
from neat.genes import DefaultNodeGene, DefaultConnectionGene
from neat.math_util import mean, stdev, stat_functions
from neat.reporting import ReporterSet
from neat.population import CompleteExtinctionException
from neat.graphs import required_for_output, creates_cycle
from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet

from src.ppo.run import run_ppo
from parallel_evaluator import ParallelEvaluator

from evogym import is_connected, has_actuator, get_full_connectivity, hashable
import evogym.envs  # To register the environments, otherwise they are not available

################################################################################
# Configuration Dataclasses
################################################################################

@dataclass
class PPOConfig:
    verbose_ppo: int = 0
    learning_rate: float = 2.5e-4
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    ent_coef: float = 0.01
    clip_range: float = 0.1
    total_timesteps: int = 500_000
    log_interval: int = 100
    n_envs: int = 4
    n_eval_envs: int = 1
    n_evals: int = 4
    eval_interval: int = 5e4


@dataclass
class ExperimentConfig:
    save_path: str
    exp_name: str = "test_cppn"
    env_name: str = "Carrier-v0"  # TODO: change if needed
    pop_size: int = 12
    structure_shape: tuple = (5, 5)
    max_evaluations: int = 59
    num_cores: int = 12
    ppo: PPOConfig = PPOConfig()


################################################################################
# NEAT-python code
# from https://github.com/yunshengtian/neat-python (commit 2762ab6)
################################################################################


class Species(object):
    def __init__(self, key, generation):
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.representative = None
        self.members = {}
        self.fitness = None
        self.adjusted_fitness = None
        self.fitness_history = []

    def update(self, representative, members):
        self.representative = representative
        self.members = members

    def get_fitnesses(self):
        return [m.fitness for m in self.members.values()]


class GenomeDistanceCache(object):
    def __init__(self, config):
        self.distances = {}
        self.config = config
        self.hits = 0
        self.misses = 0

    def __call__(self, genome0, genome1):
        g0 = genome0.key
        g1 = genome1.key
        d = self.distances.get((g0, g1))
        if d is None:
            # Distance is not already computed.
            d = genome0.distance(genome1, self.config)
            self.distances[g0, g1] = d
            self.distances[g1, g0] = d
            self.misses += 1
        else:
            self.hits += 1

        return d


class DefaultSpeciesSet(DefaultClassConfig):
    """Encapsulates the default speciation scheme."""

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.species_set_config = config
        self.reporters = reporters
        self.indexer = count(1)
        self.species = {}
        self.genome_to_species = {}

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(
            param_dict, [ConfigParameter("compatibility_threshold", float)]
        )

    def speciate(self, config, population, generation):
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        assert isinstance(population, dict)

        compatibility_threshold = self.species_set_config.compatibility_threshold

        # Find the best representatives for each existing species.
        unspeciated = set(population)
        distances = GenomeDistanceCache(config.genome_config)
        new_representatives = {}
        new_members = {}
        for sid, s in self.species.items():
            candidates = []
            for gid in unspeciated:
                g = population[gid]
                d = distances(s.representative, g)
                candidates.append((d, g))

            # The new representative is the genome closest to the current representative.
            ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
            new_rid = new_rep.key
            new_representatives[sid] = new_rid
            new_members[sid] = [new_rid]
            unspeciated.remove(new_rid)

        # Partition population into species based on genetic similarity.
        while unspeciated:
            gid = unspeciated.pop()
            g = population[gid]

            # Find the species with the most similar representative.
            candidates = []
            for sid, rid in new_representatives.items():
                rep = population[rid]
                d = distances(rep, g)
                if d < compatibility_threshold:
                    candidates.append((d, sid))

            if candidates:
                ignored_sdist, sid = min(candidates, key=lambda x: x[0])
                new_members[sid].append(gid)
            else:
                # No species is similar enough, create a new species, using
                # this genome as its representative.
                sid = next(self.indexer)
                new_representatives[sid] = gid
                new_members[sid] = [gid]

        # Update species collection based on new speciation.
        self.genome_to_species = {}
        for sid, rid in new_representatives.items():
            s = self.species.get(sid)
            if s is None:
                s = Species(sid, generation)
                self.species[sid] = s

            members = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

            member_dict = dict((gid, population[gid]) for gid in members)
            s.update(population[rid], member_dict)

        gdmean = mean(distances.distances.values())
        gdstdev = stdev(distances.distances.values())
        self.reporters.info(
            "Mean genetic distance {0:.3f}, standard deviation {1:.3f}".format(
                gdmean, gdstdev
            )
        )

    def get_species_id(self, individual_id):
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id):
        sid = self.genome_to_species[individual_id]
        return self.species[sid]


class DefaultStagnation(DefaultClassConfig):
    """Keeps track of whether species are making progress and helps remove ones that are not."""

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(
            param_dict,
            [
                ConfigParameter("species_fitness_func", str, "mean"),
                ConfigParameter("max_stagnation", int, 15),
                ConfigParameter("species_elitism", int, 0),
            ],
        )

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.stagnation_config = config

        self.species_fitness_func = stat_functions.get(config.species_fitness_func)
        if self.species_fitness_func is None:
            raise RuntimeError(
                "Unexpected species fitness func: {0!r}".format(
                    config.species_fitness_func
                )
            )

        self.reporters = reporters

    def update(self, species_set, generation):
        """
        Required interface method. Updates species fitness history information,
        checking for ones that have not improved in max_stagnation generations,
        and - unless it would result in the number of species dropping below the configured
        species_elitism parameter if they were removed,
        in which case the highest-fitness species are spared -
        returns a list with stagnant species marked for removal.
        """
        species_data = []
        for sid, s in species_set.species.items():
            if s.fitness_history:
                prev_fitness = max(s.fitness_history)
            else:
                prev_fitness = -sys.float_info.max

            s.fitness = self.species_fitness_func(s.get_fitnesses())
            s.fitness_history.append(s.fitness)
            s.adjusted_fitness = None
            if prev_fitness is None or s.fitness > prev_fitness:
                s.last_improved = generation

            species_data.append((sid, s))

        # Sort in ascending fitness order.
        species_data.sort(key=lambda x: x[1].fitness)

        result = []
        species_fitnesses = []
        num_non_stagnant = len(species_data)
        for idx, (sid, s) in enumerate(species_data):
            # Override stagnant state if marking this species as stagnant would
            # result in the total number of species dropping below the limit.
            # Because species are in ascending fitness order, less fit species
            # will be marked as stagnant first.
            stagnant_time = generation - s.last_improved
            is_stagnant = False
            if num_non_stagnant > self.stagnation_config.species_elitism:
                is_stagnant = stagnant_time >= self.stagnation_config.max_stagnation

            if (len(species_data) - idx) <= self.stagnation_config.species_elitism:
                is_stagnant = False

            if is_stagnant:
                num_non_stagnant -= 1

            result.append((sid, s, is_stagnant))
            species_fitnesses.append(s.fitness)

        return result


class DefaultReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(
            param_dict,
            [
                ConfigParameter("elitism", int, 0),
                ConfigParameter("survival_threshold", float, 0.2),
                ConfigParameter("min_species_size", int, 1),
            ],
        )

    def __init__(self, config, reporters, stagnation):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = {}

    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()

        return new_genomes

    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        """Compute the proper number of offspring per species (proportional to fitness)."""
        af_sum = sum(adjusted_fitness)

        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size

            d = (s - ps) * 0.5
            c = int(round(d))
            spawn = ps
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1

            spawn_amounts.append(spawn)

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        spawn_amounts = [
            max(min_species_size, int(round(n * norm))) for n in spawn_amounts
        ]

        return spawn_amounts

    def reproduce(self, config, species, pop_size, generation):
        """
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents.
        """
        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        # The average adjusted fitness scheme (normalized to the interval
        # [0, 1]) allows the use of negative fitness values without
        # interfering with the shared fitness scheme.
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.fitness for m in stag_s.members.values())
                remaining_species.append(stag_s)
        # The above comment was not quite what was happening - now getting fitnesses
        # only from members of non-stagnated species.

        # No species left.
        if not remaining_species:
            species.species = {}
            return {}  # was []

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in afs.members.values()])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses)  # type: float
        self.reporters.info(
            "Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness)
        )

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size = max(min_species_size, self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(
            adjusted_fitnesses, previous_sizes, pop_size, min_species_size
        )

        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.reproduction_config.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(s.members.items())
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites to new generation.
            if self.reproduction_config.elitism > 0:
                for i, m in old_members[: self.reproduction_config.elitism]:
                    new_population[i] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(
                math.ceil(
                    self.reproduction_config.survival_threshold * len(old_members)
                )
            )
            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]

            # Randomly choose parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                spawn -= 1

                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)
                child.mutate(config.genome_config)

                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population


class DefaultGenomeConfig(object):
    """Sets up and holds configuration information for the DefaultGenome class."""

    allowed_connectivity = [
        "unconnected",
        "fs_neat_nohidden",
        "fs_neat",
        "fs_neat_hidden",
        "full_nodirect",
        "full",
        "full_direct",
        "partial_nodirect",
        "partial",
        "partial_direct",
    ]

    def __init__(self, params):
        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.aggregation_function_defs = AggregationFunctionSet()
        self.aggregation_defs = self.aggregation_function_defs

        self._params = [
            ConfigParameter("num_inputs", int),
            ConfigParameter("num_outputs", int),
            ConfigParameter("num_hidden", int),
            ConfigParameter("feed_forward", bool),
            ConfigParameter("compatibility_disjoint_coefficient", float),
            ConfigParameter("compatibility_weight_coefficient", float),
            ConfigParameter("conn_add_prob", float),
            ConfigParameter("conn_delete_prob", float),
            ConfigParameter("node_add_prob", float),
            ConfigParameter("node_delete_prob", float),
            ConfigParameter("single_structural_mutation", bool, "false"),
            ConfigParameter("structural_mutation_surer", str, "default"),
            ConfigParameter("initial_connection", str, "unconnected"),
        ]

        # Gather configuration data from the gene classes.
        self.node_gene_type = params["node_gene_type"]
        self._params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params["connection_gene_type"]
        self._params += self.connection_gene_type.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

        self.connection_fraction = None

        # Verify that initial connection type is valid.
        # pylint: disable=access-member-before-definition
        if "partial" in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive."
                )

        assert self.initial_connection in self.allowed_connectivity

        # Verify structural_mutation_surer is valid.
        # pylint: disable=access-member-before-definition
        if self.structural_mutation_surer.lower() in ["1", "yes", "true", "on"]:
            self.structural_mutation_surer = "true"
        elif self.structural_mutation_surer.lower() in ["0", "no", "false", "off"]:
            self.structural_mutation_surer = "false"
        elif self.structural_mutation_surer.lower() == "default":
            self.structural_mutation_surer = "default"
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer
            )
            raise RuntimeError(error_string)

        self.node_indexer = None

    def add_activation(self, name, func):
        self.activation_defs.add(name, func)

    def add_aggregation(self, name, func):
        self.aggregation_function_defs.add(name, func)

    def save(self, f):
        if "partial" in self.initial_connection:
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive."
                )
            f.write(
                "initial_connection      = {0} {1}\n".format(
                    self.initial_connection, self.connection_fraction
                )
            )
        else:
            f.write("initial_connection      = {0}\n".format(self.initial_connection))

        assert self.initial_connection in self.allowed_connectivity

        write_pretty_params(
            f, self, [p for p in self._params if "initial_connection" not in p.name]
        )

    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            self.node_indexer = count(max(list(node_dict)) + 1)

        new_id = next(self.node_indexer)

        assert new_id not in node_dict

        return new_id

    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == "true":
            return True
        elif self.structural_mutation_surer == "false":
            return False
        elif self.structural_mutation_surer == "default":
            return self.single_structural_mutation
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer
            )
            raise RuntimeError(error_string)


class DefaultGenome(object):
    """
    A genome for generalized neural networks.

    Terminology
        pin: Point at which the network is conceptually connected to the external world;
             pins are either input or output.
        node: Analog of a physical neuron.
        connection: Connection between a pin/node output and a node's input, or between a node's
             output and a pin/node input.
        key: Identifier for an object, unique within the set of similar objects.

    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique
           neuron by an implicit connection with weight one. This connection
           is permanently enabled.
        2. The output pin's key is always the same as the key for its
           associated neuron.
        3. Output neurons can be modified but not deleted.
        4. The input values are applied to the input pins unmodified.
    """

    @classmethod
    def parse_config(cls, param_dict):
        param_dict["node_gene_type"] = DefaultNodeGene
        param_dict["connection_gene_type"] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        # Unique identifier for a genome instance.
        self.key = key

        # (gene_key, gene) pairs for gene sets.
        self.connections = {}
        self.nodes = {}

        # Fitness results.
        self.fitness = None

    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""

        # Create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

        # Add hidden nodes if requested.
        if config.num_hidden > 0:
            for i in range(config.num_hidden):
                node_key = config.get_new_node_key(self.nodes)
                assert node_key not in self.nodes
                node = self.create_node(config, node_key)
                self.nodes[node_key] = node

        # Add connections based on initial connectivity type.

        if "fs_neat" in config.initial_connection:
            if config.initial_connection == "fs_neat_nohidden":
                self.connect_fs_neat_nohidden(config)
            elif config.initial_connection == "fs_neat_hidden":
                self.connect_fs_neat_hidden(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = fs_neat will not connect to hidden nodes;",
                        "\tif this is desired, set initial_connection = fs_neat_nohidden;",
                        "\tif not, set initial_connection = fs_neat_hidden",
                        sep="\n",
                        file=sys.stderr,
                    )
                self.connect_fs_neat_nohidden(config)
        elif "full" in config.initial_connection:
            if config.initial_connection == "full_nodirect":
                self.connect_full_nodirect(config)
            elif config.initial_connection == "full_direct":
                self.connect_full_direct(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = full with hidden nodes will not do direct input-output connections;",
                        "\tif this is desired, set initial_connection = full_nodirect;",
                        "\tif not, set initial_connection = full_direct",
                        sep="\n",
                        file=sys.stderr,
                    )
                self.connect_full_nodirect(config)
        elif "partial" in config.initial_connection:
            if config.initial_connection == "partial_nodirect":
                self.connect_partial_nodirect(config)
            elif config.initial_connection == "partial_direct":
                self.connect_partial_direct(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = partial with hidden nodes will not do direct input-output connections;",
                        "\tif this is desired, set initial_connection = partial_nodirect {0};".format(
                            config.connection_fraction
                        ),
                        "\tif not, set initial_connection = partial_direct {0}".format(
                            config.connection_fraction
                        ),
                        sep="\n",
                        file=sys.stderr,
                    )
                self.connect_partial_nodirect(config)

    def configure_crossover(self, genome1, genome2, config):
        """Configure a new genome by crossover from two parent genomes."""
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit connection genes
        for key, cg1 in parent1.connections.items():
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in parent1_set.items():
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)

    def mutate(self, config):
        """Mutates this genome."""

        if config.single_structural_mutation:
            div = max(
                1,
                (
                    config.node_add_prob
                    + config.node_delete_prob
                    + config.conn_add_prob
                    + config.conn_delete_prob
                ),
            )
            r = random.random()
            if r < (config.node_add_prob / div):
                self.mutate_add_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob) / div):
                self.mutate_delete_node(config)
            elif r < (
                (config.node_add_prob + config.node_delete_prob + config.conn_add_prob)
                / div
            ):
                self.mutate_add_connection(config)
            elif r < (
                (
                    config.node_add_prob
                    + config.node_delete_prob
                    + config.conn_add_prob
                    + config.conn_delete_prob
                )
                / div
            ):
                self.mutate_delete_connection()
        else:
            if random.random() < config.node_add_prob:
                self.mutate_add_node(config)

            if random.random() < config.node_delete_prob:
                self.mutate_delete_node(config)

            if random.random() < config.conn_add_prob:
                self.mutate_add_connection(config)

            if random.random() < config.conn_delete_prob:
                self.mutate_delete_connection()

        # Mutate connection genes.
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.nodes.values():
            ng.mutate(config)

    def mutate_add_node(self, config):
        if not self.connections:
            if config.check_structural_mutation_surer():
                self.mutate_add_connection(config)
            return

        # Choose a random connection to split
        conn_to_split = random.choice(list(self.connections.values()))
        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        conn_to_split.enabled = False

        i, o = conn_to_split.key
        self.add_connection(config, i, new_node_id, 1.0, True)
        self.add_connection(config, new_node_id, o, conn_to_split.weight, True)

    def add_connection(self, config, input_key, output_key, weight, enabled):
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0
        assert isinstance(enabled, bool)
        key = (input_key, output_key)
        connection = config.connection_gene_type(key)
        connection.init_attributes(config)
        connection.weight = weight
        connection.enabled = enabled
        self.connections[key] = connection

    def mutate_add_connection(self, config):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        """
        possible_outputs = list(self.nodes)
        out_node = random.choice(possible_outputs)

        possible_inputs = possible_outputs + config.input_keys
        in_node = random.choice(possible_inputs)

        # Don't duplicate connections.
        key = (in_node, out_node)
        if key in self.connections:
            if config.check_structural_mutation_surer():
                self.connections[key].enabled = True
            return

        # Don't allow connections between two output nodes
        if in_node in config.output_keys and out_node in config.output_keys:
            return

        # No need to check for connections between input nodes:
        # they cannot be the output end of a connection (see above).

        # For feed-forward networks, avoid creating cycles.
        if config.feed_forward and creates_cycle(list(self.connections), key):
            return

        cg = self.create_connection(config, in_node, out_node)
        self.connections[cg.key] = cg

    def mutate_delete_node(self, config):
        # Do nothing if there are no non-output nodes.
        available_nodes = [k for k in self.nodes if k not in config.output_keys]
        if not available_nodes:
            return -1

        del_key = random.choice(available_nodes)

        connections_to_delete = set()
        for k, v in self.connections.items():
            if del_key in v.key:
                connections_to_delete.add(v.key)

        for key in connections_to_delete:
            del self.connections[key]

        del self.nodes[del_key]

        return del_key

    def mutate_delete_connection(self):
        if self.connections:
            key = random.choice(list(self.connections.keys()))
            del self.connections[key]

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in other.nodes:
                if k2 not in self.nodes:
                    disjoint_nodes += 1

            for k1, n1 in self.nodes.items():
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (
                node_distance
                + (config.compatibility_disjoint_coefficient * disjoint_nodes)
            ) / max_nodes

        # Compute connection gene differences.
        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in other.connections:
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, c1 in self.connections.items():
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (
                connection_distance
                + (config.compatibility_disjoint_coefficient * disjoint_connections)
            ) / max_conn

        distance = node_distance + connection_distance
        return distance

    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        num_enabled_connections = sum(
            [1 for cg in self.connections.values() if cg.enabled]
        )
        return len(self.nodes), num_enabled_connections

    def __str__(self):
        s = "Key: {0}\nFitness: {1}\nNodes:".format(self.key, self.fitness)
        for k, ng in self.nodes.items():
            s += "\n\t{0} {1!s}".format(k, ng)
        s += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)
        return s

    @staticmethod
    def create_node(config, node_id):
        node = config.node_gene_type(node_id)
        node.init_attributes(config)
        return node

    @staticmethod
    def create_connection(config, input_id, output_id):
        connection = config.connection_gene_type((input_id, output_id))
        connection.init_attributes(config)
        return connection

    def connect_fs_neat_nohidden(self, config):
        """
        Randomly connect one input to all output nodes
        (FS-NEAT without connections to hidden, if any).
        Originally connect_fs_neat.
        """
        input_id = random.choice(config.input_keys)
        for output_id in config.output_keys:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_fs_neat_hidden(self, config):
        """
        Randomly connect one input to all hidden and output nodes
        (FS-NEAT with connections to hidden, if any).
        """
        input_id = random.choice(config.input_keys)
        others = [i for i in self.nodes if i not in config.input_keys]
        for output_id in others:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def compute_full_connections(self, config, direct):
        """
        Compute connections for a fully-connected feed-forward genome--each
        input connected to all hidden nodes
        (and output nodes if ``direct`` is set or there are no hidden nodes),
        each hidden node connected to all output nodes.
        (Recurrent genomes will also include node self-connections.)
        """
        hidden = [i for i in self.nodes if i not in config.output_keys]
        output = [i for i in self.nodes if i in config.output_keys]
        connections = []
        if hidden:
            for input_id in config.input_keys:
                for h in hidden:
                    connections.append((input_id, h))
            for h in hidden:
                for output_id in output:
                    connections.append((h, output_id))
        if direct or (not hidden):
            for input_id in config.input_keys:
                for output_id in output:
                    connections.append((input_id, output_id))

        # For recurrent genomes, include node self-connections.
        if not config.feed_forward:
            for i in self.nodes:
                connections.append((i, i))

        return connections

    def connect_full_nodirect(self, config):
        """
        Create a fully-connected genome
        (except without direct input-output unless no hidden nodes).
        """
        for input_id, output_id in self.compute_full_connections(config, False):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_full_direct(self, config):
        """Create a fully-connected genome, including direct input-output connections."""
        for input_id, output_id in self.compute_full_connections(config, True):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_nodirect(self, config):
        """
        Create a partially-connected genome,
        with (unless no hidden nodes) no direct input-output connections."""
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, False)
        random.shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_direct(self, config):
        """
        Create a partially-connected genome,
        including (possibly) direct input-output connections.
        """
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, True)
        random.shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection


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


################################################################################
# CPPN code
# from PyTorch-NEAT 
################################################################################

# Activation functions

def sigmoid_activation(x):
    return torch.sigmoid(5 * x)


def tanh_activation(x):
    return torch.tanh(2.5 * x)


def abs_activation(x):
    return torch.abs(x)


def gauss_activation(x):
    return torch.exp(-5.0 * x**2)


def identity_activation(x):
    return x


def sin_activation(x):
    return torch.sin(x)


def relu_activation(x):
    return F.relu(x)


str_to_activation = {
    'sigmoid': sigmoid_activation,
    'tanh': tanh_activation,
    'abs': abs_activation,
    'gauss': gauss_activation,
    'identity': identity_activation,
    'sin': sin_activation,
    'relu': relu_activation,
}

# Aggregation functions

def sum_aggregation(inputs):
    return sum(inputs)


def prod_aggregation(inputs):
    return reduce(mul, inputs, 1)


str_to_aggregation = {
    'sum': sum_aggregation,
    'prod': prod_aggregation,
}


def create_cppn(genome, config, leaf_names, node_names, output_activation=None):
    """
    Create CPPN from genome
    """
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

################################################################################
# Co-optimizing the design and control of soft robots.
# Code taken from EvoGym - https://github.com/EvolutionGym/evogym - v2.0.0
################################################################################

class Population(neat.Population):
    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(
            config.reproduction_config, self.reporters, stagnation
        )
        if config.fitness_criterion == "max":
            self.fitness_criterion = max
        elif config.fitness_criterion == "min":
            self.fitness_criterion = min
        elif config.fitness_criterion == "mean":
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion)
            )

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(
                config.genome_type, config.genome_config, config.pop_size
            )
            self.species = config.species_set_type(
                config.species_set_config, self.reporters
            )
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
            raise RuntimeError(
                "Cannot have no generational limit with no fitness termination"
            )

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
                    new_population = self.reproduction.create_new(
                        self.config.genome_type,
                        self.config.genome_config,
                        self.config.pop_size - len(valid_genomes),
                    )
                    new_genomes = list(new_population.items())
                    validity = constraint_function(
                        new_genomes, self.config, self.generation
                    )
                    valid_idx = np.where(validity)[0]
                    valid_genomes = np.vstack(
                        [valid_genomes, np.array(new_genomes)[valid_idx]]
                    )

                self.population = dict(valid_genomes)
                self.species.speciate(self.config, self.population, self.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(
                list(self.population.items()), self.config, self.generation
            )

            # Gather and report statistics.
            best = None
            for g in self.population.values():
                if g.fitness is None:
                    raise RuntimeError(
                        "Fitness not assigned to genome {}".format(g.key)
                    )

                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(
                self.config, self.population, self.species, best
            )

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
            self.population = self.reproduction.reproduce(
                self.config, self.species, self.config.pop_size, self.generation
            )

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(
                        self.config.genome_type,
                        self.config.genome_config,
                        self.config.pop_size,
                    )
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(
                self.config, self.generation, self.best_genome
            )

        return self.best_genome


def get_cppn_input(structure_shape):
    x, y = torch.meshgrid(
        torch.arange(structure_shape[0]), torch.arange(structure_shape[1])
    )
    x, y = x.flatten(), y.flatten()
    center = (np.array(structure_shape) - 1) / 2
    d = ((x - center[0]) ** 2 + (y - center[1]) ** 2).sqrt()
    return x, y, d


def get_robot_from_genome(genome, config):
    nodes = create_cppn(
        genome,
        config,
        leaf_names=["x", "y", "d"],
        node_names=["empty", "rigid", "soft", "hori", "vert"],
    )
    structure_shape = config.extra_info["structure_shape"]
    x, y, d = get_cppn_input(structure_shape)
    material = []
    for node in nodes:
        material.append(node(x=x, y=y, d=d).numpy())
    material = np.vstack(material).argmax(axis=0)
    robot = material.reshape(structure_shape)
    return robot


def eval_genome_fitness(genome, config, genome_id, generation):
    robot = get_robot_from_genome(genome, config)
    args, env_name = config.extra_info["args"], config.extra_info["env_name"]

    connectivity = get_full_connectivity(robot)
    save_path_generation = os.path.join(
        config.extra_info["save_path"], f"generation_{generation}"
    )
    save_path_structure = os.path.join(
        save_path_generation, "structure", f"{genome_id}"
    )
    save_path_controller = os.path.join(save_path_generation, "controller")
    np.savez(save_path_structure, robot, connectivity)

    fitness = run_ppo(
        args, robot, env_name, save_path_controller, f"{genome_id}", connectivity
    )
    return fitness


def eval_genome_constraint(genome, config, genome_id, generation):
    robot = get_robot_from_genome(genome, config)
    validity = is_connected(robot) and has_actuator(robot)
    if validity:
        robot_hash = hashable(robot)
        if robot_hash in config.extra_info["structure_hashes"]:
            validity = False
        else:
            config.extra_info["structure_hashes"][robot_hash] = True
    return validity


class SaveResultReporter(neat.BaseReporter):
    """Reporter that saves generation results to disk, including robot structures and fitness rankings."""
    def __init__(self, save_path):
        """Initialize the reporter with the path where results will be saved.
        
        Args:
            save_path: Base directory path for saving results
        """
        super().__init__()
        self.save_path = save_path
        self.generation = None

    def start_generation(self, generation):
        """Called at the start of each generation to create necessary directories.
        
        Args:
            generation: Current generation number
        """
        self.generation = generation
        # Create directories for robot structure files
        save_path_structure = os.path.join(
            self.save_path, f"generation_{generation}", "structure"
        )
        # Create directories for controller files
        save_path_controller = os.path.join(
            self.save_path, f"generation_{generation}", "controller"
        )
        os.makedirs(save_path_structure, exist_ok=True)
        os.makedirs(save_path_controller, exist_ok=True)

    def post_evaluate(self, config, population, species, best_genome):
        """Called after evaluating the population to save fitness rankings.
        
        Args:
            config: NEAT configuration
            population: Current population of genomes
            species: Current species set
            best_genome: Genome with highest fitness
        """
        # Path for saving fitness rankings
        save_path_ranking = os.path.join(
            self.save_path, f"generation_{self.generation}", "output.txt"
        )
        # Convert population to lists for sorting
        genome_id_list, genome_list = (
            np.arange(len(population)),
            np.array(list(population.values())),
        )
        # Sort genomes by fitness in descending order
        sorted_idx = sorted(
            genome_id_list, key=lambda i: genome_list[i].fitness, reverse=True
        )
        genome_id_list, genome_list = (
            list(genome_id_list[sorted_idx]),
            list(genome_list[sorted_idx]),
        )
        # Write fitness rankings to file
        with open(save_path_ranking, "w") as f:
            out = ""
            for genome_id, genome in zip(genome_id_list, genome_list):
                out += f"{genome_id}\t\t{genome.fitness}\n"
            f.write(out)


class ConnectionReporter(neat.BaseReporter):
    def __init__(self, connection_history = None):
        super().__init__()
        if connection_history is None:
            self.connection_history = []
        else:
            self.connection_history = connection_history
        self.generation = None
        
    def start_generation(self, generation):
        self.generation = generation
        
    def post_evaluate(self, config, population, species, best_genome):
        # Calculate average and max number of connections for this generation
        num_connections = [len(genome.connections) for genome in population.values()]
        connection_stats = {
            "generation": self.generation,
            "avg_connections": float(np.mean(num_connections)),
            "max_connections": int(np.max(num_connections)),
            "min_connections": int(np.min(num_connections)),
            "best_genome_connections": len(best_genome.connections)
        }
        self.connection_history.append(connection_stats)
class SpeciesReporter(neat.BaseReporter):
    def __init__(self, species_history=None):
        super().__init__()
        if species_history is None:
            self.species_history = []
        else:
            self.species_history = species_history
        self.generation = None
        
    def start_generation(self, generation):
        self.generation = generation
        
    def post_evaluate(self, config, population, species, best_genome):
        # Calculate statistics about species for this generation
        species_stats = {
            "generation": self.generation,
            "num_species": len(species.species),
            "species_sizes": [len(s.members) for s in species.species.values()],
            "species_fitness": [s.fitness for s in species.species.values()]
        }
        
        # Find which species contains the best genome
        best_species_id = None
        for sid, s in species.species.items():
            if best_genome.key in s.members:
                best_species_id = sid
                break
        species_stats["best_species_id"] = best_species_id
        
        # Add average fitness per species
        species_avg_fitness = {}
        for sid, s in species.species.items():
            member_fitness = [m.fitness for m in s.members.values()]
            species_avg_fitness[sid] = float(np.mean(member_fitness))
        species_stats["species_avg_fitness"] = species_avg_fitness
        
        self.species_history.append(species_stats)


def run_cppn_neat(config: ExperimentConfig):
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
        print(f"THIS EXPERIMENT ({exp_name}) ALREADY EXISTS")
        print("Override? (y/n): ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        else:
            return None, None
        print()

    save_path_metadata = os.path.join(save_path, "metadata.txt")
    with open(save_path_metadata, "w") as f:
        f.write(
            f"POP_SIZE: {pop_size}\n"
            f"STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n"
            f"MAX_EVALUATIONS: {max_evaluations}\n"
        )

    structure_hashes = {}

    config_path = os.path.join(curr_dir, "neat.cfg")
    config = neat.Config(
        DefaultGenome,
        DefaultReproduction,
        DefaultSpeciesSet,
        DefaultStagnation,
        config_path,
        extra_info={
            "structure_shape": structure_shape,
            "save_path": save_path,
            "structure_hashes": structure_hashes,
            "args": config.ppo,  # args for run_ppo
            "env_name": env_name,
        },
        custom_config=[
            ("NEAT", "pop_size", pop_size),
        ],
    )

    pop = Population(config)
    reporters = [
        neat.StatisticsReporter(),
        neat.StdOutReporter(True),
        SaveResultReporter(save_path),
        ConnectionReporter(),
        SpeciesReporter(),
    ]
    index_connection_reporter = 3
    index_species_reporter = 4
    for reporter in reporters:
        pop.add_reporter(reporter)

    evaluator = ParallelEvaluator(
        num_cores, eval_genome_fitness, eval_genome_constraint
    )

    # Run evolution
    pop.run(
        evaluator.evaluate_fitness,
        evaluator.evaluate_constraint,
        n=np.ceil(max_evaluations / pop_size),
    )

    best_robot = get_robot_from_genome(pop.best_genome, config)
    best_fitness = pop.best_genome.fitness

    connection_history = pop.reporters.reporters[index_connection_reporter].connection_history
    species_history = pop.reporters.reporters[index_species_reporter].species_history
    
    return best_robot, best_fitness, connection_history, species_history, len(pop.best_genome.connections)

################################################################################
# Main
################################################################################

def main():
    parser = argparse.ArgumentParser(
        description="CPPN-NEAT for co-optimizing the design and control of soft robots"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Path to save experiment data"
    )
    args = parser.parse_args()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # Record start time
    start_time = datetime.now()

    config = ExperimentConfig(save_path=args.out_dir)
    best_robot, best_fitness, connection_history, species_history, best_connections = run_cppn_neat(config)

    # Record end time and calculate duration
    end_time = datetime.now()
    duration = str(end_time - start_time)

    # Convert robot list to list of block strings for visualization
    block_mapping = {0: "empty", 1: "rigid block", 2: "soft block", 3: "horizontal actuator", 4: "vertical actuator"}
    robot_blocks = [[block_mapping[cell] for cell in row] for row in best_robot]


    # Prepare final info dictionary
    final_info = {
        config.env_name: {
            "means": {
                "best_fitness": float(best_fitness) if best_fitness is not None else None, 
            },
            "best_robot_properties": {
                "desc": robot_blocks if best_robot is not None else None,
                "best_robot_shape": best_robot.shape if best_robot is not None else None,
                "best_genome_connections": best_connections,
                "connection_history": connection_history,
                "species_history": species_history
            },
            "run_info": {
                "seed": seed,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(), 
                "duration": duration,
                "out_dir": args.out_dir
            }
        }
    }

    # Save final info to JSON file
    json_path = os.path.join(args.out_dir, "final_info.json")
    with open(json_path, 'w') as f:
        json.dump(final_info, f, indent=4)

    print("Best robot:")
    print(best_robot)
    print("Best fitness:", best_fitness)
    print(f"Best genome connections: {best_connections}")
    print(f"Final info saved to: {json_path}")

if __name__ == "__main__":
    main()
