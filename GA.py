#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:57:34 2021

@author: harry
"""

import numpy as np
#import localutils


class Individual(object):
    """
	Class representing individual in population
	"""

    def __init__(self, chromosome):
        self.chromosome = chromosome
        # self.fitness = self.cal_fitness()

    @classmethod
    def mutated_genes(self):
        """
		create random genes for mutation
		"""
        #print(np.random.choice[0,1])
        return np.random.choice([0, 1])

    @classmethod
    def create_gnome_random(self, shapeslist):
        """
        create chromosome or string of genes
        """
        gnome_list = []
        for mshape in shapeslist:
            # gnome_list.append(np.random.randint(0, 2, mshape))
            # gnome_list.append(np.random.choice([0, 1.], mshape, p=[0.01, 0.99]))
            p0 = 0.1
            p1 = 1 - p0
            gnome_list.append(np.random.choice([0, 1], mshape, p=[p0, p1]))
        
        #print("LIST GNOME")
        #print(gnome_list)
        return gnome_list

    def mate(self, par2):
        """
        Perform mating and produce new offspring
        """

        # chromosome for offspring
        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):
            # print(gp1.shape)
            random_connections = False
            random_layers = True
            # bp = sorted([self, par2], key=lambda x: x.fitness)
            # prob1 = bp[0].fitness
            # prob2 = bp[1].fitness

            if random_connections:
                child_mask = np.zeros_like(gp1)

                noiseprob = 1 / gp1.size
                p2 = (1 - noiseprob) / 2
                p1 = p2
                p3 = noiseprob
                sum = p1 + p2 + p3
                p1 /= sum
                p2 /= sum
                p3 /= sum

                luck = np.random.choice([0, 1, 2], gp1.shape, p=[p1, p2, p3])

                index_p1 = luck == 0
                index_p2 = luck == 1
                index_mutation = luck == 2

                mutation = np.random.choice([0, 1], gp1.shape, p=[0.5, 0.5])
                child_mask[index_p1] = gp1[index_p1]
                child_mask[index_p2] = gp2[index_p2]
                child_mask[index_mutation] = mutation[index_mutation]
                child_chromosome.append(child_mask)

            if random_layers:
                child_mask = None
                # exchange masks randomly
                p = np.random.uniform()
                if p < .5:
                    child_mask = gp1.copy()
                else:
                    child_mask = gp2.copy()

                # do a minor random mutation in the child
                p0 = 0.05
                p1 = 1 - p0
                luck = np.random.choice([0, 1], gp1.shape, p=[p0, p1])
                index_mutation = luck == 0
                mutation = np.random.choice([0, 1], gp1.shape, p=[0.3, 0.7])
                child_mask[index_mutation] = mutation[index_mutation]
                child_chromosome.append(child_mask)

        # create new Individual(offspring) using
        # generated chromosome for offspring
        return Individual(child_chromosome)

    def cal_fitness(self, network, dataloader):
        """
		Calculate fittness score, it is the number of
		characters in string which differ from target
		string.
		"""

        i = 0
        for l in range(1, len(network.layers)):
            w = network.layers[l].get_weights()
            if isinstance(w, list):
                continue
            #print(self.chromosome)
            m1 = self.chromosome[l - 1]
            #print("M1")
            #print(m1)
            #print("\n")
            network.layers[l].set_weights([m1, w])
            i += 1

    
        coco_eval, metric_logger = evaluate(model, data_loader_test, device=device)
        # L might be tf related?
        # a is eval score can just pass coco scores back
        return a
