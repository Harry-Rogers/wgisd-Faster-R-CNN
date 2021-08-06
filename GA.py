#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:57:34 2021

@author: harry
"""

import numpy as np
from engine import evaluate
import torch



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
        print("CHROMS")
        print(Individual(child_chromosome))
        return Individual(child_chromosome)
    
    

    
    
    def cal_fitness(self, model, data_loader_test, device):
        """
		NEED A WAY TO SEE IF WEIGHT HAS BEEN MODIFIED
        COULD DO UNIQUE FROM NP
        MIGHT BE WAY IN TORCH
		"""
        for i in range(len(self.chromosome)):
            m1 = self.chromosome[i]
            m1_check = torch.tensor(m1)
            if m1_check.shape == model.backbone.body.conv1.weight.shape:
                m1 = torch.nn.Parameter(torch.tensor(m1, device='cuda:0'), False)
                m1 = torch.squeeze(m1, 0)
                model.backbone.body.conv1.weight = m1
            elif m1_check.shape == model.backbone.body.layer1[0].conv1.weight.shape:
                m1 = torch.nn.Parameter(torch.tensor(m1, device='cuda:0'), False)
                m1 = torch.squeeze(m1, 0)
                model.backbone.body.conv1.weight = m1
    
        coco_eval, metric_logger = evaluate(model, data_loader_test, device=device)
        # L might be tf related?
        # a is eval score can just pass coco scores back
        AP_1 = metric_logger[91:97]
        AP_2 = metric_logger[170:176]
        AP_3 = metric_logger[251:257]
        AP_4 = metric_logger[331:337]#Negative can ignore also no small images in data
        AP_5 = metric_logger[412:418]
        AP_6 = metric_logger[492:498]
        
        AR_1 = metric_logger[572:578]
        AR_2 = metric_logger[652:658]
        AR_3 = metric_logger[732:738]
        AR_4 = metric_logger[812:818]#Negative can ignore
        AR_5 = metric_logger[892:898]
        AR_6 = metric_logger[973:979]
        
        fitness = float(AP_1) * float(AP_2) * float(AP_3) * float(AP_5) * float(AP_6) * float(AR_1) * float(AR_2) * float(AR_3) * float(AR_5) * float(AR_6)
        
        return fitness
