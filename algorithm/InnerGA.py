from deap.tools import mutGaussian
import numpy as np
from numpy.random import randint, random, choice


class InnerGA:
    def gen_population(self, global_cap: np.array) -> np.array:
        population = np.apply_along_axis(
            func=lambda x: x*0.95, axis=0, arr=global_cap)
        return population

    def fitness_inga(self, capacity: np.array, flow: np.array) -> np.double:

        if (flow > (capacity * 0.95)).any():
            return 0

        sigma = flow.sum()
        temp = (flow/(capacity-flow)).sum()
        T = temp/sigma
        return 1/T

    def util(self, flow: np.array, capacity: np.array) -> np.double:
        return flow/capacity

    def tournament_selection(self, size: int, population: np.array, fitness: np.array, k=5) -> [np.array, np.array]:
        index = []
        population = list(population)
        select = []
        for _ in range(size):
            selection_ix = randint(len(population))
            for ix in randint(0, len(population), k-1):
                if fitness[ix] > fitness[selection_ix]:
                    selection_ix = ix
                select.append(population.pop(selection_ix))
                index.append(selection_ix)

        return np.array(select), np.array(index)

    def scatter_crossover(self, parent1: np.array, parent2: np.array, alpha: int, indPb=0.8) -> [np.array, np.array]:
        if random() > indPb:
            return False

        child1, child2 = np.copy(parent1), np.copy(parent2)
        number_of_genes = len(parent1)
        scatter_point = choice(number_of_genes, size=int(
            alpha * number_of_genes), replace=False)

        for i in scatter_point:
            child1[i] = parent2[i]
            child2[i] = parent1[i]

        return child1, child2

    def genetic_algorithm(self, size=10, generation=100, capacity=None, alpha=0.5, pMu=0.2, pCr=0.8):
        population = self.generate_population(size, capacity)

        for gen in range(generation):
            fitness = np.array(
                [self.fitness_inga(capacity=capacity, flow=individual) for individual in population])
            selected, _ = self.tournament_selection(
                size=size, population=population, fitness=fitness)

            sh = selected[0].shape[0]
            offspring = np.empty((0, sh))

            for parent1, parent2 in zip(selected[::2], selected[1::2]):
                if cross := self.scatter_crossover(alpha=alpha, parent1=parent1, parent2=parent2, indPb=pCr):
                    child1, child2 = cross
                    offspring = np.append(offspring, child1.reshape(1, sh), )
                    offspring = np.append(
                        offspring, child2.reshape(1, sh), axis=0)
                else:
                    offspring = np.append(
                        offspring, parent1.reshape(1, sh), axis=0)
                    offspring = np.append(
                        offspring, parent2.reshape(1, sh), axis=0)

            for m in offspring:
                mut = mutGaussian(m, 0, 1, pMu)[0]
                offspring = np.append(offspring, mut.reshape(1, sh), axis=0)

            pop = offspring

        return pop, np.array([self.fitness_inga(capacity=capacity, flow=individual) for individual in population])
