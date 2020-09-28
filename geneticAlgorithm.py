# -*- coding: utf-8 -*-



import io
import os
import math
import random
from json import load
from deap import base, creator, tools

def getRoute(individual, instance):
    #method that adjusts the route for future purposes

    route = []
    vehicle_capacity = instance['vehicle_capacity']
    depart_due_time = instance['depart']['due_time']
    # Initialize a sub-route
    sub_route = []
    vehicle_load = 0
    elapsed_time = 0
    last_customer_id = 0
    for customer_id in individual:
        # Update vehicle load
        demand = instance[f'customer_{customer_id}']['demand']
        updated_vehicle_load = vehicle_load + demand
        # Update elapsed time
        service_time = instance[f'customer_{customer_id}']['service_time']
        return_time = instance['distance_matrix'][customer_id][0]
        updated_elapsed_time = elapsed_time + \
            instance['distance_matrix'][last_customer_id][customer_id] + service_time + return_time
        # Validate vehicle load and elapsed time
        if (updated_vehicle_load <= vehicle_capacity) and (updated_elapsed_time <= depart_due_time):
            # Add to current sub-route
            sub_route.append(customer_id)
            vehicle_load = updated_vehicle_load
            elapsed_time = updated_elapsed_time - return_time
        else:
            # Save current sub-route
            route.append(sub_route)
            # Initialize a new sub-route and add to it
            sub_route = [customer_id]
            vehicle_load = demand
            elapsed_time = instance['distance_matrix'][0][customer_id] + service_time
        # Update last customer ID
        last_customer_id = customer_id
    if sub_route != []:
        # Save current sub-route before return if not empty
        route.append(sub_route)
    return route


def printOneRoute(route, temp=False):
    #method for printing one route
    route_str = '0'
    sub_route_count = 0
    for sub_route in route:
        sub_route_count += 1
        sub_route_str = '0'
        for customer_id in sub_route:
            sub_route_str = str(sub_route_str) + '->' + str(customer_id)
            route_str = str(route_str) + '->' + str(customer_id)
        sub_route_str = str(sub_route_str) + '->0'
        if not temp:
            print(str(sub_route_count) + ':'+ ' ' + str(sub_route_str))

        route_str = f'{route_str} - 0'
    if temp:
        print(route_str)
    return sub_route_count
    #print(sub_route_count)


def evaluationFunction(individual, instance, unit_cost=1.0, init_cost=0, wait_cost=0, delay_cost=0):
    #evaluation function written using pseudocode from publication
    route = getRoute(individual, instance)
    total_cost = 0
    for sub_route in route:
        sub_route_time_cost = 0
        sub_route_distance = 0
        elapsed_time = 0
        last_customer_id = 0
        for customer_id in sub_route:
            # Calculate section distance
            distance = instance['distance_matrix'][last_customer_id][customer_id]
            # Update sub-route distance
            sub_route_distance = sub_route_distance + distance
            # Calculate time cost
            arrival_time = elapsed_time + distance
            time_cost = wait_cost * \
                max(instance[f'customer_{customer_id}']['ready_time'] - arrival_time, 0) + \
                delay_cost * \
                max(arrival_time - instance[f'customer_{customer_id}']['due_time'], 0)
            # Update sub-route time cost
            sub_route_time_cost = sub_route_time_cost + time_cost
            # Update elapsed time
            elapsed_time = arrival_time + \
                instance[f'customer_{customer_id}']['service_time']
            # Update last customer ID
            last_customer_id = customer_id
        # Calculate transport cost
        sub_route_distance = sub_route_distance + instance['distance_matrix'][last_customer_id][0]
        sub_route_transport_cost = init_cost + unit_cost * sub_route_distance
        # Obtain sub-route cost
        sub_route_cost = sub_route_time_cost + sub_route_transport_cost
        # Update total cost
        total_cost = total_cost + sub_route_cost
    fitness = 1.0 / total_cost
    return (fitness, )


def crossIt(temp1, temp2, firstIndex, secondIndex):
    for gen in temp1:
        if gen not in firstIndex:
            firstIndex.append(gen)
    for gen in temp2:
        if gen not in secondIndex:
            secondIndex.append(gen)
    return firstIndex, secondIndex

def crossoverFunction(firstIndex, secondIndex):
    #modified crossover function which is available from deap
    size = min(len(firstIndex), len(secondIndex))
    cxFirst, cxSecond = sorted(random.sample(range(size), 2))
    temp1 = firstIndex[cxFirst:cxSecond + 1] + secondIndex
    temp2 = firstIndex[cxFirst:cxSecond + 1] + firstIndex
    firstIndex = []
    secondIndex = []
    firstIndex, secondIndex = crossIt(temp1, temp2, firstIndex, secondIndex)
    return firstIndex, secondIndex


def mutationFunction(individual):
    #adjusted mutation function of available mutation function of inverting indexes
    first, second = sorted(random.sample(range(len(individual)), 2)) #get indexes on which to invert an individual
    new_individual = individual[:first] + individual[second:first-1:-1] + individual[second+1:]
    return new_individual

def getInstance(instance_name):
    json_data_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data', 'json')
    json_file = os.path.join(json_data_dir, f'{instance_name}.json')
    with io.open(json_file, 'rt', newline='') as file_object:
        instance = load(file_object)
    return instance

def run_ga(instance_name, unit_cost, init_cost, wait_cost, delay_cost, ind_size, pop_size, \
    cx_pb, mut_pb, n_gen):
    #method from deap documentation
    instance = getInstance(instance_name)
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register('indexes', random.sample, range(1, ind_size + 1), ind_size)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', evaluationFunction, instance=instance, unit_cost=unit_cost, \
        init_cost=init_cost, wait_cost=wait_cost, delay_cost=delay_cost)
    toolbox.register('select', tools.selBest)
    toolbox.register('mate', crossoverFunction)
    toolbox.register('mutate', mutationFunction)
    pop = toolbox.population(n=pop_size)
    # Results holders for exporting results to CSV file
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print('  Evaluated ' + str(len(pop)) + ' individuals')
    # Begin the evolution
    for gen in range(n_gen):
        print('-* Generation number -> ' + str(gen) + ' *-')
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mut_pb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print('  Evaluated ' + str(len(invalid_ind)) + ' individuals')
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        #print(f"  Min {min(fits)}")
        #print(f'  Max {max(fits)}')
        print('  Avg ' + str(mean))
        print("\n")
        #print(f'  Std {std}')

    #print('-- End of (successful) evolution --')
    best_ind = tools.selBest(pop, 1)[0]
    print('Best individual -> ' + str(best_ind))
    print('Fitness: ' + str(best_ind.fitness.values[0]))
    print("\n")
    nmbr_of_routes =printOneRoute(getRoute(best_ind, instance))
    print(nmbr_of_routes)
    print(math.ceil(1 / best_ind.fitness.values[0]))

    """""
    f1 = open("output/file.txt", "w")
    print >>f1,nmbr_of_routes  
    """""