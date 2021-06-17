import random
import time
import numpy as np
rng = np.random.default_rng()

# This function is the test function F1，
# it can be obtained from Table 2 in the paper.

'''
This code is for learning only, please do not use for commercial activities.

For detailed algorithm details and test functions, 
please refer to paper "Flamingo search algorithm: A new swarm intelligence optimization algorithm".

All rights reserved. Please indicate the source of the paper.
'''
def fun(X):
    output = sum(np.square(X))
    return output

# This function is to initialize the flamingo population.
def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random()*(ub[j] - lb[j]) + lb[j]
    return X

# Calculate fitness values for each flamingo.
def CaculateFitness(X,fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


# Sort fitness.
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness,index


# Sort the position of the flamingos according to fitness.
def SortPosition(X,index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew


# Boundary detection function.
def BorderCheck(X,lb,ub,pop,dim):
    for i in range(pop):
        for j in range(dim):
            if X[i,j]<lb[j]:
                X[i,j] = ub[j]
            elif X[i,j]>ub[j]:
                X[i,j] = lb[j]
    return X


# This function is randomly evaluated between negative 1 and 1.
def rand_1():
    a=random.random()
    if a>0.5:
        return 1
    else:
        return -1


# The first phase migratory flamingo update function.
def congeal(X,PMc,dim,Xb):
    for j in range(int(PMc)):
        for i in range(dim):
            AI = rng.normal(loc=0, scale=1.2, size=1)
            X[j, i] = X[j, i] + (Xb[i] - X[j, i]) * AI
    return X

# Foraging flamingo position update function.
def untrammeled(X, Xb, PMc, PMu, dim,):
    for j in range(int(PMc), int(PMc+PMu)):
        for i in range(dim):
            X[j, i] = (X[j, i] + rand_1() * Xb[i] + np.random.randn() * (np.random.randn() * np.abs(Xb[i] + rand_1() * X[j, i]))) / (rng.chisquare(df=8, size=1))
    return X

# The second stage migratory flamingo position update function.
def flee(X, PMc, PMu, pop, dim, Xb):
    for j in range(int(PMc+PMu), pop):
        for i in range(dim):
            A1 = rng.normal(loc=0, scale=1.2, size=1)
            X[j, i] = X[j, i]+(Xb[i]-X[j, i])*A1
    return X


def MSA(pop,dim,lb,ub,Max_iter,fun,MP_b):
    X = initial(pop, dim, lb,ub)                    # Initialize the flamingo population.
    fitness = CaculateFitness(X, fun)               # Calculate fitness values for each flamingo.
    fitness, sortIndex = SortFitness(fitness)       # Sort the fitness values of flamingos.
    X = SortPosition(X, sortIndex)                  # Sort the flamingos.
    GbestScore = fitness[0]                         # The optimal value for the current iteration.
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = X[0, :]
    Curve = np.zeros([Max_iter, 1])
    for i in range(Max_iter):
        Vs=random.random()
        PMf=int((1-MP_b)*Vs*pop)                     # The number of flamingos migrating in the second stage.
        PMc=MP_b*pop                                 # The number of flamingos that migrate in the first phase.
        Pmu=pop-PMc-PMf                              # The number of flamingos foraging for food.
        Xb = X[0, :]

        # In the first stage of migration, flamingos undergo location updates.
        X = congeal(X, PMc, dim, Xb)

        # The foraging flamingos update their position.
        X = untrammeled(X, Xb, PMc, Pmu, dim)

        # In the second stage, the flamingos were relocated for location renewal.
        X = flee(X, PMc, Pmu, pop, dim, Xb)

        X = BorderCheck(X, lb, ub, pop, dim)                    # Boundary detection.
        fitness = CaculateFitness(X, fun)                       # Calculate fitness values.
        fitness, sortIndex = SortFitness(fitness)               # Sort fitness values.
        X = SortPosition(X, sortIndex)                          # Sort the locations according to fitness.
        if (fitness[0] <= GbestScore):                          # Update the global optimal solution.
            GbestScore = fitness[0]
            GbestPositon[0, :] = X[0, :]
        Curve[i] = GbestScore
    return GbestScore,GbestPositon,Curve



'''The main function '''
                            # Set relevant parameters.
time_start = time.time()
pop = 50                    # Flamingo population size.
MaxIter = 300               # Maximum number of iterations.
dim = 20                    # The dimension.
fl=-100                     # The lower bound of the search interval.
ul=100                      # The upper bound of the search interval.
lb = fl*np.ones([dim, 1])
ub = ul*np.ones([dim, 1])
MP_b=0.1                    # The basic proportion of flamingos migration in the first stage.
GbestScore, GbestPositon, Curve = MSA(pop, dim, lb, ub, MaxIter, fun, MP_b)
time_end = time.time()
print(f"The running time is: {time_end  - time_start } s")
print('The optimal value：',GbestScore)
print('The optimal solution：',GbestPositon)



