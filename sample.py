#%%
import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k):
        # k: number of bandit arms
        self.k = k
        
        # qstar: action values
        self.qstar = np.random.normal(size=k)
    
    def action(self, a):
        return np.random.normal(loc=self.qstar[a])

def greedy_action_selection(k, numsteps):
    # k: number of bandit arms
    # numsteps: number of steps (repeated action selections)
    
    # Apossible[t]: list of possible actions at step t
    Apossible = {}
    
    # A[t]: action selected at step t
    A = np.zeros((numsteps,))
    
    # N[a,t]: the number of times action a was selected 
    #         in steps 0 through t-1
    N = np.zeros((k,numsteps+1))
    
    # R[t]: reward at step t
    R = np.zeros((numsteps,))
    
    # Q[a,t]: estimated value of action a at step t
    Q = np.zeros((k,(numsteps+1)))

    # Initialize bandit
    bandit = Bandit(k)

    for t in range(numsteps):

        # Select greedy actions as possible actions
        Apossible[t] = np.argwhere(Q[:,t] == np.amax(Q[:,t])).flatten()

        # Select action randomly from possible actions
        a = Apossible[t][np.random.randint(len(Apossible[t]))]

        # Record action taken
        A[t] = a

        # Perform action (= sample reward)
        R[t] = bandit.action(a)

        # Update action counts
        N[:,t+1] = N[:,t]
        N[a,t+1] += 1

        # Update action value estimates, incrementally
        if N[a,t] > 0:
            Q[:,t+1] = Q[:,t]
            Q[a,t+1] = Q[a,t] + (R[t] - Q[a,t]) / N[a,t]
        else:
            Q[:,t+1] = Q[:,t]
            Q[a,t+1] = R[t]

    return {'bandit': bandit, 
            'Apossible': Apossible, 
            'A': A, 'N' : N, 'R' : R, 'Q' : Q}

def epsilon_greedy_action_selection(k, numsteps, epsilon):
    # k: number of bandit arms
    # numsteps: number of steps (repeated action selections)
    # epsilon: probability with which a random action is selected,
    #          as opposed to a greedy action

    # Apossible[t]: list of possible actions at step t
    Apossible = {}
    
    # A[t]: action selected at step t
    A = np.zeros((numsteps,))
    
    # N[a,t]: the number of times action a was selected 
    #         in steps 0 through t-1
    N = np.zeros((k,numsteps+1))
    
    # R[t]: reward at step t
    R = np.zeros((numsteps,))
    
    # Q[a,t]: estimated value of action a at step t
    Q = np.zeros((k,(numsteps+1)))

    # Initialize bandit
    bandit = Bandit(k)

    for t in range(numsteps):
        if np.random.rand() < epsilon:
            # All actions are equally possible
            Apossible[t] = np.arange(k)
        else:
            # Select greedy actions as possible actions
            Apossible[t] = np.argwhere(Q[:,t] == np.amax(Q[:,t])).flatten()

        # Select action randomly from possible actions
        a = Apossible[t][np.random.randint(len(Apossible[t]))]

        # Record action taken
        A[t] = a

        # Perform action (= sample reward)
        R[t] = bandit.action(a)

        # Update action counts
        N[:,t+1] = N[:,t]
        N[a,t+1] += 1

        # Update action value estimates, incrementally
        if N[a,t] > 0:
            Q[:,t+1] = Q[:,t]
            Q[a,t+1] = Q[a,t] + (R[t] - Q[a,t]) / N[a,t]
        else:
            Q[:,t+1] = Q[:,t]
            Q[a,t+1] = R[t]

    return {'bandit' : bandit,
            'numsteps' : numsteps,
            'epsilon' : epsilon,
            'Apossible': Apossible, 
            'A': A, 'N' : N, 'R' : R, 'Q' : Q}


def plot_bandit_task(bandit_task):
    numsteps = bandit_task['numsteps']
    qdist = np.zeros((numsteps,))
    for t in range(numsteps):
        qdist[t] = np.mean(np.square(bandit_task['bandit'].qstar - bandit_task['Q'][:,t]))

    # Plot
    f, axarr = plt.subplots(4, figsize=(12,18))
    axarr[0].scatter(range(bandit_task['bandit'].k),bandit_task['bandit'].qstar)
    axarr[0].set_title('q*')
    for i,val in enumerate(bandit_task['bandit'].qstar):
        axarr[0].annotate("{0:.2f}".format(val), (i-0.25,val+0.15))

    axarr[1].set_title('Reward')
    axarr[1].plot(bandit_task['R'])
    axarr[1].set_xlim(xmin=-50,xmax=numsteps)

    axarr[2].set_title('Distance Q - q*')
    axarr[2].plot(qdist)
    axarr[2].set_xlim(xmin=-50,xmax=numsteps)

    axarr[3].set_title('N')
    axarr[3].scatter(range(bandit_task['bandit'].k), bandit_task['N'][:,numsteps])
    for i,val in enumerate(bandit_task['N'][:,numsteps]):
        axarr[3].annotate("{0:.0f}".format(val), (i-0.1,val+50))

def main():

    k = 10
    numsteps = 1000
    bandit_task = greedy_action_selection(k, numsteps)
    print(bandit_task['bandit'].qstar)    

    k = 10
    numsteps = 1000
    numtasks = 2000

    avgR = np.zeros((numsteps, ))
    for task in range(2000):
        bandit_task = greedy_action_selection(k,numsteps)
        avgR += bandit_task['R']
    avgR /= numtasks

    plt.plot(avgR) ;
    plt.ylabel('Average reward') ;
    plt.xlabel('Steps') ;
    plt.xlim(-5) ;

    k = 10
    numsteps = 1000
    epsilon = 0.1
    bandit_task = epsilon_greedy_action_selection(k, numsteps, epsilon)
    print('qstar:', bandit_task['bandit'].qstar)
    print('Q:',bandit_task['Q'][:,numsteps])

    k = 10
    numsteps = 5000
    epsilon = 0.1
    bandit_task = epsilon_greedy_action_selection(k, numsteps, epsilon)
    plot_bandit_task(bandit_task)

if __name__ == "__main__":

    main()

# %%
