

import random
import argparse
import codecs
import os
import numpy as np

# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""

        transitions = {}
        emissions = {}

        # Read transition probabilities from the .trans file
        with open(f'{basename}.trans', 'r') as trans_file:
            for line in trans_file:
                parts = line.strip().split()
                state_from, state_to, probability = parts
                transitions[state_from] = transitions.get(state_from, {})
                transitions[state_from][state_to] = float(probability)

        # Read emission probabilities from the .emit file
        with open(f'{basename}.emit', 'r') as emit_file:
            for line in emit_file:
                parts = line.strip().split()
                state, output, probability = parts
                emissions[state] = emissions.get(state, {})
                emissions[state][output] = float(probability)

        self.transitions = transitions
        self.emissions = emissions

    ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        if '#' not in self.transitions:
            raise ValueError("Initial state ('#') not found in transitions.")

        observation_states = []
        observation_outputs = []

        current_state = '#'

        for _ in range(n):
            # Possible next states and their probabilities
            next_states = list(self.transitions[current_state].keys())
            probabilities = list(self.transitions[current_state].values())

            # Choosing the next state based on the probabilities
            next_state = np.random.choice(next_states, p=probabilities)
            observation_states.append(next_state)

            # Possible emissions from the next state and their probabilities
            emissions = list(self.emissions[next_state].keys())
            emission_probabilities = list(self.emissions[next_state].values())

            # Choosing an emission based on the probabilities
            emission = np.random.choice(emissions, p=emission_probabilities)
            observation_outputs.append(emission)

            current_state = next_state

        return Observation(observation_states, observation_outputs)
        # return observation_outputs

    def forward(self, observations):
        # Create a t+1 x s matrix M
        # t is the number of timesteps/observations
        # s is the number of states

        t = len(observations)
        states = list(self.transitions.keys())
        s = len(states)

        # Create a dictionary to map state names to their numerical indices.
        state_to_index = {state: i for i, state in enumerate(states)}

        # Initialize the forward matrix M with zeros using numpy
        M = np.zeros((t + 1, s))

        # Set up the initial probabilities - Initialize the start state '#' with probability 1.
        M[0][state_to_index['#']] = 1.0

        # Iterate through each subsequent timestep for each state and propagate forward
        # multiply the probability of reaching that state from
        # any prior state by the probability of seeing this observation given that state.
        for i in range(t):
            for s1 in states:
                state_to_idx = state_to_index[s1]
                # Calculate the forward probability for each state at time step i+1.
                for s2 in states:
                    state_from_idx = state_to_index[s2]

                    transition_prob = self.transitions[s2].get(s1, 0.0) # T[s2,s]

                    # for initial
                    if i == 0 and s2 == '#':
                        emission_prob = 1.0
                    else:
                        emission_prob = self.emissions.get(s2, {}).get(observations[i], 0.0) # E[O[i],s2]

                    # PSEUDO: sum += M[s2, i-1]*T[s2,s]*E[O[i],s2]
                    M[i + 1][state_to_idx] += M[i][state_from_idx] * transition_prob * emission_prob

        # Calculate the final state with the highest probability
        # print(M[t])
        final_state_idx = np.argmax(M[t])
        final_state = states[final_state_idx]

        return final_state

    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        t = len(observation)
        states = list(self.transitions.keys())
        s = len(states)

        state_to_index = {state: i for i, state in enumerate(states)}

        # Viterbi matrix
        V = np.zeros((t, s))
        backpointer = np.zeros((t, s), dtype=int)

        # the first column of V based on the initial probabilities
        for s1 in states:
            state_to_idx = state_to_index[s1]
            emission_prob = self.emissions.get(s1, {}).get(observation[0], 0.0)
            V[0][state_to_idx] = 1.0 * emission_prob  # Initialize with the emission probability

        # fill in the viterbi and backpointer matrix
        for i in range(1, t):
            for s1 in states:
                state_to_idx = state_to_index[s1]
                max_probability = -1.0
                max_state = ""

                for s2 in states:
                    state_from_index = state_to_index[s2]
                    transition_prob = self.transitions[s2].get(s1, 0.0)
                    emission_prob = self.emissions.get(s1, {}).get(observation[i], 0.0)

                    prob = V[i - 1][state_from_index] * transition_prob * emission_prob

                    if prob > max_probability:
                        max_probability = prob
                        max_state = s2

                V[i][state_to_idx] = max_probability
                backpointer[i][state_to_idx] = state_to_index[max_state]

        # Find the highest probability state
        final_state_idx = np.argmax(V[-1])
        final_state = states[final_state_idx]

        state_sequence = [final_state]
        current_state_idx = final_state_idx

        for i in range(t - 1, 0, -1):
            current_state_idx = backpointer[i][current_state_idx]
            state_sequence.insert(0, states[current_state_idx])

        return state_sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hidden Markov Model (HMM)")
    parser.add_argument("model_file", help="Path to the model file (without extension)")
    parser.add_argument("--generate", type=int, help="Generate a random observation of the specified length")
    parser.add_argument("--forward", type=str, help="Calculate the forward probability of a sequence of observations")
    parser.add_argument("--viterbi", type=str, help="Use the Viterbi algorithm to find the most likely state sequence")

    args = parser.parse_args()

    model = HMM()

    # PART 1 ---------------------------------------------
    # Load transition and emission probabilities from 'two_english'
    # model.load('two_english')

    # # Print the loaded transition probabilities
    # print("Transition Probabilities:")
    # print(model.transitions)
    # for state_from, transitions in model.transitions.items():
    #     for state_to, probability in transitions.items():
    #         print(f"Transition from {state_from} to {state_to}: {probability}")

    # Print the loaded emission probabilities
    # print("\nEmission Probabilities:")
    # print(model.emissions)
    # for state, emissions in model.emissions.items():
    #     for output, probability in emissions.items():
    #         print(f"Emission from state {state} of output {output}: {probability}")

    # PART 2 ---------------------------------------------
    # Load transition and emission probabilities from the specified model file
    model.load(args.model_file)

    if args.generate:
        n = args.generate  # Length of the observation
        generated_observation = model.generate(n)
        print("Generated Observation:")
        print(generated_observation)
        # Was able to run with "python3 hmm.py partofspeech.browntags.trained --generate 20"

    ## PART 3 foward()
    if args.forward:
        observation_sequence = args.forward.split()  # Convert observation string to a list
        final_state = model.forward(observation_sequence)
        print("Most likely final state:", final_state)

    if args.viterbi:
        observation_sequence = args.viterbi.split()
        state_sequence = model.viterbi(observation_sequence)
        print("VITERBI: Most likely state sequence:", ' '.join(state_sequence))
        # Was able to run with "python3 hmm.py partofspeech.browntags.trained --viterbi ambiguous_sents.obs"