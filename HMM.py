

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



    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hidden Markov Model (HMM) Example")
    parser.add_argument("model_file", help="Path to the model file (without extension)")
    parser.add_argument("--generate", type=int, help="Generate a random observation of the specified length")

    args = parser.parse_args()

    model = HMM()

    # PART 1 ---------------------------------------------
    # Load transition and emission probabilities from 'two_english'
    model.load('two_english')

    # # Print the loaded transition probabilities
    print("Transition Probabilities:")
    print(model.transitions)
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

    if args.generate is not None:
        n = args.generate  # Length of the observation
        generated_observation = model.generate(n)
        print("Generated Observation:")
        print(generated_observation)
        # Was able to run with "python3 hmm.py partofspeech.browntags.trained --generate 20"



