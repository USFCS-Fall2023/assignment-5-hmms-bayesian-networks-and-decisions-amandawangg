from HMM import *
from alarm import *
from carnet import *

if __name__ == "__main__":
    ## Problem 2: Hidden Markov Models
    model = HMM()

    # PART 1 ---------------------------------------------
    print("\nProblem 2 PART 1\n")
    model = HMM()

    # Load transition and emission probabilities from 'two_english'
    model.load('two_english')

    # Print the loaded transition probabilities
    print("Transition Probabilities:")
    print(model.transitions)
    # Testing purposes
    # for state_from, transitions in model.transitions.items():
    #     for state_to, probability in transitions.items():
    #         print(f"Transition from {state_from} to {state_to}: {probability}")

    # Print the loaded emission probabilities
    print("\nEmission Probabilities:")
    print(model.emissions)
    # Testing purposes
    # for state, emissions in model.emissions.items():
    #     for output, probability in emissions.items():
    #         print(f"Emission from state {state} of output {output}: {probability}")

    # PART 2 ---------------------------------------------
    print("\nProblem 2 PART 2\n")
    print("Was able to run with 'python3 hmm.py partofspeech.browntags.trained --generate 20' from terminal")
    # Example output I got when running it:
    # Generated Observation:
    # DET ADJ NOUN . NOUN ADV . DET NOUN NOUN VERB PRT VERB ADP ADJ CONJ VERB ADP DET ADJ
    # the personal points , president not -- the compass green moves to had that particular and represents in the poetic

    ## PART 3 foward()
    print("\nProblem 2 PART 3 Forward\n")
    print("Was able to run with 'python3 hmm.py partofspeech.browntags.trained --forward ambiguous_sents.obs' from terminal")
    # Example output I got when running it:
    # Most likely final state: DET

    ## PART 4 viterbi()
    print("\nProblem 2 PART 4 Viterbi\n")
    print("Was able to run with 'python3 hmm.py partofspeech.browntags.trained --viterbi ambiguous_sents.obs' from terminal")
    # Example output I got when running it:
    # Most likely final state: ADV

    print("---------------------------------------------------------------------------------------------------------------")

    ## Problem 3: Belief networks
    ## PART 1: alarm.py
    print("Problem 3 PART 1\n")
    q1 = alarm_infer.query(variables=["MaryCalls"], evidence={"JohnCalls": "yes"})
    print(q1)
    print("Problem 3 PART 1a: Probability of Mary Calling given that John called: 0.1002\n")

    q2 = alarm_infer.query(variables=["JohnCalls", "MaryCalls"], evidence={"Alarm": "yes"})
    print(q2)
    print("Problem 3 PART 1b: Probability of both John and Mary calling given Alarm: 0.0950\n")

    q3 = alarm_infer.query(variables=["Alarm"], evidence={"MaryCalls": "yes"})
    print(q3)
    print("Problem 3 PART 1c: Probability of Alarm, given that Mary called: 0.9826\n")
    print("---------------------------------------------------------------------------------------------------------------")

    ## PART 2: carnet.py
    print("Problem 3 PART 2\n")
    q1 = car_infer.query(variables=["Battery"], evidence={"Moves": "no"})
    print(q1)
    print("Problem 3 PART 2a: Given that the car will not move, the probability that the battery is not working: 0.3590\n")

    q2 = q = car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})
    print(q2)
    print("Problem 3 PART 2b: Given that the radio is not working, the probability that the car will not start: 0.8687\n")

    q3_a = q1 = car_infer.query(variables=["Radio"], evidence={"Battery": "Works"})
    print(q3_a)
    q3_b = q1 = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"})
    print(q3_b)
    print("Problem 3 PART 2c: Given that the battery is working, the probability of the radio working does not change if we discover that the car has gas in it."
        "The probability for it is 0.7500\n")

    q4_a = car_infer.query(variables=["Ignition"], evidence={"Moves": "no"})
    print(q4_a)
    q4_b = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})
    print(q4_b)
    print("Problem 3 PART 2d: Given that the car doesn't move, the probability of the ignition failing is lower if the car does not have gas in it.\n")

    q5 = car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"})
    print(q5)
    print("Problem 3 PART 2e: The probability that the car starts if the radio works and it has gas in it: 0.7212\n")
    print("---------------------------------------------------------------------------------------------------------------")

    ## PART 3: CPD Key Presents -> see carnet.py -> cpd_key_present and cpd_starts_key_present
    print("Problem 3 PART 3\n")
    q_key_present = car_infer.query(variables=["KeyPresent"])
    print(q_key_present)
    print("Problem 3 PART 3: Key present(yes): 0.7\n")