from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("KeyPresent", "Starts"),
        ("Starts","Moves")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
    evidence=["Ignition", "Gas"],
    evidence_card=[2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)

## PART 3: CPD Key Presents
cpd_key_present = TabularCPD(
    variable="KeyPresent", variable_card=2, values=[[0.7], [0.3]],
    state_names={"KeyPresent": ['yes', 'no']},
)

cpd_starts_key_present = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[
        [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    ],
    evidence=["Gas", "Ignition", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={
        "Starts": ['yes', 'no'],
        "Gas": ['Full', 'Empty'],
        "Ignition": ["Works", "Doesn't work"],
        "KeyPresent": ['yes', 'no'],
    },
)

# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_key_present, cpd_starts_key_present)

# Add the edge from KeyPresent to Starts
car_model.add_edge("KeyPresent", "Starts")

car_infer = VariableElimination(car_model)

# print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))

q1 = car_infer.query(variables=["Battery"], evidence={"Moves": "no"})
print(q1)
print("Given that the car will not move, the probability that the battery is not working: 0.3590\n")

q2 = q = car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})
print(q2)
print("Given that the radio is not working, the probability that the car will not start: 0.8687\n")

q3_a = q1 = car_infer.query(variables=["Radio"], evidence={"Battery": "Works"})
print(q3_a)
q3_b = q1 = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"})
print(q3_b)
print("Given that the battery is working, the probability of the radio working does not change if we discover that the car has gas in it."
      "The probability for it is 0.7500\n")

q4_a = car_infer.query(variables=["Ignition"], evidence={"Moves": "no"})
print(q4_a)
q4_b = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})
print(q4_b)
print("Given that the car doesn't move, the probability of the ignition failing is lower if the car does not have gas in it.\n")

q5 = car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"})
print(q5)
print("The probability that the car starts if the radio works and it has gas in it: 0.7212\n")