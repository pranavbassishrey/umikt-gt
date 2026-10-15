import pandas as pd

d = pd.read_csv("data/foundational_assist/train.csv")

print("Correctness distribution:")
print(d["correctness"].value_counts(dropna=False, normalize=True))

print("\nCorrectness by concept:")
print(d.groupby("concept_id")["correctness"].agg(["count", "mean"]))

print("\nLearner interaction counts:")
print(d.groupby("learner_id").size().describe())

print("\nConcept names:")
print(d.groupby("concept_id")["concept_name"].first())
