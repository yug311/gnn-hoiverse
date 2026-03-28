import pickle
from pprint import pprint
import json

# This script loads the original data and views it

print("Loading data from hoiverse.pkl...")


with open("hoiverse.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())  # should show: dict_keys(['data'])

dataset = data["data"]

print("Type of dataset:", type(dataset))
print("Number of entries:", len(dataset))

example = data["data"][0]
print(example.keys())




print(dataset[0]["annotations"][0]["bbox"])
print(dataset[0]["annotations"][0]["category_id"])

node_names = data["node_names"]
rel_names = data["rel_names"]

anns = example["annotations"]
rels = example["relations"]


for s, o, r in rels[:50]:  # print the first 5 relations
    print("Subject:", node_names[anns[s]["category_id"]], anns[s]["bbox"])
    print("Object: ", node_names[anns[o]["category_id"]], anns[o]["bbox"])
    print("Relation:", rel_names[r])
    print()

print(data["node_names"])
print(data["rel_names"])

sample = dataset[:5]  # get the first 5 entries for inspection
print()
# Save to JSON (readable)
with open("sample.json", "w") as f:
    json.dump(sample, f, indent=2)



print("\nFinished loading and displaying data.")