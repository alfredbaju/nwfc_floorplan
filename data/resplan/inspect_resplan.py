import pickle

with open("ResPlan.pkl", "rb") as f:
    data = pickle.load(f)

print("Top-level type:", type(data))

if isinstance(data, dict):
    print("Number of plans:", len(data))
    first_key = next(iter(data))
    first_plan = data[first_key]
elif isinstance(data, list):
    print("Number of plans:", len(data))
    first_plan = data[0]
else:
    first_plan = data

print("\nFirst plan type:", type(first_plan))
if isinstance(first_plan, dict):
    print("First plan keys:", list(first_plan.keys()))
    if "graph" in first_plan:
        g = first_plan["graph"]
        print("\nGraph type:", type(g))
        print("Graph nodes (sample):", list(g.nodes(data=True))[:3])
        print("Graph edges (sample):", list(g.edges(data=True))[:3])