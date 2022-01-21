import json
import numpy as np
import matplotlib.pyplot as plt

perf_path = (
    "/home/maxime/dev-maxime/CLR/vm-data/negotiation_meta_game_payoffs.json"
)
exploit_path = "/home/maxime/dev-maxime/CLR/vm-data/negotiation_meta_game_exploitability.json"

with (open(perf_path, "rb")) as f:
    perf_content = json.load(f)
print("perf_content", perf_content)

with (open(exploit_path, "rb")) as f:
    exploit_content = json.load(f)
print("exploit_content", exploit_content)

exploit_order = [
    exploit_one_meta_solver[0]["meta_game"]
    for exploit_one_meta_solver in exploit_content
]

print("exploit_order", exploit_order)
exploit_order[5] = "PolicyGradient"

perf_order = [
    "alpha rank mixed on welfare sets",
    "alpha rank pure on welfare sets",
    "replicator dynamic init on welfare sets",
    "replicator dynamic random on welfare sets",
    "baseline random",
    "PG",
    "LOLA-Exact",
    "SOS-Exact",
    "(announcement+tau=0)",
]


perf = [np.array(perf_content[key]).mean() for key in perf_order]

exploit = [
    (
        +exploit_one_meta_solver[0]["pl_2_mean"]
        + exploit_one_meta_solver[1]["pl_1_mean"]
    )
    / 2
    for exploit_one_meta_solver in exploit_content
]


print("perf", perf)
print("exploit", exploit)

fig, ax = plt.subplots()

ax.scatter(perf, exploit)
plt.xlabel("mean payoff in cross play")
plt.ylabel("mean payoff while exploited")
plt.xlim((0.455, 0.466))
plt.ylim((0.442, 0.45))
for i, txt in enumerate(exploit_order):
    ax.annotate(txt, (perf[i], exploit[i]), rotation=10 * i)

plt.show()
