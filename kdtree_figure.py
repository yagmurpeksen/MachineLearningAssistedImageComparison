import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import KDTree

np.random.seed(42)

# Simulate star positions (epoch 2 unmatched stars)
stars = np.random.uniform(20, 280, (40, 2))

# Query point: predicted position of an epoch-1 star
query = np.array([150, 150])
radius = 40.0

# KDTree query
tree = KDTree(stars)
candidates = tree.query_ball_point(query, radius)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_facecolor("#f9f9f9")

# All stars
ax.scatter(stars[:, 0], stars[:, 1], c="steelblue", s=40, zorder=3, label="Unmatched stars (epoch 2)")

# Candidate stars inside radius
if candidates:
    ax.scatter(stars[candidates, 0], stars[candidates, 1],
               c="crimson", s=60, zorder=4, label="Candidates within radius")

# Query point
ax.scatter(*query, c="black", s=100, marker="x", zorder=5, linewidths=2, label="Predicted position (epoch 1 star)")

# Search radius circle
circle = plt.Circle(query, radius, color="black", fill=False, linestyle="--", linewidth=1.5, label=f"Search radius = {int(radius)} px")
ax.add_patch(circle)

ax.set_xlim(0, 300)
ax.set_ylim(0, 300)
ax.set_xlabel("x (pixels)", fontsize=12)
ax.set_ylabel("y (pixels)", fontsize=12)
ax.set_title("KD-Tree Ball-Point Query for Moving Object Detection", fontsize=12)
ax.legend(loc="upper right", fontsize=9)
ax.set_aspect("equal")
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("kdtree_query.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: kdtree_query.png")
