import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

models = ["Aes-V2.5\n(photo)", "Q-Align\n(photo)", "ArtiMuse\n(multi)", "SAAN\n(art)"]
art = [0.009, 0.126, 0.199, 0.472]
colors = ["#d1495b","#d1495b","#e8a35c","#3a7ca5"]

fig, ax = plt.subplots(figsize=(6,4))
bars = ax.bar(models, art, color=colors)
for b,v in zip(bars,art):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=10)
ax.set_ylabel("SRCC on artistic domain (BAID)")
ax.set_title("Aesthetic competence on art by specialisation")
ax.set_ylim(0,0.55)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#d1495b",label="Photo-specialist"),
                   Patch(color="#e8a35c",label="Multi-domain"),
                   Patch(color="#3a7ca5",label="Art-specialist")], fontsize=8)
plt.tight_layout()
plt.savefig("results/fig3_grid.png", dpi=200, bbox_inches="tight")
plt.savefig("results/fig3_grid.pdf", bbox_inches="tight")
print("saved fig3")