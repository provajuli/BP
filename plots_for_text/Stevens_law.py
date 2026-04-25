import matplotlib.pyplot as plt
import numpy as np

GAMMA0 = 0.33
GAMMA2 = 3.5

def stevens_law(x, gamma):
    return x ** gamma

x = np.linspace(0, 2, 100)
y0 = stevens_law(x, GAMMA0)
y2 = stevens_law(x, GAMMA2) 
y3 = stevens_law(x, 1)  # Linear case for comparison
plt.figure(figsize=(8, 6))
plt.plot(x, y0, label=r'$\gamma = 0.33$, Jas', color='blue', linewidth=2)
plt.plot(x, y2, label=r'$\gamma = 3.5$, Elektrický šok', color='orange', linewidth=2)
plt.plot(x, y3, label=r'$\gamma = 1$, Délka úsečky', color='green', linestyle='--', linewidth=2)
plt.xlabel('Intenzita stimulu', fontsize=13)
plt.ylabel('Intenzita počitku', fontsize=13)
plt.legend()
plt.tight_layout()  # lepší spacing
plt.xlim(0, 2)
plt.ylim(0, 2)

plt.savefig('stevens_law.pdf', dpi=300, bbox_inches='tight')
