from hg_boltzmann import HGRBM
from matplotlib import pyplot as plt

rbm = HGRBM(visible_size=16)
energy = []
for i in range(10):
    energy.append(rbm.energy().numpy())
    rbm.inward()
    rbm.outward()
plt.scatter(range(10), energy)
