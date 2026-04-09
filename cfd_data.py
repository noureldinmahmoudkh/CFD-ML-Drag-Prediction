import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Renolds number range
Re=np.linspace(10,200,50)
#Simple empirical-like relation for drag coefficient
Cd=24/Re+6/(1+np.sqrt(Re))+0.4
noise=np.random.normal(0,0.05,size=Cd.shape)
Cd_noisy=Cd+noise
# Create dataset
data=pd.DataFrame({
    "Reynolds_Number":Re,
    "Drag_Coefficient":Cd_noisy
})
#Save to CSV
data.to_csv("cfd_dataset.csv",index=False)
#Show first rows
print(data.head())
#plot
# plt.figure()
# plt.plot(Re,Cd_noisy)
# plt.xlabel("Reynolds Number (Re)")
# plt.ylabel("Noisy_Drag_Coefficient (Cd_noisy)")
# plt.title("Cd vs Reynolds Number (Cylinder Flow)")
# plt.grid()
# plt.show()