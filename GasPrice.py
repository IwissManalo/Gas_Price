import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed()
days = np.arange(1, 101)
initial_prices = np.array([64.15, 53.17, 45.13])  # Initial value for each gas price
gas_prices = np.zeros((3, 100))
gas_prices[:, 0] = initial_prices

for i in range(3):
    for day in range(1, 100):
        gas_prices[i, day] = gas_prices[i, day - 1] + np.random.uniform(-2, 2)  # Random fluctuations between -2 and 2


daily_changes = np.diff(gas_prices[0])

print("----------------------------------\n   Quantitative Data - Midterms \n\tBy Althea Irish Manalo \n----------------------------------")

# Best Day
best_day = np.argmax(daily_changes < 0) + 1
print("Best Day to Fill Up Gas: Day", best_day)

# Probability of reduction per day
deduction_probability = np.mean(daily_changes < 0)
print("Probability of Price Reduction per Day:", deduction_probability)

# Worst day
worst_day = np.argmax(daily_changes) + 1
print("Worst Day to Fill Up Gas: Day", worst_day)

plt.figure(figsize=(10, 6))
plt.title('Quantitative Data - Midterms', fontsize=16, pad=20, color='black')
plt.scatter(days, gas_prices[0], color='blue', zorder=2)

model = LinearRegression()
model.fit(days.reshape(-1, 1), gas_prices[0].reshape(-1, 1))

predicted_prices = model.predict(days.reshape(-1, 1))

plt.plot(days, predicted_prices, color='red', linewidth=2, zorder=3)

plt.text(0.5, 1.02, "By Althea Irish Manalo", fontsize=10, ha='center', color='black', transform=plt.gca().transAxes, fontdict={'color': 'red', 'weight': 'bold'})
plt.xlabel('Days', fontsize=14, color='black', fontdict={'color': 'red', 'weight': 'bold'})
plt.ylabel('Gas Price', fontsize=14, color='black', fontdict={'color': 'red', 'weight': 'bold'})
plt.grid(True)
plt.xticks(fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')
plt.show()
