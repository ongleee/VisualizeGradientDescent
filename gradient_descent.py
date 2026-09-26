import numpy as np
import matplotlib.pyplot as plt

#function
def f(x):
  return x ** 2


#Derivative
def gradient(x):
  return 2 * x


x = 5

# Learning rate
learning_rate = 0.1

history = []


# Gradient Descent
for i in range(40):
  history.append(x)

  x = x - learning_rate * gradient(x)


print("Minimum x =", x)

# สร้างกราฟ function
x_values = np.linspace(-6, 6, 100)

plt.plot(
    x_values,
    f(x_values),
    label="f(x) = x²"
)

history = np.array(history) # จุดที่ Gradient Descent เดินผ่าน

plt.scatter(
    history,
    f(history),
    color="red",
    label="Gradient Descent"
)

# เส้นเชื่อมแต่ละจุด
plt.plot(
    history,
    f(history),
    color="red",
    linestyle="--"
)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gradient Descent")

plt.legend()
plt.grid()

plt.show()