import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from matplotlib.figure import Figure

class DifferentialEquation:
    def __init__(self, n, a, b, c):
        self.n = n
        self.a = a
        self.b = b
        self.c = c
        self.h = (self.b - self.a) / self.n
        self.c1 = 4*(self.a**3)/(self.c+2/self.a) - self.a**4

    def f(self, x, y):
        return 4 / (x * x) - y / x - y * y

    def euler_method(self):
        x = [self.a]
        u = [self.c]
        for i in range(1, self.n):
            x += [x[i - 1] + self.h]
            d = self.h * self.f(x[i - 1], u[i - 1])
            u += [u[i - 1] + d]
        return x, u

    def improved_euler_method(self):
        x = [self.a]
        u = [self.c]
        for i in range(1, self.n):
            x += [x[i - 1] + self.h]
            k1 = self.f(x[i - 1], u[i - 1])
            k2 = self.f(x[i], u[i - 1] + self.h * k1)
            u += [u[i - 1] + self.h * (k1 + k2) / 2]
        return x, u

    def runge_kutta_method(self):
        x = [self.a]
        u = [self.c]
        for i in range(1, self.n):
            x += [x[i - 1] + self.h]
            k1 = self.h * self.f(x[i - 1], u[i - 1])
            k2 = self.h * self.f(x[i - 1] + self.h / 2, u[i - 1] + k1 / 2)
            k3 = self.h * self.f(x[i - 1] + self.h / 2, u[i - 1] + k2 / 2)
            k4 = self.h * self.f(x[i - 1] + self.h, u[i - 1] + k3)
            u += [u[i - 1] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)]
        return x, u

    def exact_solution(self):
        x = [self.a]
        u = [self.c]
        for i in range(1,self.n):
            x += [x[i - 1] + self.h]
            u += [-2/x[i-1] + 4*(x[i-1]**3)/(x[i-1]**4 + self.c1)]
        return x, u

class Plotter():
    def __init__(self, n,a,b,c):
        self.DE = DifferentialEquation(n, a, b, c)
        self.plot()

    def plot(self):
        plt.plot(self.DE.euler_method()[0], self.DE.euler_method()[1])
        plt.plot(self.DE.improved_euler_method()[0], self.DE.improved_euler_method()[1])
        plt.plot(self.DE.runge_kutta_method()[0], self.DE.runge_kutta_method()[1])
        plt.plot(self.DE.exact_solution()[0], self.DE.exact_solution()[1])
        plt.show()


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.label1 = tk.Label(self, text="N")
        self.n = tk.Entry(self, width=20)
        self.label1.grid(row=0, pady=20, padx=20)
        self.n.grid(row=0, column=1, padx=20)

        self.label2 = tk.Label(self, text="x0")
        self.x0 = tk.Entry(self, width=20)
        self.label2.grid(row=1, pady=20)
        self.x0.grid(row=1, column=1)

        self.label3 = tk.Label(self, text="X")
        self.X = tk.Entry(self, width=20)
        self.label3.grid(row=2, pady=20)
        self.X.grid(row=2, column=1)

        self.label4 = tk.Label(self, text="y0")
        self.y0 = tk.Entry(self, width=20)
        self.label4.grid(row=3, pady=20)
        self.y0.grid(row=3, column=1)

        self.plot_button = tk.Button(self, text="Plot", command=self.plot)
        self.plot_button.grid(row=4, column=1)

    def plot(self):
        self.plotter = Plotter(int(self.n.get()), int(self.x0.get()), int(self.X.get()), int(self.y0.get()))
        self.plotter.plot()


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
