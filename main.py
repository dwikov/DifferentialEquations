import math
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
            u += [u[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4)/6]
        return x, u

    def exact_solution(self):
        x = [self.a]
        u = [self.c]
        for i in range(1,self.n):
            x += [x[i - 1] + self.h]
            u += [-2/x[i] + 4*(x[i]**3)/(x[i]**4 + self.c1)]
        return x, u

    def error_function(self, exact, approx):
        return exact[0], [abs(exact[1][i]-approx[1][i]) for i in range(len(exact[1]))]

    def error_analysis(self, exact, approx, n0, N):
        x = []
        u = []
        for i in range(n0,N+1):
            x += [i]
            self.n = i
            self.h = (self.b - self.a) / self.n
            u += [max(self.error_function(exact(),approx())[1])]
        return x, u

class Plotter():
    def __init__(self, n,a,b,c):
        self.DE = DifferentialEquation(n, a, b, c)

    def plot_methods(self):
        plt.plot(self.DE.euler_method()[0], self.DE.euler_method()[1], label="Euler method")
        plt.plot(self.DE.improved_euler_method()[0], self.DE.improved_euler_method()[1], label="Improved Euler method")
        plt.plot(self.DE.runge_kutta_method()[0], self.DE.runge_kutta_method()[1], label="Runge Kutta method")
        plt.plot(self.DE.exact_solution()[0], self.DE.exact_solution()[1], label="Exact solution")
        plt.legend()
        plt.show()

    def plot_error(self):
        plt.plot(self.DE.error_function(self.DE.exact_solution(),self.DE.euler_method())[0],
                 self.DE.error_function(self.DE.exact_solution(),self.DE.euler_method())[1], label="Euler method")
        plt.plot(self.DE.error_function(self.DE.exact_solution(), self.DE.improved_euler_method())[0],
                 self.DE.error_function(self.DE.exact_solution(), self.DE.improved_euler_method())[1],label="Improved Euler method")
        plt.plot(self.DE.error_function(self.DE.exact_solution(), self.DE.runge_kutta_method())[0],
                 self.DE.error_function(self.DE.exact_solution(), self.DE.runge_kutta_method())[1],label="Runge Kutta method")
        plt.legend()
        plt.show()

    def plot_error_analysis(self,n0,N):
        plt.plot(self.DE.error_analysis(self.DE.exact_solution,self.DE.euler_method,n0,N),label="Euler method")
        plt.plot(self.DE.error_analysis(self.DE.exact_solution,self.DE.improved_euler_method,n0,N),label="Improved Euler method")
        plt.plot(self.DE.error_analysis(self.DE.exact_solution,self.DE.runge_kutta_method,n0,N),label="Runge Kutta method")
        plt.legend()
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
        self.n.insert(0,'20')
        self.label1.grid(row=0, pady=20, padx=20)
        self.n.grid(row=0, column=1, padx=20)

        self.label2 = tk.Label(self, text="x0")
        self.x0 = tk.Entry(self, width=20)
        self.x0.insert(0,'1')
        self.label2.grid(row=1, pady=20)
        self.x0.grid(row=1, column=1)

        self.label3 = tk.Label(self, text="X")
        self.X = tk.Entry(self, width=20)
        self.X.insert(0,'7')
        self.label3.grid(row=2, pady=20)
        self.X.grid(row=2, column=1)

        self.label4 = tk.Label(self, text="y0")
        self.y0 = tk.Entry(self, width=20)
        self.y0.insert(0,'0')
        self.label4.grid(row=3, pady=20)
        self.y0.grid(row=3, column=1)

        self.plot_button = tk.Button(self, text="Plot methods", command=self.plot_methods)
        self.plot_button.grid(row=4, column=1)

        self.plot_button = tk.Button(self, text="Plot error", command=self.plot_errors)
        self.plot_button.grid(row=5, column=1)

        self.label5 = tk.Label(self, text="n0")
        self.n0 = tk.Entry(self, width=20)
        self.label5.grid(row=6, pady=20, padx=20)
        self.n0.grid(row=6, column=1, padx=20)

        self.label6 = tk.Label(self, text="N")
        self.N = tk.Entry(self, width=20)
        self.label6.grid(row=7, pady=20, padx=20)
        self.N.grid(row=7, column=1, padx=20)

        self.plot_button = tk.Button(self, text="Error analysis", command=self.plot_analysis)
        self.plot_button.grid(row=8, column=1)


    def plot_methods(self):
        self.plotter = Plotter(int(self.n.get()), int(self.x0.get()), int(self.X.get()), int(self.y0.get()))
        self.plotter.plot_methods()

    def plot_errors(self):
        self.plotter = Plotter(int(self.n.get()), int(self.x0.get()), int(self.X.get()), int(self.y0.get()))
        self.plotter.plot_error()

    def plot_analysis(self):
        self.plotter = Plotter(int(self.n.get()), int(self.x0.get()), int(self.X.get()), int(self.y0.get()))
        self.plotter.plot_error_analysis(int(self.n0.get()),int(self.N.get()))

if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
