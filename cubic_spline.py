import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt


def cubic_spline(x, y):


    n = len(x) - 1  
    h = np.diff(x)  

    a = y.copy()
    b = np.zeros(n)
    c = np.zeros(n + 1)
    d = np.zeros(n)

    A = np.zeros((n + 1, n + 1))
    r = np.zeros(n + 1)

    A[0, 0] = 1
    A[n, n] = 1

    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        r[i] = 3 * ((a[i+1] - a[i]) / h[i] - (a[i] - a[i-1]) / h[i-1])

    c = np.linalg.solve(A, r)

    for i in range(n):
        b[i] = (a[i+1] - a[i]) / h[i] - h[i] * (2*c[i] + c[i+1]) / 3
        d[i] = (c[i+1] - c[i]) / (3 * h[i])

    splines = []
    for i in range(n):
        spline = (a[i], b[i], c[i], d[i])
        splines.append(spline)

    return splines
def calculate_spline(splines, x_vals, x):
    y_vals = np.zeros_like(x_vals)
    i = 0
    while i < len(splines):
        a, b, c, d = splines[i]
        x_i, x_next = x[i], x[i + 1]
        index = (x_vals >= x_i) & (x_vals <= x_next)
        dx = x_vals[index] - x_i
        y_vals[index] = a + b * dx + c * dx**2 + d * dx**3
        i += 1
    return y_vals


def draw(x, y):

    splines = cubic_spline(x,y)

    x_smooth_values = np.linspace(x[0],x[-1],10000)
    y_smooth_values = calculate_spline(splines,x_smooth_values,x)

    print(y_smooth_values)


    plt.plot(x,y,"o")
    plt.plot(x_smooth_values,y_smooth_values,"red")

    plt.show()
    return y_smooth_values

class CubicSplineApp:
    def __init__(self, root):
        self.root = root

        self.x = [0]
        self.y = [0]
        self.smooth = []

        self.create_widgets()
        self.update_plot()


    def create_widgets(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X)

        tk.Label(control_frame, text="X:").pack(side=tk.LEFT)
        self.x_entry = tk.Entry(control_frame)
        self.x_entry.pack(side=tk.LEFT)

        tk.Label(control_frame, text="Y:").pack(side=tk.LEFT)
        self.y_entry = tk.Entry(control_frame)
        self.y_entry.pack(side=tk.LEFT)

        add_button = tk.Button(control_frame, text="Add Point", command=self.add_point)
        add_button.pack(side=tk.LEFT)

        remove_button = tk.Button(control_frame, text="Remove Point", command=self.remove_point)
        remove_button.pack(side=tk.LEFT)

        get_value_button = tk.Button(control_frame, text = "Get value", command=self.get_value)
        get_value_button.pack(side=tk.LEFT)

        derivative_button = tk.Button(control_frame, text = "Calculate first derivative", command=self.calculate_derivative)
        derivative_button.pack(side=tk.LEFT)

        derivative_button2 = tk.Button(control_frame, text = "Calculate second derivative", command=self.calculate_derivative2)
        derivative_button2.pack(side=tk.LEFT)

        self.status_label = tk.Label(control_frame, text="")
        self.status_label.pack(side=tk.LEFT, padx=15)


    def update_plot(self):
        plt.clf()
        self.smooth = draw(self.x, self.y)

    def add_point(self):
        try:

            new_x = float(self.x_entry.get())
            new_y = float(self.y_entry.get())

            if (new_x, new_y) in zip(self.x, self.y):
                self.status_label.config(text=f"Point ({new_x}, {new_y}) already exists.")

            elif new_x in self.x:
                self.status_label.config(text=f"A point at x: {new_x} already exists. Try a different point")

            else:
                self.x.append(new_x)
                self.y.append(new_y)

                sorted_pairs = sorted(zip(self.x, self.y))
                self.x, self.y = zip(*sorted_pairs)
                self.x, self.y = list(self.x), list(self.y)

                self.status_label.config(text=f"Added point ({new_x}, {new_y})")
                self.update_plot()
        except ValueError:
            self.status_label.config(text="Invalid input")


    def remove_point(self):
        try:
            remove_x = float(self.x_entry.get())
            remove_y = float(self.y_entry.get())

            if (remove_x, remove_y) in zip(self.x, self.y):
                index = (self.x.index(remove_x), self.y.index(remove_y))
                removed_point = (self.x.pop(index[0]), self.y.pop(index[1]))
                self.status_label.config(text=f"Removed point {removed_point}")
                self.update_plot()
            else:
                self.status_label.config(text="Point not found")
        except ValueError:
            self.status_label.config(text="Invalid input")

    def calculate_derivative(self):
        try:
            point = float(self.x_entry.get())
            splines = cubic_spline(self.x, self.y)

            for i in range(len(self.x) - 1):
                if self.x[i] <= point <= self.x[i + 1]:
                    _, b, c, d = splines[i]
                    dx = point - self.x[i]
                    deriv = b + 2 * c * dx + 3 * d * dx**2
                    self.status_label.config(text=f"First derivative at x={point}: {deriv:.2f}")
                    return

            self.status_label.config(text="x out of range.")
        except ValueError:
            self.status_label.config(text="Invalid input.")

    def calculate_derivative2(self):
        try:
            point = float(self.x_entry.get())
            splines = cubic_spline(self.x, self.y)

            for i in range(len(self.x) - 1):
                if self.x[i] <= point <= self.x[i + 1]:
                    _, _, c, d = splines[i]
                    dx = point - self.x[i]
                    deriv2 = 2 * c + 6 * d * dx
                    self.status_label.config(text=f"Second derivative at x={point}: {deriv2:.2f}")
                    return

            self.status_label.config(text="x out of range.")
        except ValueError:
            self.status_label.config(text="Invalid input.")


    def get_value(self):
        try:
            point = float(self.x_entry.get())

            t = int(round(abs(point/(self.x[-1] - self.x[0])*10000)))
            point_value = self.smooth[t-1]


            self.status_label.config(text=f"Value at x={point}: {point_value}")
        except ValueError:
            self.status_label.config(text="Invalid input.")
        except ZeroDivisionError:
            self.status_label.config(text="Invalid input.")
        except IndexError:
            self.status_label.config(text="x out of range")


if __name__ == "__main__":
    root = tk.Tk()
    app = CubicSplineApp(root)
    root.mainloop()
