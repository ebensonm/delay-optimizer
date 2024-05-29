import numpy as np

class ObjectiveFunction:
    def __init__(self, n, domain, minimizer):
        self.n = n
        self.domain = domain
        self.minimizer = minimizer

    def get_loss_function(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_grad_function(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def loss(self, x):
        return self.get_loss_function()(x)

    def grad(self, x):
        return self.grad_function()(x)
    
    def __call__(self, x):
        return self.loss(x)

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n}, domain={self.domain})"

    def __str__(self):
        return self.__class__.__name__


class Ackley(ObjectiveFunction):
    def __init__(self, n, domain=[-32.768, 32.768], a=20, b=0.2, c=2*np.pi):
        super().__init__(n, domain=domain, minimizer=np.zeros(n))
        self.a = a
        self.b = b
        self.c = c

    def get_loss_function(self):
        def ackley(x):
            sum_x2 = np.sum(np.square(x), axis=-1)
            part_1 = np.exp(-self.b * np.sqrt(sum_x2 / self.n))
            part_2 = np.exp(np.sum(np.cos(self.c * x), axis=-1) / self.n)
            return -self.a * part_1 - part_2 + self.a + np.e
        return ackley

    def get_grad_function(self):
        def ackley_grad(x):
            sum_x2 = np.sum(np.square(x), axis=-1)
            part_1a = self.a * self.b * x / np.sqrt(self.n * sum_x2)
            part_1b = np.exp(-self.b * np.sqrt(sum_x2 / self.n))
            part_2a = self.c * np.sin(self.c * x) / self.n
            part_2b = np.exp(np.sum(np.cos(self.c * x), axis=-1) / self.n)
            return part_1a * part1b + part_2a * part_2b
        return ackley_grad


class Rastrigin(ObjectiveFunction):
    def __init__(self, n, domain=[-5.12, 5.12]):
        super().__init__(n, domain=domain, minimizer=np.zeros(n))

    def get_loss_function(self):
        def rastrigin(x):
            summand = np.square(x) - 10 * np.cos(2*np.pi * x)
            return 10*self.n + np.sum(summand, axis=-1)
        return ackley

    def get_grad_function(self):
        def rastrigin_grad(x):
            return 2*x + 20*np.pi * np.sin(2*np.pi * x)
        return ackley_grad


class Rosenbrock(ObjectiveFunction):
    def __init__(self, n, domain=[-5., 10.], a=1, b=100):
        super().__init__(n, domain=domain, minimizer=np.ones(n))
        self.a = a
        self.b = b

    def get_loss_function(self):
        def rosenbrock(x):
            x0 = x[:-1] # Account for difference between i and i+1
            x1 = x[1:]
            summand_1 = self.b * np.square(x1 - np.square(x0))
            summand_2 = np.square(self.a - x0)
            return np.sum(summand_1 + summand_2, axis=-1)
        return rosenbrock

    def get_grad_function(self):
        def rosen_grad(x):
            grad = np.zeros(self.n)
            x0 = x[:-1]
            x1 = x[1:]

            part_1 = self.b * (x1 - np.square(x0))
            grad[:-1] += -4 * x0 * part_1 - 2*(self.a - x0)
            grad[1:] += 2 * part_1
            return grad
        return rosen_grad


class Zakharov(ObjectiveFunction):
    def __init__(self, n, domain=[-5., 10.]):
        super().__init__(n, domain=domain, minimizer=np.zeros(n))

    def get_loss_function(self):
        def zakharov(x):
            i_half = np.arange(.5, (self.n+1)*.5, .5)   # Equal to 0.5i
            isum = np.sum(i_half * x, axis=-1)
            x2_sum = np.sum(np.square(x), axis=-1)
            return x2_sum + np.square(isum) + np.power(isum, 4)
        return zakharov

    def get_grad_function(self):
        def zakharov_grad(x):
            i_half = np.arange(.5, (self.n+1)*.5, .5)   # Equal to 0.5i
            isum = np.sum(i_half * x, axis=-1)
            coeff = isum + 2*np.power(isum, 3)  # Move *2 to next line to reduce flops
            return 2*(x + coeff * i_half)
        return zakharov_grad

