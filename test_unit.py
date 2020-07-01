import pytest
import numpy as np
from autograd import grad
from Optimizer_Scripts.functions import ackley_gen, rastrigin_gen, ackley_deriv_gen, rast_deriv_gen
from Optimizer_Scripts.optimizers import Adam, Momentum, NesterovMomentum
from Optimizer_Scripts.Delayer import Delayer

def test_optimizers():
    n = 2
    ack = ackley_gen(n)
    ack_deriv_grad = grad(ack)
    ack_deriv = ackley_deriv_gen(n)
    rast = rastrigin_gen(n)
    rast_deriv = rast_deriv_gen(n)
    rast_deriv_grad = grad(rast)
    adam_opt = Adam()
    momentum_opt = Momentum()
    n_m_opt = NesterovMomentum()
    #first test initializer
    x_init = np.random.uniform(-5.12,5.12,n)
    adam_opt.initialize(x_init)
    momentum_opt.initialize(x_init)
    n_m_opt.initialize(x_init)
    #check initial x_value
    assert np.allclose(n_m_opt.x_state, x_init), "Nesterov initial value fail"
    assert np.allclose(momentum_opt.x_state, x_init), "Momentum initial value fail"
    assert np.allclose(adam_opt.x_state, x_init), "Adam initial value fail" 
    #check initial grad_helper
    assert np.allclose(n_m_opt.grad_helper, np.zeros(n)), "Nesterov grad helper fail"
    assert np.allclose(momentum_opt.grad_helper, np.zeros(n)), "Momentum grad helper fail"
    assert np.allclose(adam_opt.grad_helper, np.zeros(n)), "Adam grad helper fail"
    #reset the initalized
    adam_opt.initialized = False
    momentum_opt.initialized = False
    n_m_opt.initialized = False
    #now test an update
    state_n_m_opt = n_m_opt(x_init, ack_deriv(x_init), iteration_num = 1)
    state_momentum = momentum_opt(x_init, ack_deriv(x_init), iteration_num = 1)
    state_adam = adam_opt(x_init, ack_deriv(x_init), iteration_num = 1)
    assert np.allclose(state_n_m_opt, x_init - n_m_opt.learning_rate * ack_deriv(x_init)), "First update Nesterov failed"
    #assert np.allclose(state_adam, np.zeros(n)), "First update adam failed"
    assert np.allclose(state_momentum, x_init - momentum_opt.learning_rate * ack_deriv(x_init)), "First update momentum failed"
    assert not np.allclose(n_m_opt.grad_helper, np.zeros(n)), "Update of the gradient helper failed in Nesterov"
    print("done")
