import numpy
import copy
import sys


def gen_new_src(dim, lbs, ubs):
    new_src = numpy.empty(dim)
    for i in range(0, dim):
        new_src[i] = numpy.random.uniform(lbs[i], ubs[i])

    return new_src


class Source:
    def __init__(self, dim, lbs, ubs):
        self.x = gen_new_src(dim, lbs, ubs)
        self.fit = None
        self.trial_cnt = 0


class ABC:
    def __init__(self, dim_input, fit_func, lbs, ubs, best_x=None, size_pop=100, opt_type='min', lim_trial=5, pnt_func=None):
        self.dim_input = dim_input
        self.fit_func = fit_func
        self.lbs = lbs
        self.ubs = ubs
        self.best_x = best_x
        self.size_pop = size_pop
        self.opt_type = opt_type
        self.lim_trial = lim_trial
        self.pnt_func = pnt_func
        self.pop = None
        self.best_sol = None
        self.best_fit = -1e+8

        if opt_type is not 'min' and opt_type is not 'max':
            print('Optimization type error (given: ' + opt_type + ')')
            sys.exit()

    def calc_fit(self, x):
        x = x.reshape(1, -1)

        if self.opt_type == 'min':
            fit = 1 / self.fit_func(x)
        elif self.opt_type == 'max':
            fit = self.fit_func(x)

        if self.pnt_func is not None:
            fit += self.pnt_func(x)

        return fit

    def rlt_wheel_sel(self, probs):
        prob_sel = numpy.random.rand()
        sum_probs = 0

        for i in range(0, self.size_pop):
            sum_probs += probs[i]

            if prob_sel < sum_probs:
                return i

    def init_pop(self):
        self.pop = [Source(self.dim_input, self.lbs, self.ubs) for i in range(0, self.size_pop)]

        if self.best_x is not None:
            self.pop[0].x = self.best_x

        for i in range(0, self.size_pop):
            self.pop[i].fit = self.calc_fit(self.pop[i].x)

    def search_src(self, src, olk_bee=True):
        new_src = Source(self.dim_input, self.lbs, self.ubs)
        coeff = numpy.random.uniform(-1, 1, self.dim_input)
        sel_neb_idx = numpy.random.randint(0, self.size_pop)

        if olk_bee:
            new_src.x = src.x + coeff * (src.x - self.pop[sel_neb_idx].x)
        else:
            sel_src_idx = numpy.random.randint(0, self.size_pop)
            new_src.x = self.best_sol + coeff * (self.pop[sel_src_idx].x - self.pop[sel_neb_idx].x)

        new_src.fit = self.calc_fit(new_src.x)

        if new_src.fit > src.fit:
            return new_src
        else:
            src.trial_cnt += 1
            return copy.deepcopy(src)

    def work_emp_bees(self):
        new_srcs = list()

        for i in range(0, self.size_pop):
            new_srcs.append(self.search_src(self.pop[i], olk_bee=False))

        self.pop = new_srcs

    def work_olk_bees(self):
        fits = numpy.array([src.fit for src in self.pop])
        prob_srcs = fits / numpy.sum(fits)
        new_srcs = list()

        for i in range(0, self.size_pop):
            sel_src_idx = self.rlt_wheel_sel(prob_srcs)
            new_srcs.append(self.search_src(self.pop[sel_src_idx]))

        self.pop = new_srcs

    def work_sct_bees(self):
        for i in range(0, self.size_pop):
            if self.pop[i].trial_cnt > self.lim_trial:
                self.pop[i] = Source(self.dim_input, self.lbs, self.ubs)
                self.pop[i].fit = self.calc_fit(self.pop[i].x)

    def set_best_sol(self):
        fits = numpy.array([src.fit for src in self.pop]).flatten()
        cur_best_fit = numpy.max(fits)

        if cur_best_fit > self.best_fit:
            self.best_sol = self.pop[numpy.argmax(fits)].x
            self.best_fit = cur_best_fit

    def run(self, max_iter):
        self.init_pop()
        self.set_best_sol()

        for i in range(0, max_iter):
            self.work_emp_bees()
            self.work_olk_bees()
            self.work_sct_bees()
            self.set_best_sol()
            print(i, self.best_fit)

        return self.best_sol, self.best_fit
