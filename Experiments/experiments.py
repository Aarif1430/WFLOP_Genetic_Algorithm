from API.misc import *
from GA_Optimizer.WindForm_GA import *
from main import *

# --EXPERIMENTS--
def test_pop_size():
    optimizer = GAOptimizer(rows=rows_cells, cols=cols_cells, no_of_turbines=number_of_turbines,
                            pop_size=population_size, iteration=iteration_times, cell_width=cell_width,
                            elite_rate=elite_rate, cross_rate=cross_rate, random_rate=random_rate,
                            mutate_rate=mutate_rate)
    random_pop_performance = []
    run_time_performance = []
    for _ in range(10):
        optimizer.gen_init_pop()
        run_time, conv_eff, _, _ = optimizer.evolve()
        random_pop_performance.append(np.mean(conv_eff))
        run_time_performance.append(np.mean(run_time))
    generic_plot(range(len(random_pop_performance)), random_pop_performance)
    generic_bar(run_time_performance, random_pop_performance, title='Run time performance',
                xlabel='Run Time', ylabel='Conversion Efficiency', save=True, in_parent=False)


def test_mutation_rate():
    mutation_rate_performances = []
    mutation_rates = [round(np.random.uniform(0, 1) / 10, 3) for _ in range(8)]
    for mr in mutation_rates:
        print(f"Mutation Rate: {mr}")
        current_iter_performances = []
        for i in range(1, 6):  # repeat experiment for reliability
            print(f"Repetition: {i}")
            optimizer = GAOptimizer(rows=rows_cells, cols=cols_cells, no_of_turbines=number_of_turbines,
                                    pop_size=population_size, iteration=20, cell_width=cell_width,
                                    elite_rate=elite_rate, cross_rate=cross_rate, random_rate=random_rate,
                                    mutate_rate=mr)
            optimizer.gen_init_pop()
            _, conv_eff, _, _ = optimizer.evolve()
            current_iter_performances.append(np.mean(conv_eff))
        mutation_rate_performances.append(np.mean(current_iter_performances))
    mutation_rates = [str(round(mr, 2)) for mr in mutation_rates]
    generic_bar(mutation_rates, mutation_rate_performances, title='Analysing the Mutation Rate Performance Impact',
                xlabel='Mutation Rate', ylabel='Conversion Efficiency', save=True, in_parent=False)


def test_crossover_rate():
    crossover_rate_performances = []
    crossover_rates = [np.random.uniform(0, 1) for _ in range(8)]
    for cr in crossover_rates:
        print(f"Crossover Rate: {cr}")
        current_iter_performances = []
        for i in range(1, 6):  # repeat experiment for reliability
            print(f"Repetition: {i}")
            optimizer = GAOptimizer(rows=rows_cells, cols=cols_cells, no_of_turbines=number_of_turbines,
                                    pop_size=population_size, iteration=20, cell_width=cell_width,
                                    elite_rate=elite_rate, cross_rate=cr, random_rate=random_rate,
                                    mutate_rate=mutate_rate)
            optimizer.gen_init_pop()
            _, conv_eff, _, _ = optimizer.evolve()
            current_iter_performances.append(np.mean(conv_eff))
        crossover_rate_performances.append(np.mean(current_iter_performances))
    crossover_rates = [str(round(cr, 2)) for cr in crossover_rates]
    generic_bar(crossover_rates, crossover_rate_performances, title='Analysing the Crossover Rate Performance Impact',
                xlabel='Crossover Rate', ylabel='Conversion Efficiency', save=True, in_parent=False)



if __name__=='__main__':
    test_pop_size()
    test_mutation_rate()
    test_crossover_rate()
