import csv
import copy
import os
import time

# from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
# from hyperopt.pyll.base import scope
import torch
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from deep_rl_torch.trainer import Trainer
from train import create_arg_dict


def meanSmoothing(x, N):
    x = np.array(x)
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for i in range(dim_len):
        if N % 2 == 0:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 2
        else:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 1
        # cap indices to min and max indices
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b])
    return out


def calculate_reduced_idxs(len_of_point_list, max_points):
    if max_points != 0:
        step_size = len_of_point_list // max_points
        step_size += 1 if len_of_point_list % max_points else 0
    else:
        return range(len_of_point_list)
    return range(0, len_of_point_list, step_size)


def reducePoints(list_of_points, max_points_per_line):
    if max_points_per_line != 0:
        step_size = len(list_of_points) // max_points_per_line
        step_size += 1 if len(list_of_points) % max_points_per_line else 0
    else:
        return range(len(list_of_points)), list_of_points
    steps = range(0, len(list_of_points), step_size)
    list_of_points = [np.mean(list_of_points[i:i + step_size]) for i in steps]
    return list_of_points


def mean_final_percent(result_list, percentage=0.1):
    final_percent_idx = int(len(result_list) * (1 - percentage))
    return np.mean(result_list[final_percent_idx:])


def run_metric(result_list, percentage=0.1, final_percentage_weight=1):
    return np.mean(result_list) * (1 - final_percentage_weight) + mean_final_percent(result_list,
                                                                                     percentage) * final_percentage_weight


def plot_rewards(rewards, name=None, xlabel="Step"):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel(xlabel)
    plt.ylabel('Return of current Episode')

    idxs = calculate_reduced_idxs(len(rewards), 1000)
    rewards = reducePoints(rewards, 1000)

    plt.plot(idxs, rewards)
    # Apply mean-smoothing and plot result
    window_size = len(rewards) // 10
    window_size += 1 if window_size % 2 == 0 else 0
    means = meanSmoothing(rewards, window_size)
    max_val = np.max(means)
    min_val = np.min(means)
    # plt.ylim(min_val, max_val * 1.1)
    plt.plot(idxs, means)
    if name is None:
        plt.savefig("current_test.pdf")
    else:
        plt.savefig(name + "_current.pdf")
    plt.pause(0.001)  # pause a bit so that plots are updated


def saveList(some_list, path):
    with open(path + ".pt", 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(some_list)


def saveDict(some_dict, path):
    torch.save(some_dict, path + ".pt")
    return


def saveDict_simple(some_dict, path):
    torch.save(some_dict, path + ".pt")
    return


def max_trial_idx(path):
    names = os.listdir(path)
    pt_files = [int(name[:-3]) for name in names if name[-3:] == ".pt"]
    if len(pt_files) > 0:
        return max(pt_files) + 1
    else:
        return 1


def storeResults(list_of_results, path, base_idx=None):
    if not os.path.exists(path):
        os.makedirs(path)

    if base_idx is None:
        base_idx = max_trial_idx(path)

    for idx, result_list in enumerate(list_of_results):
        full_path = path + str(base_idx + idx)
        torch.save(result_list, full_path + ".pt")


def readResults(path, name_list):
    # for file in path. if file is not dir, then it is result, therefore store in list
    dict_of_results = {}
    for alg_name in name_list:
        result_list = []
        for run in sorted(os.listdir(path + alg_name)):
            run_path = path + alg_name + "/" + run
            if not os.path.isdir(run_path) and run[0] != "0":
                # result = readList(run_path)
                result = torch.load(run_path)
                result_list.append(result)
        dict_of_results[alg_name] = result_list
    return dict_of_results

def store_logs(list_of_log_dicts, path, base_idx=None):
    os.makedirs(path, exist_ok=True)
    if base_idx is None:
        base_idx = max_trial_idx(path)
    for idx, log_of_run in enumerate(list_of_log_dicts):
        log_path = path + "/" + str(base_idx + idx) + ".pt"
        log_of_run.cleanse()
        torch.save(log_of_run, log_path)

def read_logs(path, name_list):
    """Recovers the logs for every alg from disc."""
    alg_logs = {}
    #print("Reading logs: ")
    for alg_name in name_list:
        #print(alg_name)
        alg_path = path + alg_name + "/logs/"
        logs = []
        for log_file_name in os.listdir(alg_path):
            if log_file_name[-3:] != ".pt" or log_file_name[0] == "0":
                continue
            log = torch.load(alg_path + "/" + log_file_name)
            #print("added a log")
            logs.append(log)
        alg_logs[alg_name] = logs
    return alg_logs


def reshape_logs(alg_logs, attribute):
    """Extracts attribute from the logs in alg_logs and brings them into a plottable shape.
    Now we have the form: {"Q": [log1, log2], "Q+TDEC": [log1, log2]} where log1,log2 both contain dicts
    Bring it into the form i.e. {"Epsilon":{"Q":[[1,2,3], [1,1,1]], "Q+TDEC":[[1,2,3], [1,2,1]},
                                   "Loss_TDEC":{"Q+TDEC":[1,2,3]}}
    """
    dict_of_logs = {}
    for alg_name in alg_logs:
        logs = alg_logs[alg_name]
        for log in logs:
            log_storage = getattr(log, attribute)
            for key in log_storage:
                values = log_storage[key]
                # Add values to dict in correct form:
                if key in dict_of_logs:
                    if alg_name in dict_of_logs[key]:
                        dict_of_logs[key][alg_name].append(values)
                    else:
                        dict_of_logs[key][alg_name] = [values]
                else:
                    dict_of_logs[key] = {alg_name: [values]}
    return dict_of_logs

def store_logs_old(list_of_log_dicts, path, base_idx=None):
    os.makedirs(path, exist_ok=True)

    if base_idx is None:
        base_idx = max_trial_idx(path)

    for idx, log_dict_of_run in enumerate(list_of_log_dicts):
        for key in log_dict_of_run:
            log_name = path + key + "/"
            if not os.path.exists(log_name):
                os.makedirs(log_name)
            # base_idx = max_trial_idx(log_name)
            full_path = log_name + str(base_idx + idx)
            log_list = log_dict_of_run[key]
            torch.save(log_list, full_path + ".pt")
            # saveList(log_list, name)


def read_logs_old(path, name_list):
    # get to shape of form {"Epsilon":{"Q":[[1,2,3], [1,1,1]], "Q+TDEC":[[1,2,3], [1,2,1]}, "Loss_TDEC":{"Q+TDEC":[1,2,3]}}
    dict_of_logs = {}
    for alg_name in name_list:
        for log_name in sorted(os.listdir(path + alg_name + "/logs/")):
            values = []
            for run_name in sorted(os.listdir(path + alg_name + "/logs/" + log_name + "/")):
                if run_name[-3:] != ".pt" or run_name[0] == "0":
                    continue
                file_path = path + alg_name + "/logs/" + log_name + "/" + run_name
                # log_of_run = readList(file_path)
                log_of_run = torch.load(file_path)
                print(log_of_run)
                print(type(log_of_run))
                values.append(log_of_run)

            if log_name in dict_of_logs:
                dict_of_logs[log_name][alg_name] = values
            else:
                dict_of_logs[log_name] = {alg_name: values}
    return dict_of_logs


def storeHyperparameters(hyperparam_list, path):
    os.makedirs(path, exist_ok=True)

    for idx, hyperparam_dict in hyperparam_list:
        name = path + str(idx)
        torch.save(hyperparam_dict, name + ".pt")
        # saveDict(hyperparam_dict, name)


def create_masked_array(list_of_lists):
    """ Input list is a list of lists. Each list has shape [N], where N
    is the number of timesteps. It returns a numpy array of shape [N_max],
    where N_max is the maximum length of N in the list. """
    #print(len(list_of_lists))
    #print(list_of_lists[0].shape)
    #print(len(list_of_lists[0]))
    max_len = max([len(x) for x in list_of_lists])
    if isinstance(list_of_lists[0][0], np.ndarray) and len(list_of_lists[0][0].shape) > 0:
        shape = [len(list_of_lists), max_len] +  list(list_of_lists[0][0].shape)
    else:
        shape = [len(list_of_lists), max_len]
    masked_array = np.ma.masked_array(np.zeros(shape), mask=True)
    for idx, item_list in enumerate(list_of_lists):
        list_len = len(item_list)
        masked_array[idx, :list_len] = item_list
    return masked_array


def apply_to_log_list(func, log_list):
    """Applies __func__ to the ensemble of logs stored in log_list"""
    max_len = len(max(log_list, key=lambda x: len(x)))
    if isinstance(log_list[0][0], list) or isinstance(log_list[0][0], np.ndarray):
        pass
    else:
        shape = (len(log_list), max_len)
    masked_array = np.ma.masked_array(np.zeros(shape), mask=True)
    for idx, item_list in enumerate(log_list):
        list_len = len(item_list)
        masked_array[idx, :list_len] = item_list
    return masked_array


from torch.nn.utils.rnn import pad_sequence


def take_mean_over_steps(timeseries_list, max_timesteps=None):
    """ Input list is a list of tensors. Each tensor has shape [N, F], where N
    is the number of timesteps, which varies from tensor to tensor, and F is
    the constant number of Features. It returns a numpy array of shape [N_max, F],
    where N_max is the maximum length of N in the list. For each step in N, the
    result is the average over all members of the initial list. If a list is not
    long enough it is not included in that timestep."""
    timeseries_list = pad_sequence(timeseries_list, batch_first=True, padding_value=-1).numpy()
    mask = timeseries_list < 0
    masked_series = np.ma.masked_array(timeseries_list, mask=mask)
    if max_timesteps is not None:
        masked_series = masked_series[:, :max_timesteps]
    return masked_series.mean(axis=0), masked_series.std(axis=0)


def plotDict(dict_of_alg_runs, step_dict, name, plot_best_idxs=None, path="", max_points=0, length_of_tests=100000,
             totalName=""):
    prop_cycle = (cycler('linestyle', ['-', '--', ':', '-.']) * cycler('color', ['r', 'g', 'b', 'y', 'm', 'c', 'k']))
    for alg_name, props, count in zip(sorted(dict_of_alg_runs), prop_cycle, range(len(dict_of_alg_runs))):
        color = props["color"]
        linestyle = props["linestyle"]


        values = np.array(dict_of_alg_runs[alg_name])

        if  len(values) == 0 or (isinstance(values[0][0], np.ndarray) and len(values[0][0].shape) > 0):
#        if len(values) == 0 or np.array(values[0]).ndim > 1:
            # TODO: plot heatmap instead of lineplot if ndim > 1
            continue
        masked_vals = create_masked_array(values)

        
        steps = np.array(step_dict[alg_name])
        steps = create_masked_array(steps).mean(axis=0).astype(int)


        #if plot_best_idxs is not None:
        #    values = [values[idx] for idx in plot_best_idxs[count]]

     
        

        means = np.mean(masked_vals, axis=0)
        stdEs = np.std(masked_vals, axis=0)

        window_size = len(means) // 10
        if window_size % 2 == 0:
            window_size += 1
        means = meanSmoothing(means, window_size)
        stdEs = meanSmoothing(stdEs, window_size)

        # idxs = calculate_reduced_idxs(length_of_tests, len(means))
        #idxs = range(len(means))

        p = plt.plot(steps, means, label=alg_name, color=color, linestyle=linestyle)
        plt.fill_between(steps, means - stdEs, means + stdEs, alpha=.25, color=p[0].get_color())
    fileName = name if plot_best_idxs is None else name + "_bestRuns_"
    if name == "Total Reward":
        title = "Rewards per Episode during Training without Exploration"
    else:
        title = name + " Values during Training" if plot_best_idxs is None else name + " Values for best Runs during Training"
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(name)
    filePath = path + totalName + fileName
    plt.legend()
    plt.savefig(filePath + ".pdf")
    plt.clf()


def generate_log_plots_new(dict_of_logs, dict_of_steps, path="", max_points=2000, length_of_tests=100000, totalName=""):
    # shape in form of {"Epsilon":{"Q":[[1,2,3], [1,1,1]], "Q+TDEC":[[1,2,3], [1,2,1]}, "Loss_TDEC":{"Q+TDEC":[1,2,3]}}

    path += "/logs/"
    os.makedirs(path, exist_ok=True)
    for log_name in dict_of_logs:
        alg_dict = dict_of_logs[log_name]
        step_dict = dict_of_steps[log_name]
        plotDict(alg_dict, step_dict, log_name, path=path, max_points=max_points, length_of_tests=length_of_tests,
                 totalName=totalName)


def generatePlot(dict_of_results, drawIQR=False, smoothing="yes", plot_best=10, path="", env="",
                 totalName="", draw_all_lines=False, max_points=0, length_of_tests=100000):
    prop_cycle = (cycler('linestyle', ['-', '--', ':', '-.']) * cycler('color', ['r', 'g', 'b', 'y', 'm', 'c', 'k']))

    for name, props in zip(sorted(dict_of_results), prop_cycle):
        color = props["color"]
        linestyle = props["linestyle"]

        result_list = dict_of_results[name]

        window_size = len(result_list[0]) // 10
        if window_size % 2 == 0:
            window_size += 1
        # print(length_of_tests, len(result_list[0]))

        if plot_best:
            # best_idxs = np.argpartition(np.mean(result_list, axis=1), -plot_best)[-plot_best:]
            # result_list = [result_list[idx] for idx in best_idxs]
            result_list = getBestResults(result_list, plot_best)

        if draw_all_lines:
            for count, line in enumerate(result_list):
                line = meanSmoothing(line, window_size)
                if count == 0:
                    plt.plot(idxs, line, color=color, linestyle=linestyle, label=name)
                else:
                    plt.plot(idxs, line, color=color, linestyle=linestyle)
            continue

        #print(dict_of_results)
        #print(name)
        #print(result_list)
        #print(len(result_list))
        results = create_masked_array(result_list)
        means = np.mean(results, axis=0)
        stds = np.std(results, axis=0)
        medians = np.median(results, axis=0)
        IQRs = np.quantile(results, [0.25, 0.75], axis=0)
        # idxs = calculate_reduced_idxs(length_of_tests, len(result_list[0]))
        idxs = np.arange(len(means)).astype(int) * (length_of_tests // len(means))

        if drawIQR:
            data = medians
            shadingLower = IQRs[0]
            shadingUpper = IQRs[1]
        else:
            data = means
            shadingLower = means - stds
            shadingUpper = means + stds

        smoothedData = meanSmoothing(data, window_size)
        smoothedShadingLower = meanSmoothing(shadingLower, window_size)
        smoothedShadingUpper = meanSmoothing(shadingUpper, window_size)

        if smoothing == "yes" or smoothing == "both":
            plt.plot(idxs, smoothedData, label=name, color=color, linestyle=linestyle)
            plt.fill_between(idxs, smoothedShadingLower, smoothedShadingUpper, alpha=.25, color=color)
        if smoothing == "no" or smoothing == "both":
            if smoothing == "both":
                alpha = .3
                plt.plot(idxs, data, color=color, linestyle=linestyle, alpha=alpha)
            else:
                alpha = 1
                plt.fill_between(idxs, shadingLower, shadingUpper, alpha=.25, color=color)
                plt.plot(idxs, data, label=name, color=color, linestyle=linestyle, alpha=alpha)

    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Return")

    title = "Rewards per Episode during Training for " + env
    plt.title(title)
    fileName = totalName
    if smoothing == "yes":
        fileName += "Smoothed"
    elif smoothing == "both":
        fileName += "Both"
    elif smoothing == "no":
        fileName += "Unsmoothed"
    if drawIQR:
        fileName += "_IQR"
    else:
        fileName += "_stdErr"
    if draw_all_lines:
        fileName += "_allLines"
    if plot_best:
        fileName += "_plotBest" + str(plot_best)
    plt.savefig(path + fileName + ".pdf")
    plt.clf()


def getBestIdxs(list_of_result_lists, plot_best):
    list_of_idx_lists = []
    for result_list in list_of_result_lists:
        if len(result_list) <= plot_best:
            return None
        best_idxs = np.argpartition(np.mean(result_list, axis=1), -plot_best)[-plot_best:]
        list_of_idx_lists.append(best_idxs)
    return list_of_idx_lists


def reducePoints(list_of_points, max_points_per_line):
    if max_points_per_line != 0:
        step_size = len(list_of_points) // max_points_per_line
        step_size += 1 if len(list_of_points) % max_points_per_line else 0
    else:
        return range(len(list_of_points)), list_of_points
    steps = range(0, len(list_of_points), step_size)
    list_of_points = [np.mean(list_of_points[i:i + step_size]) for i in steps]
    return list_of_points


def createRandomParamList(number_of_tests, verbose=True):
    import random
    randomizedParamList = []

    discountOptions = [0.9, 0.95, 0.98, 0.99, 0.999]
    lrOptions = [0.005, 0.001, 0.0005, 0.0001]
    targetUpdateOptions = [10, 50, 100, 1000, 5000, 10000]
    neuronNumberOptions = [32, 64, 128, 256]
    hiddenLayerOptions = [1, 2]
    batchSizeOptions = [16, 32, 64, 128, 256]
    memorySizeOptions = [1024, 5000, 10000, 50000, 100000]
    epsDecayOptions = [0.25, 0.1, 0.05, 0.01, 0.001]
    activation = ["sigmoid", "relu", "elu", "selu"]

    for i in range(number_of_tests):
        randomizedParams = {"target_network_steps": random.sample(targetUpdateOptions, 1)[0],
                            "lr_Q": random.sample(lrOptions, 1)[0], "gamma_Q": random.sample(discountOptions, 1)[0],
                            "hidden_neurons": random.sample(neuronNumberOptions, 1)[0],
                            "hidden_layers": random.sample(hiddenLayerOptions, 1)[0],
                            "batch_size": random.sample(batchSizeOptions, 1)[0],
                            "replay_buffer_size": random.sample(memorySizeOptions, 1)[0],
                            "epsilon_mid": random.sample(epsDecayOptions, 1)[0],
                            "activation_function": random.sample(activation, 1)[0]}
        randomizedParamList.append(randomizedParams)

        if verbose:
            print("Run " + str(i) + ":")
            for key in randomizedParams:
                print(key + ":", randomizedParams[key], end="  ")
            print()
    return randomizedParamList


def getBestResults(list_of_results, amount, run_metric_percentage=0.1, run_metric_final_percentage_weight=1):
    results = sorted(list_of_results, key=lambda result: run_metric(result, percentage=run_metric_percentage,
                                                                    final_percentage_weight=run_metric_final_percentage_weight),
                     reverse=True)
    return results[:amount]


def get_best_result_idxs(list_of_results, number, run_metric_percentage=0.1, run_metric_final_percentage_weight=1):
    best_results = getBestResults(list_of_results, number, run_metric_percentage=run_metric_percentage,
                                  run_metric_final_percentage_weight=run_metric_final_percentage_weight)
    idxs = []
    for idx, result in enumerate(list_of_results):
        if result in best_results:
            idxs.append(idx)
            best_results.remove(result)
    return idxs


def get_results_for_idx(path, idx):
    result_list = []
    for item in sorted(os.listdir(path)):
        if os.path.isdir(path + item):
            if item == str(idx):
                for run in sorted(os.listdir(path + item + "/")):
                    # result_of_run = readList(path + item + "/" + run)
                    result_of_run = torch.load(path + item + "/" + run + ".pt")
                    result_list.append(result_of_run)
        else:
            if item[:-4] == str(idx):
                # result_of_run = readList(path + item)
                result_of_run = torch.load(path + item + ".pt")
                result_list.append(result_of_run)
    return result_list


def store_optimizer_runs(path, runs, idx):
    pass


def train_model_to_optimize_tpe(space, verbose=False, comet_ml=None, run_metric_percentage=0.1,
                                run_metric_final_percentage_weight=1):
    length_of_tests = space.pop("length_of_tests")
    env = space.pop("env")
    device = space.pop("device")
    trial = space.pop("trial")
    max_points = space.pop("max_points")
    trials = space.pop("trials")
    optimization_experiment = space.pop("optimization_experiment")
    evals_per_optimization_step = space.pop("evals_per_optimization_step")

    iteration = len(trials.results)

    performance = train_model_to_optimize(space, length_of_tests, env, device, trial, max_points,
                                          evals_per_optimization_step, iteration, trials=trials,
                                          run_metric_percentage=run_metric_percentage,
                                          run_metric_final_percentage_weight=run_metric_final_percentage_weight)
    optimization_experiment.log_metric("Performance during optimization", performance * -1)
    optimization_experiment.set_step(iteration)
    return {"loss": performance, 'status': STATUS_OK, "params": space, "iteration": iteration}


def train_model_to_optimize(hyperparam_dict, length_of_tests, env, device, trial, max_points,
                            evals_per_optimization_step, iteration, trials=None, verbose=False,
                            run_metric_percentage=0.1, run_metric_final_percentage_weight=1):
    resultList, logs = run_trial(env, evals_per_optimization_step, length_of_tests, trial,
                                 hyperparamDict=hyperparam_dict,
                                 max_points=max_points)
    performances = [run_metric(log["Metrics/Reward"], percentage=run_metric_percentage,
                               final_percentage_weight=run_metric_final_percentage_weight) * -1 for log in logs]
    performance = np.mean(performances)

    print("Run ", str(iteration + 1), " Performance: ", round(performance, 1) * -1)
    if verbose:
        print("Hyperparams:", end=" ")
        for key in hyperparam_dict:
            print(key + str(
                round(hyperparam_dict[key], 5) if type(hyperparam_dict[key]) is type(1.0) else hyperparam_dict[
                    key]) + "|", end=" ")
        print()
    return performance


def count_files(path, name):
    """Counts the amount of files in the folder at path that contain name"""
    return len([file_name for file_name in os.listdir(path) if name in file_name])


def has_enough_runs(path, n_runs):
    count = count_files(path, ".pt")
    return n_runs == count


def print_trial_stats(logs, run_metric_percentage, run_metric_final_percentage_weight, trial_start_time):
    returns = create_masked_array([log["Metrics/Return"] for log in logs])
    # meansPerEpisode = np.mean(returns, 0)
    # overAllFinalPercent = round(run_metric(meansPerEpisode, percentage=run_metric_percentage, final_percentage_weight=run_metric_final_percentage_weight), 2)
    overallEpisodeMean = round(np.mean(returns), 2)
    overallEpisodeStd = round(np.std(returns), 2)
    overallEpisodeMedian = round(np.median(returns), 2)

    test_score_per_training_run = create_masked_array([log["Metrics/Test Return"] for log in logs])
    # [run_metric(log["Metrics/Test Return"], percentage=run_metric_percentage, final_percentage_weight=run_metric_final_percentage_weight) for log in logs]
    test_score_mean = round(np.mean(test_score_per_training_run), 3)
    test_score_std = round(np.std(test_score_per_training_run), 3)

    print("Returns: ")
    print("Test mean score:", test_score_mean)
    print("Test std of score:", test_score_std)
    # print("Trial score: ", overAllFinalPercent)
    print("Trial mean: ", overallEpisodeMean)
    print("Trial median: ", overallEpisodeMedian)
    print("Trial std: ", overallEpisodeStd)
    print("Trial time: ", round((time.time() - trial_start_time) / 60, 2), " minutes.")
    print()


class MockTrialFile:
    def __init__(self, path, num):
        self.path = path
        self.name = "0" + str(num) + ".pt"
        self.file_path = path + self.name

    def __enter__(self):
        open(self.file_path, 'a').close()

    def __exit__(self, _type, value, traceback):
        try:
            os.remove(self.file_path)
        except:
            os.remove(self.file_path)


def run_trial(env, number_of_tests, length_of_tests, hyperparameters, path, randomizeList=[],
              max_points=2000, hyperparamDict={}, verbose=True):
    logs = []

    args, _ = create_arg_dict([], env=env)
    args.update(hyperparameters)
    args["steps"] = length_of_tests
    i = 0
    trainer = None
    print("Still need to run ", number_of_tests - count_files(path, ".pt"), "experiments...")
    while count_files(path, ".pt") < number_of_tests:
        if len(randomizeList) > 0:
            randomizedParams = randomizeList[i]
            print("Hyperparameters:")
            for key in randomizedParams:
                print(key + ":", randomizedParams[key], end=" ")
            print()
            args.update(randomizeParams)

        # Create mock file such that other processes know that this process is running:
        trial_idx = max_trial_idx(path)
        with MockTrialFile(path, trial_idx):
            # Train a model:
            if trainer is None:
                trainer = Trainer(env, args)
            else:
                trainer.reset()
            n_eps, log = trainer.run(verbose=False, total_steps=length_of_tests, disable_tqdm=False)

        returns = log["Metrics/Return"]
        if number_of_tests > 1:
            mean_reward = np.mean(returns)
            print("Run ", str(i + 1), "/", str(number_of_tests), end=" | ")
            print("Mean Return ", round(mean_reward, 1), end=" ")
            print("Score: ", round(run_metric(returns), 1))
        # Extract returns
        # Store finished test log:
        store_logs([log], path + "logs/", trial_idx)
        # Store steps:
        #store_steps([log], path + "steps/", trial_idx)
        # Store returns in main folder (is used to keep track of trial progress):
        storeResults([returns], path, trial_idx)
        # Save log in list:
        logs.append(log)
        i += 1
    return logs


def run_exp(env, alg_hyperparam_list, number_of_tests=20, length_of_tests=600, window_size=None, randomizeParams=False,
            path="", max_points=2000, optimize="no", number_of_best_runs_to_check=5,
            number_of_checks_best_runs=5, final_evaluation_runs=20, number_of_hyperparam_optimizations=2,
            evals_per_optimization_step=2, optimize_only_lr=False, optimize_only_Q_params=False,
            run_metric_percentage=0.1, run_metric_final_percentage_weight=0):
    print("Experiment on ", env)
    experiment_start_time = time.time()

    # Set up data path:
    root = "Experiments/"
    exp_path = root + "Data/" + env + "_"
    if optimize != "no":
        exp_path += "optimize" + optimize + "_"
        if optimize[-4:] == "best":
            exp_path += str(number_of_best_runs_to_check) + "_" + str(number_of_checks_best_runs) + "_" \
                        + str(final_evaluation_runs) + "_" + str(number_of_hyperparam_optimizations) + "_" \
                        + str(evals_per_optimization_step) + "_" + str(run_metric_percentage) + "_" + str(
                run_metric_final_percentage_weight) + "_"
            if optimize_only_lr:
                exp_path += "lr_opt_"
            if optimize_only_Q_params:
                exp_path += "no_occams_opt_"
    elif randomizeParams:
        exp_path += "randomize_"
    exp_path += str(length_of_tests) + "/"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    plots_path = root + "Plots/"
    os.makedirs(plots_path + path, exist_ok=True)

    # Create randomized hyperparameter list:
    rand_hyperparams = []
    if randomizeParams:
        rand_hyperparams = createRandomParamList(number_of_tests, verbose=False)

    name_list = []
    for hyperparam_dict in alg_hyperparam_list:
        trial_start_time = time.time()

        name = hyperparam_dict.pop("name")
        if name not in name_list:
            name_list.append(name)

        trial_path = exp_path + name + "/"

        # Check if and how many trials need to be run:
        os.makedirs(trial_path, exist_ok=True)

        enough_trials = has_enough_runs(trial_path, number_of_tests)
        if not enough_trials:
            print("Running  trials for ", name)
            logs = run_trial(env, number_of_tests, length_of_tests, hyperparam_dict,
                             path=trial_path,
                             randomizeList=rand_hyperparams,
                             max_points=max_points)
            print_trial_stats(logs, run_metric_percentage, run_metric_final_percentage_weight,
                              trial_start_time)
        else:
            print("There are already enough trials stored for trial ", name)

    #self.visualize_logs(sorted(name_list))

    # Read data:
    name_list = sorted(name_list)
    loading_name_list = [name for name in name_list]
    dict_of_results = readResults(exp_path, loading_name_list)
    logs = read_logs(exp_path, loading_name_list)
    dict_of_logs = reshape_logs(logs, "storage")
    dict_of_steps = reshape_logs(logs, "step_dict")
    # list_of_hyperparameter_dicts = readHyperparameters(folderName, name_list)

    # Plot data:
    path = plots_path + path + "/"
    print("Plot path: ", path)
    os.makedirs(path, exist_ok=True)

    totalName = str(number_of_tests) + "_" + str(length_of_tests) + "_" + optimize
    if optimize[-4:] == "best":
        totalName += "_" + str(number_of_best_runs_to_check) + "_" + str(number_of_checks_best_runs) + "_" \
                     + str(final_evaluation_runs) + "_" + str(number_of_hyperparam_optimizations) + "_" \
                     + str(evals_per_optimization_step)

    for shadingOption in [True, False]:
        for smoothingOption in ["yes"]:  # , "both"]:
            for plotBestOption in [0, 5, 10]:
                if number_of_tests > plotBestOption:
                    for drawAll in [True, False]:
                        if not drawAll or (
                                drawAll and 15 > plotBestOption > 0 and smoothingOption == "yes" and shadingOption == False):
                            generatePlot(dict_of_results, drawIQR=shadingOption, smoothing=smoothingOption,
                                         plot_best=plotBestOption, path=path, env=env,
                                         draw_all_lines=drawAll, max_points=max_points, totalName=totalName,
                                         length_of_tests=length_of_tests)

    # Plot log plots:
    generate_log_plots_new(dict_of_logs, dict_of_steps, path=path, max_points=max_points, length_of_tests=length_of_tests,
                           totalName=totalName)

    exp_end_time = time.time() - experiment_start_time
    print("The whole experiment took ", round(exp_end_time / 60, 2), " minutes.")






















def optimize_comet(env, device, max_evals, length_of_tests, trial, max_points, optimization_experiment,
                   number_best_runs=1,
                   evals_per_optimization_step=1, optimize_only_lr=False, optimize_only_Q_params=False,
                   run_metric_percentage=0.1, run_metric_final_percentage_weight=1):
    optimizer = Optimizer("M03EcOc9o9kiG95hws4mq1uqI")
    # Declare your hyper-parameters:
    if optimize_only_lr:
        params = """
        lr_Q real [0.00001, 0.005] [0.0005] log
        """
    else:
        params = """
        lr_Q real [0.00001, 0.005] [0.0005] log
        target_network_steps integer [5, 5000] [1000]
        batch_size integer [16, 256] [128]
        epsilon_mid real [0.001, 0.25] [0.1]
        """

    if not optimize_only_Q_params and ("SPLIT_BELLMAN" in trial and trial["SPLIT_BELLMAN"] == True) or (
            "QV_SPLIT_V" in trial and trial["QV_SPLIT_V"] == True) or (
            "QV_SPLIT_Q" in trial and trial["QV_SPLIT_Q"] == True):
        params += "lr_r real [0.00001, 0.005] [0.0005] log\n    "

    '''
    filtered hyperparams:
       hidden_neurons integer [64, 128] [128]
       hidden_layers integer [1, 2] [2]
       gamma_Q real [0.9, 0.999] [0.99] log
       activation_function categorical {"sigmoid","relu","elu","selu"} ["sigmoid"]
       replay_buffer_size integer [1024, 50000] [25000]
    '''
    optimizer.set_params(params)

    trial_results = []

    for iteration in range(max_evals):
        # Get a suggestion
        suggestion = optimizer.get_suggestion()

        # Create a new experiment associated with the Optimizer
        # experiment = Experiment("M03EcOc9o9kiG95hws4mq1uqI", project_name="trash", workspace="antonwiehe")

        hyperparamDict = {}
        for key in suggestion:
            hyperparamDict[key] = suggestion[key]

        # Test the model
        score = train_model_to_optimize(hyperparamDict, length_of_tests, env, device, trial, max_points,
                                        evals_per_optimization_step, iteration,
                                        run_metric_percentage=run_metric_percentage,
                                        run_metric_final_percentage_weight=run_metric_final_percentage_weight)

        optimization_experiment.log_metric("Performance during optimization", score * -1)
        optimization_experiment.set_step(iteration)
        # Report the score back
        suggestion.report_score("accuracy", score)

        run = {"loss": score, "params": hyperparamDict}
        trial_results.append(run)
    return trial_results


def optimize_tpe(env, device, max_evals, length_of_tests, trial, max_points, optimization_experiment,
                 number_best_runs=1,
                 evals_per_optimization_step=1, optimize_only_lr=False, optimize_only_Q_params=False,
                 run_metric_percentage=0.1, run_metric_final_percentage_weight=1):
    trials = Trials()

    space = {"env": env, "device": device, "trial": trial, "max_points": max_points,
             "length_of_tests": length_of_tests, "trials": trials, "optimization_experiment": optimization_experiment,
             "evals_per_optimization_step": evals_per_optimization_step,
             "lr_Q": hp.uniform("lr_Q", 0.00001, 0.005),
             "run_metric_percentage": run_metric_percentage,
             "run_metric_final_percentage_weight": run_metric_final_percentage_weight
             }
    if not optimize_only_lr:
        space.update({  # "activation_function": hp.choice("activation_function", ["sigmoid", "relu", "elu", "selu"]),
            # "gamma_Q": hp.uniform("gamma_Q", 0.9, 0.999),
            "target_network_steps": scope.int(hp.quniform("target_network_steps", 5, 5000, 1)),
            # "hidden_neurons": scope.int(hp.quniform("hidden_neurons", 64, 128, 1)),
            # "hidden_layers": scope.int(hp.quniform("hidden_layers", 1, 2, 1)),
            "batch_size": scope.int(hp.quniform("batch_size", 16, 256, 1)),
            # "replay_buffer_size": scope.int(hp.quniform("replay_buffer_size", 1024, 50000, 1)),
            "epsilon_mid": hp.uniform("epsilon_mid", 0.001, 0.25)
        })

    if not optimize_only_Q_params and ("SPLIT_BELLMAN" in trial and trial["SPLIT_BELLMAN"] == True) or (
            "QV_SPLIT_V" in trial and trial["QV_SPLIT_V"] == True) or (
            "QV_SPLIT_Q" in trial and trial["QV_SPLIT_Q"] == True):
        space.update({"lr_r": hp.uniform("lr_r", 0.00001, 0.005)})

    print("Starting TPE optimization...")
    best = fmin(fn=train_model_to_optimize_tpe, space=space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
    trials_results = sorted(trials.results, key=lambda x: x['loss'])

    return trials.results


# Checks for each hyperparameter in regular intervals between min and max value of the hyperparameter how the algorithm performs as the hyperparameter value changes.
# Plots hyperparameter value vs performance for each algorithm
# Possibly applies interaction test to test hyperparameter interactions for significance 

# (are the standard tests only for linear interaction?) #TODO: look up
def hyperparameter_interaction_exp(env, algList, hyperparam_dict, number_of_samples=10, runs_per_sample=25,
                                   length_of_tests=50000, max_points=2000, run_metric_percentage=0.1,
                                   run_metric_final_percentage_weight=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plotPath = "Experiments/Plots/Interaction_Tests"
    if not os.path.exists(plotPath):
        os.mkdir(plotPath)
    dataPath = "Experiments/Data"
    if not os.path.exists(dataPath):
        os.mkdir(dataPath)
    trialFolder = dataPath + "/" + str(env) + "_" + str(length_of_tests) + "_" + str(runs_per_sample) + "_" + str(
        number_of_samples)
    if not os.path.exists(trialFolder):
        os.mkdir(trialFolder)

    for hyperparam in hyperparam_dict:
        # Create list of values to test for the current hyperparameter
        hyperparam_value_type = hyperparam_dict[hyperparam][0]
        hyperparam_choices = hyperparam_dict[hyperparam][1:]
        if hyperparam_value_type == 'float':
            min_val = hyperparam_choices[0]
            max_val = hyperparam_choices[1]
            hyperparam_range = max_val - min_val
            step_size = hyperparam_range / number_of_samples
            partition = [min_val + i * step_size for i in range(number_of_samples)]
        elif hyperparam_value_type == 'int':
            min_val = hyperparam_choices[0]
            max_val = hyperparam_choices[1]
            hyperparam_range = max_val - min_val
            step_size = math.ceil(hyperparam_range / number_of_samples)
            partition = [min_val + i * step_size for i in range(math.min(number_of_samples, hyperparam_range))]
        elif hyperparam_value_type == 'cat':
            partition = hyperparam_choices
        partition.sort()

        # Test each alg for this hyperparam
        hyperparam_results = {}
        for alg in algList:
            alg_folder = trialFolder + "/" + alg
            if not os.path.exists(alg_folder):
                os.mkdir(alg_folder)
            alg_results = {}
            for hyperparam_value in partition:
                name = str(hyperparam) + "_" + str(hyperparam_value)
                folder_name = alg_folder + "/" + name
                trial = {"name": name, hyperparam: hyperparam_value}
                result_list, logs = run_trial(env, device, runs_per_sample, length_of_tests, trial,
                                              max_points=max_points)

                store_results([log["Metrics/Reward"] for log in logs], folder_name)

                scores = [run_metric(log["Metrics/Reward"], percentage=run_metric_percentage,
                                     final_percentage_weight=run_metric_final_percentage_weight) for log in logs]
                hyperparam_value_score = np.mean(scores)
                hyperparam_value_std = np.std(scores)
                alg_results[hyperparam_value] = [hyperparam_value_score, hyperparam_value_std]
            hyperparam_results[alg] = alg_results

        # Visualize results:
        for alg, color in zip(hyperparam_results, ['r', 'g', 'b', 'y', 'm']):
            alg_results = hyperparam_results[alg]
            values = [value for value in alg_results]
            means = numpy.array([alg_results[key][0] for key in alg_results])
            stds = numpy.array([alg_results[key][1] for key in alg_results])
            stdEs = stds / math.sqrt(runs_per_sample)

            plt.plot(values, means, label=alg, color=color)
            plt.fill_between(values, means - stdEs, means + stdEs, alpha=0.3, color=color)
            plt.xlabel("Hyperparameter Value")
            plt.ylabel("Score")
            plt.show()

            # TODO: save plot in correct folder
            plt.savefig('Results/Hyperparam Value and Score for paramn' + hyperparam + '.png')

        # TODO: add loading if folder already exists

        apply_stats_to_hyperparam_results(hyperparam_results)  # TODO: interaction test


def opt_checks():
    if optimize == "Marco":
        result_list, logs = run_trial(env, device, number_of_tests, length_of_tests, trial,
                                      randomizeList=randomizedParamList,
                                      max_points=max_points)

        bestResultIdxs = get_best_result_idxs(result_list, 5, run_metric_percentage=run_metric_percentage,
                                              run_metric_final_percentage_weight=run_metric_final_percentage_weight)
        best_params = [randomizedParamList[i] for i in bestResultIdxs]
        initialBestResults = [result_list[i] for i in bestResultIdxs]

        # runs = get_results_for_idx(path, idx)

        resultList_optim, logs_optim = run_trial(env, device, 5, length_of_tests, trial,
                                                 randomizeList=best_params,
                                                 max_points=max_points)
        winnerIdx = getBestResults(resultList_optim, 1)
        winnerParam = randomizedParamList[winnerIdx]
        # winner_original_idx =
        # winnerAdditonalResults = resultList_optim[winnerIdx]

        resultList_winner, logs_winner = run_trial(env, device, 20, length_of_tests, trial,
                                                   randomizeList=[winnerParam],
                                                   max_points=max_points)
        # do hyperparameters optimization stuff
    elif optimize == "tpe":
        trials_data = optimize_tpe(env, device, number_of_tests, length_of_tests, trial,
                                   max_points, number_best_runs=3,
                                   evals_per_optimization_step=evals_per_optimization_step)
        # Select the most promising hyperparameter sets
        trials_results = sorted(trials_data.results, key=lambda x: x['loss'])
        best_params = trials_results[0]['params']

        result_list, logs = run_trial(env, device, final_evaluation_runs, length_of_tests, trial,
                                      hyperparamDict=best_params, max_points=max_points)
    elif optimize == "tpe_best" or optimize == "comet_best":
        # Let TPE run several times, each for number_of_tests runs
        trials_data = []
        for i in range(number_of_hyperparam_optimizations):
            print("Optimization run ", i, "/", number_of_hyperparam_optimizations)
            optimization_experiment = OfflineExperiment(offline_directory="/tmp", project_name="TDEC_BS_optimizations",
                                                        workspace="antonwiehe", log_code=False, log_graph=False,
                                                        auto_param_logging=False, auto_metric_logging=False,
                                                        log_env_details=False, auto_output_logging=None)
            optimization_experiment.add_tag(path + ": " + name)
            if optimize == "tpe_best":
                optimization_results = optimize_tpe(env, device, number_of_tests, length_of_tests, trial,
                                                    max_points, optimization_experiment,
                                                    number_best_runs=number_of_best_runs_to_check,
                                                    evals_per_optimization_step=evals_per_optimization_step,
                                                    optimize_only_lr=optimize_only_lr,
                                                    optimize_only_Q_params=optimize_only_Q_params,
                                                    run_metric_percentage=run_metric_percentage,
                                                    run_metric_final_percentage_weight=run_metric_final_percentage_weight)
            else:
                optimization_results = optimize_comet(env, device, number_of_tests, length_of_tests, trial,
                                                      max_points, optimization_experiment,
                                                      number_best_runs=number_of_best_runs_to_check,
                                                      evals_per_optimization_step=evals_per_optimization_step,
                                                      optimize_only_lr=optimize_only_lr,
                                                      optimize_only_Q_params=optimize_only_Q_params,
                                                      run_metric_percentage=run_metric_percentage,
                                                      run_metric_final_percentage_weight=run_metric_final_percentage_weight)

            print("Optimization complete!")
            print("Best runs: ")
            for i in range(number_of_best_runs_to_check):
                print(optimization_results[i])
            trials_data.append(optimization_results)
        print()
        # Select the most promising hyperparameter sets for each optimization
        best_runs = []
        for i in range(number_of_hyperparam_optimizations):
            trials_results_current_optimization = sorted(trials_data[i], key=lambda x: x['loss'])
            best_runs_current_optimization = [result for result in
                                              trials_results_current_optimization[:number_of_best_runs_to_check]]
            best_runs.extend(best_runs_current_optimization)

        print("Running additional tests to determine the true winner...")
        # Run additional tests for the most promising sets
        max = None
        for run in best_runs:
            params = run["params"]
            previous_score = run["loss"] * -1
            result_list, logs = run_trial(env, device, number_of_checks_best_runs, length_of_tests, trial,
                                          hyperparamDict=params, max_points=max_points)
            scores_of_trials = [run_metric(log["Metrics/Reward"]) for log in logs]
            # Do not append score, as previous_score is very likely an outlier (if only tested once in the objective function)
            # scores_of_trials.append(previous_score)
            score = np.mean(scores_of_trials)
            print("Score:", score)
            if max is None or score > max:
                max = score
                best_params = params
                best_results = (result_list, logs)

        print("Running final experiment:")
        # Run final experiments with the best hyperparameter set
        result_list, logs = run_trial(env, device, final_evaluation_runs, length_of_tests, trial,
                                      hyperparamDict=best_params, max_points=max_points)
        result_list.extend(best_results[0])
        logs.extend(best_results[1])
