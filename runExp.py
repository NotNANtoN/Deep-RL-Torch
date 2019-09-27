from comet_ml import Optimizer, Experiment, OfflineExperiment
from trainer import *
import csv
from cycler import cycler
import os
import time
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope
import copy


def testSetup(env, device, number_of_tests, length_of_tests, trialParams, randomizeList=[], on_server=False,
              max_points=2000, hyperparamDict={}, verbose=True):
    results_len = []
    results = []
    logs = []

    for i in range(number_of_tests):
        if len(randomizeList) > 0:
            randomizedParams = randomizeList[i]
            print("Hyperparameters:")
            for key in randomizedParams:
                print(key + ":", randomizedParams[key], end=" ")
            print()
        else:
            randomizedParams = {}

        trainer = Trainer(env, device, **trialParams, **randomizedParams, **hyperparamDict)
        steps, rewards, log = trainer.run(verbose=False, n_steps=length_of_tests, on_server=on_server)

        rewards = reducePoints(rewards, max_points)
        for key in log:
            log[key] = reducePoints(log[key], max_points)

        results_len.append(len(rewards))
        results.append(rewards)
        logs.append(log)

        if number_of_tests > 1:
            print("Run ", str(i), "/", str(number_of_tests), end=" | ")
            print("Mean reward ", round(np.mean(rewards), 1), end=" ")
            print("Score: ", round(run_metric(log["Total Reward"]), 1))
    return results, logs

def saveList(some_list, path):
    with open(path + ".csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(some_list)


def saveDict(some_dict, path):
    with open(path + ".csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        keys = []
        values = []
        for key in some_dict:
            value = some_dict[key]
            keys.append(key)
            values.append(value)
        types = [type(value) for value in values]
        wr.writerow(keys)
        wr.writerow(values)
        wr.writerow(types)
        
def saveDict_simple(some_dict, path):
    with open(path + ".txt", 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for key in some_dict:
            value = some_dict[key]
            wr.writerow([key + ": " + str(round(value, 7))]) 
 

def readList(path):
    some_list = []
    with open(path, 'r', newline='') as myfile:
        reader = csv.reader(myfile, delimiter=",", quotechar="\"")
        for row in reader:
            return [float(item) for item in row]


def readDict(path):
    some_dict = {}
    with open(path, 'r', newline='') as myfile:
        reader = csv.reader(myfile, delimiter=",", quotechar="\"")
        for idx, row in enumerate(reader):
            if idx == 0:
                keys = [item for item in row]
            elif idx == 1:
                values = [item for item in row]
            elif idx == 2:
                types = [item for item in row]
                for idx, key in enumerate(keys):
                    if types[idx] == "<class 'str'>":
                        typefunc = str
                    elif types[idx] == "<class 'int'>":
                        typefunc = int
                    elif types[idx] == "<class 'float'>":
                        typefunc = float
                    elif types[idx] == "<class 'bool'>":
                        typefunc = bool
                    some_dict[key] = typefunc(values[idx])
                return some_dict


def storeResults(list_of_results, path):
    if not os.path.exists(path):
        os.makedirs(path)

    for idx, result_list in enumerate(list_of_results):
        name = path + str(idx)
        saveList(result_list, name)


def readResults(path, name_list):
    # for file in path. if file is not dir, then it is result, therefore store in list
    dict_of_results = {}
    for alg_name in sorted(os.listdir(path)):
        if alg_name not in name_list:
            continue
        result_list = []
        for run in sorted(os.listdir(path + alg_name)):
            if not os.path.isdir(path + alg_name + "/" + run):
                result = readList(path + alg_name + "/" + run)
                result_list.append(result)
        # Remove the "_done":
        alg_name = alg_name[:-5]
        dict_of_results[alg_name] = result_list
    return dict_of_results


def storeLogs(list_of_log_dicts, path):
    if not os.path.exists(path):
        os.makedirs(path)

    for idx, log_dict_of_run in enumerate(list_of_log_dicts):
        for key in log_dict_of_run:
            log_name = path + key + "/"
            if not os.path.exists(log_name):
                os.makedirs(log_name)
            name = log_name + str(idx)
            log_list = log_dict_of_run[key]
            saveList(log_list, name)


def readLogsNew(path, name_list):
    # get to shape of form {"Epsilon":{"Q":[[1,2,3], [1,1,1]], "Q+TDEC":[[1,2,3], [1,2,1]}, "Loss_TDEC":{"Q+TDEC":[1,2,3]}}
    dict_of_logs = {}
    for alg_name in sorted(os.listdir(path)):
        if alg_name not in name_list:
            continue

        for log_name in sorted(os.listdir(path + alg_name + "/logs/")):
            values = []
            for run_name in sorted(os.listdir(path + alg_name + "/logs/" + log_name + "/")):
                log_of_run = readList(path + alg_name + "/logs/" + log_name + "/" + run_name)
                values.append(log_of_run)

            # Remove the "_done":
            stored_alg_name = alg_name[:-5]
            if log_name in dict_of_logs:
                dict_of_logs[log_name][stored_alg_name] = values
            else:
                dict_of_logs[log_name] = {stored_alg_name: values}
    return dict_of_logs


def storeHyperparameters(hyperparam_list, path):
    if not os.path.exists(path):
        os.makedirs(path)

    for idx, hyperparam_dict in hyperparam_list:
        name = path + str(idx)
        saveDict(hyperparam_dict, name)


def plotDict(dict_of_alg_runs, name, plot_best_idxs=None, path="", max_points=0, length_of_tests=100000,
             totalName=""):
    plt.xlabel("Step")
    plt.ylabel(name)

    print("Plotting log of ", name)

    dict_of_lines_to_plot = dict_of_alg_runs

    prop_cycle = (cycler('linestyle', ['-', '--', ':', '-.']) * cycler('color', ['r', 'g', 'b', 'y', 'm', 'c', 'k']))
    for alg_name, props, count in zip(sorted(dict_of_lines_to_plot), prop_cycle, range(len(dict_of_lines_to_plot))):
        color = props["color"]
        linestyle = props["linestyle"]

        values = dict_of_lines_to_plot[alg_name]

        if plot_best_idxs is not None:
            values = [values[idx] for idx in plot_best_idxs[count]]

        means = np.mean(values, axis=0)
        stdEs = np.std(values, axis=0) / np.sqrt(len(values))

        window_size = len(means) // 10
        if window_size % 2 == 0:
            window_size += 1
        means = meanSmoothing(means, window_size)
        stdEs = meanSmoothing(stdEs, window_size)

        idxs = calculate_reduced_idxs(length_of_tests, len(means))

        plt.plot(idxs, means, label=alg_name, color=color, linestyle=linestyle)
        plt.fill_between(idxs, means - stdEs, means + stdEs, alpha=.25, color=color)

    fileName = name if plot_best_idxs is None else name + "_bestRuns_"
    if name == "Total Reward":
        title = "Rewards per Episode during Training without Exploration"
    else:
        title = name + " Values during Training" if plot_best_idxs is None else name + " Values for best Runs during Training"
    plt.title(title)
    filePath = path + totalName + fileName
    plt.legend()
    plt.savefig(filePath + ".pdf")
    plt.clf()


def generate_log_plots_new(dict_of_logs, path="", max_points=2000, length_of_tests=100000, totalName=""):
    # shape in form of {"Epsilon":{"Q":[[1,2,3], [1,1,1]], "Q+TDEC":[[1,2,3], [1,2,1]}, "Loss_TDEC":{"Q+TDEC":[1,2,3]}}

    path += "logs/"
    if not os.path.exists(path):
        os.makedirs(path)

    for log_name in dict_of_logs:
        alg_dict = dict_of_logs[log_name]
        plotDict(alg_dict, log_name, path=path, max_points=max_points, length_of_tests=length_of_tests,
                 totalName=totalName)


def generatePlot(dict_of_results, drawIQR=False, smoothing="yes", plot_best=10, path="", window_size=15, env="",
                 totalName="", draw_all_lines=False, max_points=0, length_of_tests=100000):
    prop_cycle = (cycler('linestyle', ['-', '--', ':', '-.']) * cycler('color', ['r', 'g', 'b', 'y', 'm', 'c', 'k']))

    for name, props in zip(sorted(dict_of_results), prop_cycle):
        color = props["color"]
        linestyle = props["linestyle"]

        result_list = dict_of_results[name]

        window_size = len(result_list[0]) // 10
        if window_size % 2 == 0:
            window_size += 1
        idxs = calculate_reduced_idxs(length_of_tests, len(result_list[0]))

        if plot_best:
            #best_idxs = np.argpartition(np.mean(result_list, axis=1), -plot_best)[-plot_best:]
            #result_list = [result_list[idx] for idx in best_idxs]
            result_list = getBestResults(result_list, plot_best)

        if draw_all_lines:
            for count, line in enumerate(result_list):
                line = meanSmoothing(line, window_size)
                if count == 0:
                    plt.plot(idxs, line, color=color, linestyle=linestyle, label=name)
                else:
                    plt.plot(idxs, line, color=color, linestyle=linestyle)
            continue

        means = np.mean(result_list, axis=0)
        stds = np.std(result_list, axis=0) / np.sqrt(len(result_list))
        medians = np.median(result_list, axis=0)
        IQRs = np.quantile(result_list, [0.25, 0.75], axis=0)

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
    plt.ylabel("Total Reward")
    # if env == "CartPole-v1":
    #    plt.ylim(0, 350)
    # elif env == "LunarLander-v2":
    #    plt.ylim(-150, 200)
    # elif env == "Acrobot-v1":
    #    plt.ylim(-550, -50)
    # elif env == "MountainCar-v0":
    #    pass
    # plt.ylim(-1500, 0)
    # else:
    #    print("No ylim defined for env ", env)
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
    return steps, list_of_points


def createRandomParamList(number_of_tests, verbose=True):
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


def lemmatize_name(name):
    if name[-5:] == "_done":
        return name[:-5]
    else:
        return name


def alg_in_folder(name, folderName):
    for item in sorted(os.listdir(folderName)):
        if os.path.isdir(folderName + item):
            item_name = lemmatize_name(item)
            if item_name == name:
                return True
    return False


def is_done(name, folderName):
    # assumes that name is in folder
    for item in sorted(os.listdir(folderName)):
        if os.path.isdir(folderName + item):
            item_name = lemmatize_name(item)
            if item_name == name:
                if item[-5:] == "_done":
                    return True
                else:
                    return False


def getBestResults(list_of_results, number, run_metric_percentage=0.1, run_metric_final_percentage_weight=1):
    results = sorted(list_of_results, key=lambda result: run_metric(result, percentage=run_metric_percentage, final_percentage_weight=run_metric_final_percentage_weight), reverse=True)
    return results[:number]


def get_best_result_idxs(list_of_results, number, run_metric_percentage=0.1, run_metric_final_percentage_weight=1):
    best_results = getBestResults(list_of_results, number, run_metric_percentage=run_metric_percentage, run_metric_final_percentage_weight=run_metric_final_percentage_weight)
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
                    result_of_run = readList(path + item + "/" + run)
                    result_list.append(result_of_run)
        else:
            if item[:-4] == str(idx):
                result_of_run = readList(path + item)
                result_list.append(result_of_run)
    return result_list


def store_optimizer_runs(path, runs, idx):
    pass


def train_model_to_optimize_tpe(space, verbose=False, comet_ml=None, run_metric_percentage=0.1, run_metric_final_percentage_weight=1):
    length_of_tests = space.pop("length_of_tests")
    env = space.pop("env")
    device = space.pop("device")
    on_server = space.pop("on_server")
    trial = space.pop("trial")
    max_points = space.pop("max_points")
    trials = space.pop("trials")
    optimization_experiment = space.pop("optimization_experiment")
    evals_per_optimization_step = space.pop("evals_per_optimization_step")

    iteration = len(trials.results)

    performance = train_model_to_optimize(space, length_of_tests, env, device, on_server, trial, max_points, evals_per_optimization_step, iteration, trials=trials, run_metric_percentage=run_metric_percentage, run_metric_final_percentage_weight=run_metric_final_percentage_weight )
    optimization_experiment.log_metric("Performance during optimization", performance * -1)
    optimization_experiment.set_step(iteration)
    return {"loss": performance, 'status': STATUS_OK, "params": space, "iteration": iteration}
    
    
def train_model_to_optimize(hyperparam_dict, length_of_tests, env, device, on_server, trial, max_points, evals_per_optimization_step, iteration, trials=None, verbose=False, run_metric_percentage=0.1, run_metric_final_percentage_weight=1):
    resultList, logs = testSetup(env, device, evals_per_optimization_step, length_of_tests, trial, hyperparamDict=hyperparam_dict,
                                 on_server=on_server, max_points=max_points)
    performances = [run_metric(log["Total Reward"], percentage=run_metric_percentage, final_percentage_weight=run_metric_final_percentage_weight) * -1 for log in logs]
    performance = np.mean(performances)

    print("Run ", str(iteration + 1), " Performance: ", round(performance, 1) * -1)
    if verbose:
        print("Hyperparams:", end=" ")
        for key in hyperparam_dict:
            print(key + str(round(hyperparam_dict[key], 5) if type(hyperparam_dict[key]) is type(1.0) else hyperparam_dict[key]) + "|",  end=" ")
        print()
    return performance

def optimize_comet(env, device, max_evals, length_of_tests, trial, on_server, max_points, optimization_experiment, number_best_runs=1,
                 evals_per_optimization_step=1, optimize_only_lr=False, optimize_only_Q_params=False, run_metric_percentage=0.1, run_metric_final_percentage_weight=1):
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
        
    if not optimize_only_Q_params and ("SPLIT_BELLMAN" in trial and trial["SPLIT_BELLMAN"] == True) or ("QV_SPLIT_V" in trial and trial["QV_SPLIT_V"] == True) or ("QV_SPLIT_Q" in trial and trial["QV_SPLIT_Q"] == True):
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
        #experiment = Experiment("M03EcOc9o9kiG95hws4mq1uqI", project_name="trash", workspace="antonwiehe")

        hyperparamDict = {}
        for key in suggestion:
            hyperparamDict[key] = suggestion[key]

        # Test the model
        score = train_model_to_optimize(hyperparamDict, length_of_tests, env, device, on_server, trial, max_points, evals_per_optimization_step, iteration, run_metric_percentage=run_metric_percentage, run_metric_final_percentage_weight=run_metric_final_percentage_weight)

        optimization_experiment.log_metric("Performance during optimization", score * -1)
        optimization_experiment.set_step(iteration)
        # Report the score back
        suggestion.report_score("accuracy",score)
        
        run = {"loss":score, "params":hyperparamDict}
        trial_results.append(run)
    return trial_results
        
  

def optimize_tpe(env, device, max_evals, length_of_tests, trial, on_server, max_points, optimization_experiment, number_best_runs=1,
                 evals_per_optimization_step=1, optimize_only_lr=False, optimize_only_Q_params=False, run_metric_percentage=0.1, run_metric_final_percentage_weight=1):
    trials = Trials()

    space = {"env": env, "device": device, "on_server": on_server, "trial": trial, "max_points": max_points,
                 "length_of_tests": length_of_tests, "trials": trials, "optimization_experiment":optimization_experiment,
                 "evals_per_optimization_step": evals_per_optimization_step,
                 "lr_Q": hp.uniform("lr_Q", 0.00001, 0.005),
                 "run_metric_percentage":run_metric_percentage, 
                 "run_metric_final_percentage_weight":run_metric_final_percentage_weight
                 }
    if not optimize_only_lr:
        space.update({#"activation_function": hp.choice("activation_function", ["sigmoid", "relu", "elu", "selu"]),
                 #"gamma_Q": hp.uniform("gamma_Q", 0.9, 0.999),
                 "target_network_steps": scope.int(hp.quniform("target_network_steps", 5, 5000, 1)),
                 #"hidden_neurons": scope.int(hp.quniform("hidden_neurons", 64, 128, 1)),
                 #"hidden_layers": scope.int(hp.quniform("hidden_layers", 1, 2, 1)),
                 "batch_size": scope.int(hp.quniform("batch_size", 16, 256, 1)),
                 #"replay_buffer_size": scope.int(hp.quniform("replay_buffer_size", 1024, 50000, 1)),
                 "epsilon_mid": hp.uniform("epsilon_mid", 0.001, 0.25)
                 })
                 
    if not optimize_only_Q_params and ("SPLIT_BELLMAN" in trial and trial["SPLIT_BELLMAN"] == True) or ("QV_SPLIT_V" in trial and trial["QV_SPLIT_V"] == True) or ("QV_SPLIT_Q" in trial and trial["QV_SPLIT_Q"] == True):
        space.update({"lr_r": hp.uniform("lr_r", 0.00001, 0.005)})

    print("Starting TPE optimization...")
    best = fmin(fn=train_model_to_optimize_tpe, space=space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
    trials_results = sorted(trials.results, key=lambda x: x['loss'])

    return trials.results

# Checks for each hyperparameter in regular intervals between min and max value of the hyperparameter how the algorithm performs as the hyperparameter value changes. 
# Plots hyperparameter value vs performance for each algorithm
# Possibly applies interaction test to test hyperparameter interactions for significance 

#(are the standard tests only for linear interaction?) #TODO: look up
def hyperparameter_interaction_exp(env, algList, hyperparam_dict, number_of_samples=10, runs_per_sample=25, length_of_tests=50000, on_server=True, max_points=2000, run_metric_percentage=0.1, run_metric_final_percentage_weight=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    plotPath = "Experiments/Plots/Interaction_Tests"
    if not os.path.exists(plotPath):
        os.mkdir(plotPath)
    dataPath = "Experiments/Data"
    if not os.path.exists(dataPath):
        os.mkdir(dataPath)
    trialFolder = dataPath + "/" + str(env) + "_" + str(length_of_tests) + "_" + str(runs_per_sample) + "_" + str(number_of_samples)
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
                trial = {"name": name, hyperparam:hyperparam_value}
                result_list, logs = testSetup(env, device, runs_per_sample, length_of_tests, trial,
                                                 on_server=on_server, max_points=max_points)

                
                store_results([log["Total Reward"] for log in logs], folder_name)
                
                scores = [run_metric(log["Total Reward"], percentage=run_metric_percentage, final_percentage_weight=run_metric_final_percentage_weight) for log in logs]
                hyperparam_value_score = np.mean(scores)
                hyperparam_value_std = np.std(scores)
                alg_results[hyperparam_value] = [hyperparam_value_score, hyperparam_value_std]
            hyperparam_results[alg] = alg_results
        
        # Visualize results:
        for alg, color in zip(hyperparam_results, ['r','g','b','y','m']):
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

            #TODO: save plot in correct folder
            plt.savefig('Results/Hyperparam Value and Score for paramn' + hyperparam + '.png')
            
        #TODO: add loading if folder already exists
        
        apply_stats_to_hyperparam_results(hyperparam_results) #TODO: interaction test

        
def runExp(env, algList, number_of_tests=20, length_of_tests=600, window_size=None, randomizeParams=False,
           path="", on_server=False, max_points=2000, optimize="no", number_of_best_runs_to_check=5,
           number_of_checks_best_runs=5, final_evaluation_runs=20, number_of_hyperparam_optimizations=2,
           evals_per_optimization_step=2, optimize_only_lr=False, optimize_only_Q_params=False, run_metric_percentage=0.1, run_metric_final_percentage_weight=0.5):
    experiment_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Experiment on ", env)
    
    algList = copy.deepcopy(algList)

    if window_size is None:
        window_size = length_of_tests // 10
        if window_size % 2 == 0:
            window_size += 1

    root = "Experiments/"
    folderName = root + "Data/" + env + "_" + str(number_of_tests) + "_"
    if optimize != "no":
        folderName += "optimize" + optimize + "_"
        if optimize[-4:] == "best":
            folderName += str(number_of_best_runs_to_check) + "_" + str(number_of_checks_best_runs) + "_" \
                     + str(final_evaluation_runs) + "_" + str(number_of_hyperparam_optimizations) + "_" \
                     + str(evals_per_optimization_step) + "_" + str(run_metric_percentage) + "_" +  str(run_metric_final_percentage_weight) + "_"
            if optimize_only_lr:
                folderName += "lr_opt_"
            if optimize_only_Q_params:
                folderName += "no_occams_opt_"
                
    elif randomizeParams:
        folderName += "randomize_"
    folderName += str(length_of_tests) + "/"
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    plotsPath = root + "Plots/"
    if not os.path.exists(plotsPath):
        os.makedirs(plotsPath)
    if not os.path.exists(plotsPath + path):
        os.makedirs(plotsPath + path)

    randomizedParamList = []
    if randomizeParams:
        randomizedParamList = createRandomParamList(number_of_tests, verbose=False)

    totalName = folderName
    name_list = []
    for trial in algList:
        trial_start_time = time.time()

        name = trial.pop("name")
        if name not in name_list:
            name_list.append(name)
        totalName += name + "_"
        trialFolder = folderName + name + "_done/"
        incompleteTrialFolder = folderName + name + "/"


        if alg_in_folder(name, folderName):
            if is_done(name, folderName):
                print(name + " trial was done in a previous setup...")
                print()
                continue
            else:
                # should skip to the next trial, but also return to load this trial later.
                algList.append(trial)
                if algList.count(trial) > 2:
                    os.rmdir(folderName + name)
                    print("Deleting the unfinished folder of ", name, "...")
                else:
                    print("Skipping ", name, " for now, as another process runs a trial for it.")
                trial["name"] = name
                continue
        else:
            print("Testing " + name + "...")
            if not os.path.exists(incompleteTrialFolder):
                os.makedirs(incompleteTrialFolder)

            if optimize == "Marco":
                result_list, logs = testSetup(env, device, number_of_tests, length_of_tests, trial,
                                             randomizeList=randomizedParamList, on_server=on_server,
                                             max_points=max_points)

                bestResultIdxs = get_best_result_idxs(result_list, 5, run_metric_percentage=run_metric_percentage, run_metric_final_percentage_weight=run_metric_final_percentage_weight)
                best_params = [randomizedParamList[i] for i in bestResultIdxs]
                initialBestResults = [result_list[i] for i in bestResultIdxs]

                # runs = get_results_for_idx(path, idx)

                resultList_optim, logs_optim = testSetup(env, device, 5, length_of_tests, trial,
                                                         randomizeList=best_params, on_server=on_server,
                                                         max_points=max_points)
                winnerIdx = getBestResults(resultList_optim, 1)
                winnerParam = randomizedParamList[winnerIdx]
                # winner_original_idx =
                # winnerAdditonalResults = resultList_optim[winnerIdx]

                resultList_winner, logs_winner = testSetup(env, device, 20, length_of_tests, trial,
                                                           randomizeList=[winnerParam], on_server=on_server,
                                                           max_points=max_points)
                # do hyperparameters optimization stuff
            elif optimize == "tpe":
                trials_data = optimize_tpe(env, device, number_of_tests, length_of_tests, trial, on_server,
                                           max_points, number_best_runs=3,
                                           evals_per_optimization_step=evals_per_optimization_step)
                # Select the most promising hyperparameter sets
                trials_results = sorted(trials_data.results, key=lambda x: x['loss'])
                best_params = trials_results[0]['params']

                result_list, logs = testSetup(env, device, final_evaluation_runs, length_of_tests, trial,
                                             hyperparamDict=best_params, on_server=on_server, max_points=max_points)
            elif optimize == "tpe_best" or optimize == "comet_best":
                # Let TPE run several times, each for number_of_tests runs
                trials_data = []
                for i in range(number_of_hyperparam_optimizations):
                    print("Optimization run ", i, "/", number_of_hyperparam_optimizations)
                    optimization_experiment = OfflineExperiment(offline_directory="/tmp", project_name="TDEC_BS_optimizations", workspace="antonwiehe", log_code=False, log_graph=False, auto_param_logging=False, auto_metric_logging=False, log_env_details=False, auto_output_logging=None )
                    optimization_experiment.add_tag(path + ": " + name)
                    if optimize == "tpe_best":
                        optimization_results = optimize_tpe(env, device, number_of_tests, length_of_tests, trial, on_server,
                                                            max_points, optimization_experiment, number_best_runs=number_of_best_runs_to_check,
                                                            evals_per_optimization_step=evals_per_optimization_step, optimize_only_lr=optimize_only_lr, optimize_only_Q_params=optimize_only_Q_params, run_metric_percentage=run_metric_percentage, run_metric_final_percentage_weight=run_metric_final_percentage_weight)
                    else:
                        optimization_results = optimize_comet(env, device, number_of_tests, length_of_tests, trial, on_server,
                                                            max_points, optimization_experiment, number_best_runs=number_of_best_runs_to_check,
                                                            evals_per_optimization_step=evals_per_optimization_step, optimize_only_lr=optimize_only_lr, optimize_only_Q_params=optimize_only_Q_params, run_metric_percentage=run_metric_percentage, run_metric_final_percentage_weight=run_metric_final_percentage_weight)
                    
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
                    result_list, logs = testSetup(env, device, number_of_checks_best_runs, length_of_tests, trial,
                                                 hyperparamDict=params, on_server=on_server, max_points=max_points)
                    scores_of_trials = [run_metric(log["Total Reward"]) for log in logs]
                    # Do not append score, as previous_score is very likely an outlier (if only tested once in the objective function)
                    #scores_of_trials.append(previous_score)
                    score = np.mean(scores_of_trials)
                    print("Score:", score)
                    if max is None or score > max:
                        max = score
                        best_params = params
                        best_results = (result_list, logs)

                print("Running final experiment:")
                # Run final experiments with the best hyperparameter set
                result_list, logs = testSetup(env, device, final_evaluation_runs, length_of_tests, trial,
                                             hyperparamDict=best_params, on_server=on_server, max_points=max_points)
                result_list.extend(best_results[0])
                logs.extend(best_results[1])
              
            else:
                result_list, logs = testSetup(env, device, number_of_tests, length_of_tests, trial,
                                             randomizeList=randomizedParamList, on_server=on_server,
                                             max_points=max_points)
            if os.path.exists(incompleteTrialFolder):
                 os.rmdir(incompleteTrialFolder)
            print()

        
        # Print trial stats
        meansPerEpisode = np.mean(result_list, 0)
        overAllFinalPercent = round(run_metric(meansPerEpisode, percentage=run_metric_percentage, final_percentage_weight=run_metric_final_percentage_weight), 2)
        overallEpisodeMean = round(np.mean(meansPerEpisode), 2)
        overallEpisodeStd = round(np.std(meansPerEpisode), 2)
        overallEpisodeMedian = round(np.median(meansPerEpisode), 2)



        test_score_per_training_run = [run_metric(log["Total Reward"], percentage=run_metric_percentage, final_percentage_weight=run_metric_final_percentage_weight) for log in logs]
        test_score_mean = round(np.mean(test_score_per_training_run), 3)
        test_score_std = round(np.std(test_score_per_training_run), 3)

        print("Test mean score:", test_score_mean)
        print("Test std of score:", test_score_std)
        print("Trial score: ", overAllFinalPercent)
        print("Trial mean: ", overallEpisodeMean)
        print("Trial median: ", overallEpisodeMedian)
        print("Trial std: ", overallEpisodeStd)
        print("Trial time: ", time.time() - trial_start_time)
        print()

        storeResults(result_list, trialFolder)
        storeLogs(logs, trialFolder + "logs/")
        
        # Store those hyperparameters in a file:
        saveList([test_score_mean, test_score_std],  plotsPath + path + "Stats" + name)
        saveList(test_score_per_training_run,  plotsPath + path + "Test_Scores" + name)
        if optimize == "tpe_best" or optimize == "comet_best":
            saveDict_simple(best_params, plotsPath + path + "Hyperparams_" + name)
        

    # Read data:
    name_list = sorted(name_list)
    loading_name_list = [name + "_done" for name in name_list]
    dict_of_results = readResults(folderName, loading_name_list)
    dict_of_logs = readLogsNew(folderName, loading_name_list)
    # list_of_hyperparameter_dicts = readHyperparameters(folderName, name_list)

    # Plot data:
    path = plotsPath + path
    if not os.path.exists(path):
        os.makedirs(path)
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
                                         plot_best=plotBestOption, path=path, window_size=window_size, env=env,
                                         draw_all_lines=drawAll, max_points=max_points, totalName=totalName,
                                         length_of_tests=length_of_tests)

    # Plot log plots:
    generate_log_plots_new(dict_of_logs, path=path, max_points=max_points, length_of_tests=length_of_tests,
                           totalName=totalName)

    exp_end_time = time.time() - experiment_start_time
    print("The whole experiment took ", exp_end_time, " seconds.")


lunar = "LunarLander-v2"
cart = "CartPole-v1"
acro = "Acrobot-v1"
mountain = "MountainCar-v0"

paramsQ = {"name": "Q"}
paramsQ1 = {"name": "Q1"}
paramsQ2 = {"name": "Q2"}
paramsQ3 = {"name": "Q3"}
paramsQ4 = {"name": "Q4"}
paramsQ5 = {"name": "Q5"}
paramsQ6 = {"name": "Q6"}
paramsQ7 = {"name": "Q7"}
paramsQ8 = {"name": "Q8"}
paramsQ9 = {"name": "Q9"}
paramsQ10 = {"name": "Q10"}
paramsQ11 = {"name": "Q11"}
paramsQ12 = {"name": "Q12"}
paramsQ13 = {"name": "Q13"}
paramsQ14 = {"name": "Q14"}
paramsQ15 = {"name": "Q15"}
paramsQ16 = {"name": "Q16"}
paramsQ17 = {"name": "Q17"}
paramsQ18 = {"name": "Q18"}
paramsQ19 = {"name": "Q19"}
paramsQ20 = {"name": "Q20"}
paramsQ21 = {"name": "Q21"}
paramsQ22 = {"name": "Q22"}
paramsQ23 = {"name": "Q23"}
paramsQ24 = {"name": "Q24"}
paramsQ25 = {"name": "Q25"}
paramsQ26 = {"name": "Q26"}
paramsQ27 = {"name": "Q27"}
paramsQ28 = {"name": "Q28"}
paramsQ29 = {"name": "Q29"}
paramsQ30 = {"name": "Q30"}

Q_no_exp_rep = {"name": "Q-Online", "USE_EXP_REP": False}

Q_offset_01 = { "name": "Q-Offset:-0.1", "critic_output_offset": -0.1}
Q_offset_1 = { "name": "Q-Offset:-1", "critic_output_offset": -1}
Q_offset_10 = { "name": "Q-Offset:-10", "critic_output_offset": -10}

paramsQ_target_net_250 = {"name": "Q+Target_250", "TARGET_UPDATE": 250}
paramsQ_target_net_100 = {"name": "Q+Target_100", "TARGET_UPDATE": 100}
paramsQ_target_net_50 = {"name": "Q+Target_50", "TARGET_UPDATE": 50}

paramsQV = {"name": "QV", "USE_QV": True}
paramsQV_split_Q = {"name": "QV-Split_Q", "USE_QV": True, "QV_SPLIT_Q": True}
paramsQV_split_V = {"name": "QV-Split_V", "USE_QV": True, "QV_SPLIT_V": True}
paramsQV_split_V_and_Q = {"name": "QV-Split_V_and_Q", "USE_QV": True, "QV_SPLIT_Q": True, "QV_SPLIT_V": True}
paramsQV_NoTargetQ = {"name": "QV-NoTargetQ", "USE_QV": True, "QV_NO_TARGET_Q": True}
paramsQV_split_Q_NoTargetQ = {"name": "QV-Split_Q-NoTargetQ", "USE_QV": True, "QV_SPLIT_Q": True,
                              "QV_NO_TARGET_Q": True}
paramsQV_split_V_NoTargetQ = {"name": "QV-Split_V-NoTargetQ", "USE_QV": True, "QV_SPLIT_V": True,
                              "QV_NO_TARGET_Q": True}
paramsQV_split_V_and_Q_NoTargetQ = {"name": "QV-Split_V_and_Q-NoTargetQ", "USE_QV": True, "QV_SPLIT_Q": True,
                                    "QV_SPLIT_V": True, "QV_NO_TARGET_Q": True}

paramsEps_2 = {"name": "EpsMid 0.2", "EPS_MID": 0.2}
paramsEps_1 = {"name": "EpsMid 0.1", "EPS_MID": 0.1}
paramsEps_05 = {"name": "EpsMid 0.05", "EPS_MID": 0.05}
paramsEps_01 = {"name": "EpsMid 0.01", "EPS_MID": 0.01}
paramsEps_005 = {"name": "EpsMid 0.005", "EPS_MID": 0.005}
paramsEps_001 = {"name": "EpsMid 0.001", "EPS_MID": 0.001}
paramsEps_0001 = {"name": "EpsMid 0.0001", "EPS_MID": 0.0001}

# QV_CURIOSITY_MID 0.01

paramsQVCAbs1 = {"name": "QVC-Abs-1", "USE_QV": True, "QV_CURIOSITY_ENABLED": True, "QV_CURIOSITY_SCALE": 1,
                 "QV_CURIOSITY_ERROR_FUNC": "absolute"}
paramsQVCAbs0_75 = {"name": "QVC-Abs-0_75", "USE_QV": True, "QV_CURIOSITY_ENABLED": True, "QV_CURIOSITY_SCALE": 0.75,
                    "QV_CURIOSITY_ERROR_FUNC": "absolute"}
paramsQVCAbs0_5 = {"name": "QVC-Abs-0_5", "USE_QV": True, "QV_CURIOSITY_ENABLED": True, "QV_CURIOSITY_SCALE": 0.5,
                   "QV_CURIOSITY_ERROR_FUNC": "absolute"}
paramsQVCAbs0_25 = {"name": "QVC-Abs-0_25", "USE_QV": True, "QV_CURIOSITY_ENABLED": True, "QV_CURIOSITY_SCALE": 0.25,
                    "QV_CURIOSITY_ERROR_FUNC": "absolute"}
paramsQVC1 = {"name": "QVC-Diff-1", "USE_QV": True, "QV_CURIOSITY_ENABLED": True, "QV_CURIOSITY_SCALE": 1,
              "QV_CURIOSITY_ERROR_FUNC": "Diff"}
paramsQVC0_75 = {"name": "QVC-Diff-0_75", "USE_QV": True, "QV_CURIOSITY_ENABLED": True, "QV_CURIOSITY_SCALE": 0.75,
                 "QV_CURIOSITY_ERROR_FUNC": "Diff"}
paramsQVC0_5 = {"name": "QVC-Diff-0_5", "USE_QV": True, "QV_CURIOSITY_ENABLED": True, "QV_CURIOSITY_SCALE": 0.5,
                "QV_CURIOSITY_ERROR_FUNC": "Diff"}
paramsQVC0_25 = {"name": "QVC-Diff-0_25", "USE_QV": True, "QV_CURIOSITY_ENABLED": True, "QV_CURIOSITY_SCALE": 0.25,
                 "QV_CURIOSITY_ERROR_FUNC": "Diff"}
paramsQVC0 = {"name": "QVC-Diff-0", "USE_QV": True, "QV_CURIOSITY_ENABLED": True, "QV_CURIOSITY_SCALE": 0,
              "QV_CURIOSITY_ERROR_FUNC": "Diff"}
paramsQVCAbs0_5_NoTarget = {"name": "QVC-Abs-0_5NoTarget", "USE_QV": True, "QV_CURIOSITY_ENABLED": True,
                            "QV_CURIOSITY_SCALE": 0.5, "QV_CURIOSITY_ERROR_FUNC": "absolute",
                            "QV_CURIOSITY_USE_TARGET_NET": False}
paramsQVCDiff0_5_NoTarget = {"name": "QVC-Diff-0_5NoTarget", "USE_QV": True, "QV_CURIOSITY_ENABLED": True,
                             "QV_CURIOSITY_SCALE": 0.5, "QV_CURIOSITY_ERROR_FUNC": "diff",
                             "QV_CURIOSITY_USE_TARGET_NET": False}

paramsQVCMid0_75 = {"name": "QVC-0.75", "USE_QV": True, "QV_CURIOSITY_ENABLED": True, "QV_CURIOSITY_MID": 0.75,
                    "QV_CURIOSITY_ERROR_FUNC": "absolute"}
paramsQVCMid0_5 = {"name": "QVC-0.5", "USE_QV": True, "QV_CURIOSITY_ENABLED": True, "QV_CURIOSITY_MID": 0.5,
                   "QV_CURIOSITY_ERROR_FUNC": "absolute"}
paramsQVCMid0_1 = {"name": "QVC-0.1", "USE_QV": True, "QV_CURIOSITY_ENABLED": True, "QV_CURIOSITY_MID": 0.1,
                   "QV_CURIOSITY_ERROR_FUNC": "absolute"}
paramsQVCMid0_05 = {"name": "QVC-0.05", "USE_QV": True, "QV_CURIOSITY_ENABLED": True, "QV_CURIOSITY_MID": 0.05,
                    "QV_CURIOSITY_ERROR_FUNC": "absolute"}
paramsQVCMid0_01 = {"name": "QVC-0.01", "USE_QV": True, "QV_CURIOSITY_ENABLED": True, "QV_CURIOSITY_MID": 0.01,
                    "QV_CURIOSITY_ERROR_FUNC": "absolute"}
paramsQVCMid0_001 = {"name": "QVC-0.001", "USE_QV": True, "QV_CURIOSITY_ENABLED": True, "QV_CURIOSITY_MID": 0.001,
                     "QV_CURIOSITY_ERROR_FUNC": "absolute"}
paramsQVCMid0_0001 = {"name": "QVC-0.0001", "USE_QV": True, "QV_CURIOSITY_ENABLED": True, "QV_CURIOSITY_MID": 0.0001,
                      "QV_CURIOSITY_ERROR_FUNC": "absolute"}

paramsAddHidden = {"name": "AddHidden", "SPLIT_BELLMAN": True, "SPLIT_BELL_add_hidden_layer": True}
paramsUseSep = {"name": "SeparateNets", "SPLIT_BELLMAN": True, "SPLIT_BELL_use_separate_nets": True}
paramsIndividualHidden = {"name": "Ind. Hidden", "SPLIT_BELLMAN": True,
                          "SPLIT_BELL_additional_individual_hidden_layer": True}
paramsNoTarget_r = {"name": "Q+Split-NoTarget_r", "SPLIT_BELLMAN": True, "SPLIT_BELL_NO_TARGET_r": True}
paramsSplit = {"name": "Q+Split", "SPLIT_BELLMAN": True, "SPLIT_BELL_NO_TARGET_r": False}
paramsAvg = {"name": "Avg for r(s',a)", "SPLIT_BELLMAN": True, "SPLIT_BELL_AVG_r": True}
paramsNoTarget_r_ind_hidden = {"name": "Q+Split-NoTarget_r+Ind.Hidden", "SPLIT_BELLMAN": True, "SPLIT_BELL_NO_TARGET_r": True,
                               "SPLIT_BELL_additional_individual_hidden_layer": True}
paramsNoTarget_r_use_sep = {"name": "Q+Split-NoTarget_r+Sep.Nets", "SPLIT_BELLMAN": True, "SPLIT_BELL_NO_TARGET_r": True,
                            "SPLIT_BELL_use_separate_nets": True}
params_split_ind_hidden = {"name": "Q+Split+Ind.Hidden", "SPLIT_BELLMAN": True,
                               "SPLIT_BELL_additional_individual_hidden_layer": True}
params_split_use_sep = {"name": "Q+Split+Sep.Nets", "SPLIT_BELLMAN": True,
                            "SPLIT_BELL_use_separate_nets": True}
params_Q_2_hidden = {"name": "Q-2Hidden", "hidden_layers": 2}
params_split_2_hidden = {"name": "Q+Split-2Hidden", "hidden_layers": 2}
paramsNoTarget_r_2_hidden = {"name": "Q+Split-NoTarget_r-2Hidden", "hidden_layers": 2}    

params_Q_reward_noise_0_1 = {"name": "Q-RewardNoise0.1", "reward_added_noise_std": 0.1}
params_Q_reward_noise_1 = {"name": "Q-RewardNoise1", "reward_added_noise_std": 1.0}
params_Q_reward_noise_10 = {"name": "Q-RewardNoise10", "reward_added_noise_std": 10.0}
params_split_reward_noise_0_1 = {"name": "Q+Split-RewardNoise0.1", "SPLIT_BELLMAN": True, "reward_added_noise_std": 0.1}
params_split_reward_noise_1 = {"name": "Q+Split-RewardNoise1", "SPLIT_BELLMAN": True, "reward_added_noise_std": 1.0}
params_split_reward_noise_10 = {"name": "Q+Split-RewardNoise10", "SPLIT_BELLMAN": True, "reward_added_noise_std": 10.0}  
paramsNoTarget_r_split_reward_noise_0_1 = {"name": "Q+Split-NoTarget_r-RewardNoise0.1", "SPLIT_BELLMAN": True, "SPLIT_BELL_NO_TARGET_r": True, "reward_added_noise_std": 0.1}
paramsNoTarget_r_split_reward_noise_1 = {"name": "Q+Split-NoTarget_r-RewardNoise1", "SPLIT_BELLMAN": True, "SPLIT_BELL_NO_TARGET_r": True, "reward_added_noise_std": 1.0}
paramsNoTarget_r_split_reward_noise_10 = {"name": "Q+Split-NoTarget_r-RewardNoise10", "SPLIT_BELLMAN": True, "SPLIT_BELL_NO_TARGET_r": True, "reward_added_noise_std": 10.0}                         

                            
paramsNoTargetAtAll = {"name": "No Target at all", "SPLIT_BELLMAN": True, "SPLIT_BELL_NO_TARGET_AT_ALL": True}
paramsNoTargetAtAll_ind_hidden = {"name": "No Target at all+Ind. Hidden", "SPLIT_BELLMAN": True,
                                  "SPLIT_BELL_NO_TARGET_AT_ALL": True,
                                  "SPLIT_BELL_additional_individual_hidden_layer": True}

paramsLimitEps15 = {"name": "Q+Limit15", "MAX_EPISODE_STEPS": 15}
paramsLimitEps30 = {"name": "Q+Limit30", "MAX_EPISODE_STEPS": 30}
paramsLimitEps60 = {"name": "Q+Limit60", "MAX_EPISODE_STEPS": 60}
paramsLimitEps100 = {"name": "Q+Limit100", "MAX_EPISODE_STEPS": 100}
paramsLimitEps200 = {"name": "Q+Limit200", "MAX_EPISODE_STEPS": 200}
paramsLimitEps500 = {"name": "Q+Limit500", "MAX_EPISODE_STEPS": 500}
paramsLimitEps15Split = {"name": "Q+Split+Limit15", "MAX_EPISODE_STEPS": 15, "SPLIT_BELLMAN": True}
paramsLimitEps30Split = {"name": "Q+Split+Limit30", "MAX_EPISODE_STEPS": 30, "SPLIT_BELLMAN": True}
paramsLimitEps60Split = {"name": "Q+Split+Limit60", "MAX_EPISODE_STEPS": 60, "SPLIT_BELLMAN": True}
paramsLimitEps100Split = {"name": "Q+Split+Limit100", "MAX_EPISODE_STEPS": 100, "SPLIT_BELLMAN": True}
paramsLimitEps200Split = {"name": "Q+Split+Limit200", "MAX_EPISODE_STEPS": 200, "SPLIT_BELLMAN": True}
paramsLimitEps500Split = {"name": "Q+Split+Limit500", "MAX_EPISODE_STEPS": 500, "SPLIT_BELLMAN": True}

paramsQVC = {"name": "QVC", "USE_QV": True, "QV_CURIOSITY_ENABLED": True}
paramsQVC_abs = {"name": "QVC_abs", "USE_QV": True, "QV_CURIOSITY_ENABLED": True, "QVC_TRAIN_ABS_TDE": True}
paramsQVC_two_heads = {"name": "QVC_2heads", "USE_QV": True, "QV_CURIOSITY_ENABLED": True,
                       "QV_CURIOSITY_TWO_HEADS": True}
paramsQVC_two_heads_absTrain = {"name": "QVC_2heads_abs-train", "USE_QV": True, "QV_CURIOSITY_ENABLED": True,
                                "QVC_TRAIN_ABS_TDE": True, "QV_CURIOSITY_TWO_HEADS": True}
paramsQVC_two_heads_absAct_0 = {"name": "QVC_2heads_abs-act_scale 0", "USE_QV": True, "QV_CURIOSITY_ENABLED": True,
                                "QVC_USE_ABS_FOR_ACTION": True, "QV_CURIOSITY_TWO_HEADS": True, "QV_CURIOSITY_SCALE": 0}
paramsQVC_two_heads_absAct_0_5 = {"name": "QVC_2heads_abs-act_scale 0.5", "USE_QV": True, "QV_CURIOSITY_ENABLED": True,
                                  "QVC_USE_ABS_FOR_ACTION": True, "QV_CURIOSITY_TWO_HEADS": True,
                                  "QV_CURIOSITY_SCALE": 0.5}
paramsQVC_two_heads_absAct_0_99 = {"name": "QVC_2heads_abs-act_scale 0.99", "USE_QV": True,
                                   "QV_CURIOSITY_ENABLED": True, "QVC_TRAIN_ABS_TDE": True,
                                   "QV_CURIOSITY_TWO_HEADS": True, "QV_CURIOSITY_SCALE": 0.99}
paramsQVC_two_heads_absAct_0_9 = {"name": "QVC_2heads_abs-act_scale 0.9", "USE_QV": True, "QV_CURIOSITY_ENABLED": True,
                                  "QVC_TRAIN_ABS_TDE": True, "QV_CURIOSITY_TWO_HEADS": True, "QV_CURIOSITY_SCALE": 0.9}
paramsQVC_two_heads_absAct_0_75 = {"name": "QVC_2heads_abs-act_scale 0.75", "USE_QV": True,
                                   "QV_CURIOSITY_ENABLED": True, "QVC_TRAIN_ABS_TDE": True,
                                   "QV_CURIOSITY_TWO_HEADS": True, "QV_CURIOSITY_SCALE": 0.75}

paramsQVC_scale_0_1 = {"name": "QVC_2heads_abs_scale_0_1", "USE_QV": True, "QV_CURIOSITY_ENABLED": True,
                       "QVC_TRAIN_ABS_TDE": True, "QV_CURIOSITY_TWO_HEADS": True, "QV_CURIOSITY_SCALE": 0.1}
paramsQVC_mid_0_1 = {"name": "QVC_2heads_abs_mid_0_1", "USE_QV": True, "QV_CURIOSITY_ENABLED": True,
                     "QVC_TRAIN_ABS_TDE": True, "QV_CURIOSITY_TWO_HEADS": True, "QV_CURIOSITY_MID": 0.1}

TDEC_pure = {"name": "Q+TDEC", "TDEC_ENABLED": True}
TDEC_pure_offset_01 = {"name": "Q+TDEC-Offset:-0.1", "TDEC_ENABLED": True, "critic_output_offset": -0.1}
TDEC_pure_offset_1 = {"name": "Q+TDEC-Offset:-1", "TDEC_ENABLED": True, "critic_output_offset": -1}
TDEC_pure_offset_10 = {"name": "Q+TDEC-Offset:-10", "TDEC_ENABLED": True, "critic_output_offset": -10}

TDEC_gamma_0_8 = {"name": "Q+TDEC-Gamma0.8", "TDEC_ENABLED": True, "TDEC_GAMMA": 0.8}
TDEC_no_target = {"name": "Q+TDEC-NoTarget", "TDEC_ENABLED": True, "TDEC_USE_TARGET_NET": False}
TDEC_mid = {"name": "Q+TDEC-DecayEps", "TDEC_ENABLED": True, "TDEC_MID": 0.1}
TDEC_abs_act = {"name": "Q+TDEC-absAct", "TDEC_ENABLED": True, "TDEC_ACT_FUNC": "absolute"}

TDEC_abs_train = {"name": "Q+TDEC-absTrain", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute"}
TDEC_mse_train = {"name": "Q+TDEC-mseTrain", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "mse"}
TDEC_abs_train_no_target = {"name": "Q+TDEC-absTrain-NoTarget", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute",
                            "TDEC_USE_TARGET_NET": False}
TDEC_abs_train_0_9 = {"name": "Q+TDEC-absTrain-Scale0.9", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute",
                      "TDEC_SCALE": 0.9}
TDEC_abs_train_0_1 = {"name": "Q+TDEC-absTrain-Scale0.1", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute",
                      "TDEC_SCALE": 0.1}
TDEC_abs_train_mid_0_9 = {"name": "Q+TDEC-absTrain-Mid0.9", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute",
                          "TDEC_MID": 0.9}
TDEC_abs_train_mid_0_5 = {"name": "Q+TDEC-absTrain-Mid0.5", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute",
                          "TDEC_MID": 0.5}
TDEC_abs_train_mid_0_2 = {"name": "Q+TDEC-absTrain-Mid0.2", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute",
                          "TDEC_MID": 0.2}
TDEC_abs_train_mid_0_1 = {"name": "Q+TDEC-absTrain-Mid0.1", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute",
                          "TDEC_MID": 0.1}
TDEC_abs_train_mid_0_01 = {"name": "Q+TDEC-absTrain-Mid0.01", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "absolute",
                           "TDEC_MID": 0.01}

TDEC_pos_act = {"name": "Q+TDEC-posAct", "TDEC_ENABLED": True, "TDEC_ACT_FUNC": "positive"}
TDEC_pos_train = {"name": "Q+TDEC-posTrain", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "positive"}
TDEC_pos_train_no_target = {"name": "Q+TDEC-posTrain-NoTarget", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "positive",
                            "TDEC_USE_TARGET_NET": False}
TDEC_pos_train_decay = {"name": "Q+TDEC-posTrain-DecayEps", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "positive",
                        "TDEC_MID": 0.1}
TDEC_pos_train_no_target_decay = {"name": "Q+TDEC-posTrain-NoTarget-DecayEps", "TDEC_ENABLED": True,
                                  "TDEC_TRAIN_FUNC": "positive", "TDEC_MID": 0.1, "TDEC_USE_TARGET_NET": False}
TDEC_pos_train_no_target = {"name": "Q+TDEC-posTrain-NoTarget", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "positive",
                            "TDEC_USE_TARGET_NET": False}
TDEC_pos_train_0_9 = {"name": "Q+TDEC-posTrain-Scale0.9", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "positive",
                      "TDEC_SCALE": 0.9}
TDEC_pos_train_1 = {"name": "Q+TDEC-posTrain-Scale1", "TDEC_ENABLED": True, "TDEC_TRAIN_FUNC": "positive",
                    "TDEC_SCALE": 1.0}

# Always normalize from now on
#Q_normalized = {"name": "Q-normalizedObs", "normalize_observations": True}
#normalized_test_Q = [paramsQ, Q_normalized]

reliabilityTest = [paramsQ, paramsQ1, paramsQ2, paramsQ3]

reliabilityTest_long = [paramsQ, paramsQ1, paramsQ2, paramsQ3, paramsQ4, paramsQ5, paramsQ6, paramsQ7, paramsQ8, paramsQ9, paramsQ10, paramsQ11, paramsQ12, paramsQ13, paramsQ14, paramsQ15, paramsQ16, paramsQ17, paramsQ18, paramsQ19, paramsQ20, paramsQ21, paramsQ22, paramsQ23, paramsQ24, paramsQ25, paramsQ26, paramsQ27, paramsQ28, paramsQ29, paramsQ30]

epsilonList = [paramsEps_2, paramsEps_1, paramsEps_05, paramsEps_01, paramsEps_005, paramsEps_001, paramsEps_0001]
splitBellList = [paramsQ, paramsSplit, paramsNoTarget_r, paramsNoTarget_r_ind_hidden, params_split_use_sep, params_split_ind_hidden, paramsNoTarget_r_use_sep]       

checkAddedParamsInfluenceList = [paramsQ, paramsSplit, paramsNoTarget_r, paramsNoTarget_r_ind_hidden, params_split_use_sep, params_split_ind_hidden, paramsNoTarget_r_use_sep, params_Q_2_hidden, params_split_2_hidden, paramsNoTarget_r_2_hidden]              

noisyRewardList = [params_Q_reward_noise_0_1, params_Q_reward_noise_1, params_Q_reward_noise_10, params_split_reward_noise_0_1, params_split_reward_noise_1, params_split_reward_noise_10, paramsNoTarget_r_split_reward_noise_0_1, paramsNoTarget_r_split_reward_noise_1, paramsNoTarget_r_split_reward_noise_10]

splitShortList = [paramsQ, paramsSplit, paramsNoTarget_r]
QV_split = [paramsQ, paramsQV, paramsQV_split_V, paramsQV_split_Q, paramsQV_split_V_and_Q]
QV_no_target = [paramsQV, paramsQV_NoTargetQ, paramsQV_split_V_and_Q, paramsQV_split_V_and_Q_NoTargetQ]

Q_list = [paramsQ]

TDEC_basic = [paramsQ, TDEC_pure, TDEC_abs_train, TDEC_pos_train, TDEC_mse_train]
TDEC_scaling_abs = [paramsQ, TDEC_abs_train, TDEC_abs_train_0_9, TDEC_abs_train_0_1, TDEC_abs_train_mid_0_9,
                    TDEC_abs_train_mid_0_5, TDEC_abs_train_mid_0_2, TDEC_abs_train_mid_0_1,
                    TDEC_abs_train_mid_0_01]  # ....
TDEC_noTargets = [paramsQ, TDEC_pure, TDEC_no_target, TDEC_abs_train, TDEC_abs_train_no_target, TDEC_pos_train,
                  TDEC_pos_train_no_target]
TDEC_train_or_act = [paramsQ, TDEC_pure, TDEC_abs_train, TDEC_abs_act, TDEC_pos_train, TDEC_pos_act]

TDEC_list = [paramsQ, TDEC_pure, TDEC_no_target, TDEC_mid, TDEC_abs_act, TDEC_abs_train, TDEC_pos_act, TDEC_pos_train]

TDEC_smart_list = [paramsQ, TDEC_pure, TDEC_abs_train, TDEC_pos_train, TDEC_pos_train_no_target, TDEC_pos_train_0_9,
                   TDEC_pos_train_1]
                   
TDEC_mse = [TDEC_mse_train]

QV_list = [paramsQ, paramsQV, paramsQV_NoTargetQ]

Q_offset_list = [paramsQ, Q_offset_01, Q_offset_1, Q_offset_10]
Q_no_exp_rep_list = [paramsQ, Q_no_exp_rep]
TDEC_offset_list = [TDEC_pure, TDEC_pure_offset_01, TDEC_pure_offset_1, TDEC_pure_offset_10]

tracemalloc.start()

# Test:
#runExp(cart, Q_list, number_of_tests=5, length_of_tests=100, path="test/", on_server=True, optimize="comet_best", run_metric_percentage=1, run_metric_final_percentage_weight=0)

############ Exps:
#runExp(cart, reliabilityTest_long, number_of_tests=50, length_of_tests=50000, path="long_rel_test_opt", on_server=True, optimize="comet_best")
#runExp(cart, reliabilityTest_long, number_of_tests=25, length_of_tests=50000, path="long_rel_test_opt", on_server=True, optimize="no")




#runExp(cart, QV_list, number_of_tests=50, length_of_tests=50000, path="QV_cart/", on_server=True, optimize="comet_best")
#runExp(lunar, QV_list, number_of_tests=50, length_of_tests=50000, path="QV_lunar/", on_server=True, optimize="comet_best")

#runExp(cart, TDEC_basic, number_of_tests=50, length_of_tests=50000, path="TDEC_basic_cart/", on_server=True, optimize="comet_best")
#runExp(lunar, TDEC_basic, number_of_tests=50, length_of_tests=50000, path="TDEC_basic_lunar/", on_server=True, optimize="comet_best")

# Bellman Split basics:
#runExp(cart, splitShortList, number_of_tests=50, length_of_tests=50000, path="split_cart/", on_server=True, optimize="comet_best")
#runExp(lunar, splitShortList, number_of_tests=50, length_of_tests=50000, path="split_lunar/", on_server=True, optimize="comet_best")
# Optimize for separate nets and for individual hidden layers
#runExp(cart, splitBellList, number_of_tests=50, length_of_tests=50000, path="split_netArch_cart/", on_server=True, optimize="comet_best")
#runExp(lunar, splitBellList, number_of_tests=50, length_of_tests=50000, path="split_netArch_lunar/", on_server=True, optimize="comet_best")
# Check if added params for different architecture is the cause behind performance boost:
runExp(cart, checkAddedParamsInfluenceList, number_of_tests=50, length_of_tests=50000, path="split_AddedParamsEffect_cart/", on_server=True, optimize="comet_best")
#runExp(lunar, checkAddedParamsInfluenceList, number_of_tests=50, length_of_tests=50000, path="split_AddedParamsEffect_lunar/", on_server=True, optimize="comet_best")
# Check if spit approaches are more robuts to Gaussian noise (they should be in theory):
#runExp(cart, noisyRewardList, number_of_tests=50, length_of_tests=50000, path="split_GaussianNoise_cart/", on_server=True, optimize="comet_best")
#runExp(lunar, noisyRewardList, number_of_tests=50, length_of_tests=50000, path="split_GaussianNoise_lunar/", on_server=True, optimize="comet_best")
# Optimize only lr for split
#runExp(cart, splitShortList, number_of_tests=50, length_of_tests=50000, path="split_optOnlyLr_cart/", on_server=True, optimize="comet_best", optimize_only_lr=True)
#runExp(lunar, splitShortList, number_of_tests=50, length_of_tests=50000, path="split_optOnlyLr_lunar/", on_server=True, optimize="comet_best", optimize_only_lr=True)
# Optimize only Q params for split:
#runExp(cart, splitShortList, number_of_tests=50, length_of_tests=50000, path="split_optOnlyQparams_cart/", on_server=True, optimize="comet_best", optimize_only_Q_params=True)
#runExp(lunar, splitShortList, number_of_tests=50, length_of_tests=50000, path="split_optOnlyQparams_lunar/", on_server=True, optimize="comet_best", optimize_only_Q_params=True)





#runExp(cart, Q_no_exp_rep_list, number_of_tests=50, length_of_tests=50000, path="no_exp_rep_cart/", on_server=True, optimize="comet_best")

#runExp(lunar, TDEC_offset_list, number_of_tests=50, length_of_tests=50000, path="Q_offsets_lunar/", on_server=True, optimize="comet_best")





############ How reliable given same hyperparameters?
#runExp(cart, reliabilityTest, number_of_tests=10, length_of_tests=50000, path="no_optimization_10/", on_server=True, optimize="no")

#runExp(cart, reliabilityTest, number_of_tests=15, length_of_tests=50000, path="no_optimization_15/", on_server=True, optimize="no")

#runExp(cart, reliabilityTest, number_of_tests=20, length_of_tests=50000, path="no_optimization_20/", on_server=True, optimize="no")

#runExp(cart, reliabilityTest, number_of_tests=30, length_of_tests=50000, path="no_optimization_30/", on_server=True, optimize="no")

#runExp(cart, reliabilityTest, number_of_tests=50, length_of_tests=50000, path="no_optimization_50/", on_server=True, optimize="no")

#runExp(cart, reliabilityTest, number_of_tests=100, length_of_tests=50000, path="no_optimization_100/", on_server=True, optimize="no")

# still needs to be tested
########### What is the influence of more runs?
#runExp(cart, reliabilityTest, number_of_tests=50, length_of_tests=50000, path="comet_75_sets/", on_server=True, optimize="comet_best", number_of_best_runs_to_check=5, number_of_checks_best_runs=5, final_evaluation_runs=15, number_of_hyperparam_optimizations=1, evals_per_optimization_step=2)

#runExp(cart, reliabilityTest, number_of_tests=50, length_of_tests=50000, path="comet_100_sets/", on_server=True, optimize="comet_best", number_of_best_runs_to_check=5, number_of_checks_best_runs=5, final_evaluation_runs=15, number_of_hyperparam_optimizations=1, evals_per_optimization_step=2)




########## How many optimizations?
#runExp(cart, reliabilityTest, number_of_tests=50, length_of_tests=50000, path="comet_1_optim/", on_server=True, optimize="comet_best", number_of_best_runs_to_check=5, number_of_checks_best_runs=5, final_evaluation_runs=15, number_of_hyperparam_optimizations=1, evals_per_optimization_step=2)

#runExp(cart, reliabilityTest, number_of_tests=50, length_of_tests=50000, path="comet_2_optim/", on_server=True, optimize="comet_best", number_of_best_runs_to_check=5, number_of_checks_best_runs=5, final_evaluation_runs=15, number_of_hyperparam_optimizations=2, evals_per_optimization_step=2)

#runExp(cart, reliabilityTest, number_of_tests=50, length_of_tests=50000, path="comet_3_optim/", on_server=True, optimize="comet_best", number_of_best_runs_to_check=5, number_of_checks_best_runs=5, final_evaluation_runs=15, number_of_hyperparam_optimizations=3, evals_per_optimization_step=2)

#runExp(lunar, splitShortList, number_of_tests=50, length_of_tests=50000, path="split_short_comet_lunar/", on_server=True, optimize="comet_best", number_of_best_runs_to_check=5, number_of_checks_best_runs=5, final_evaluation_runs=15, number_of_hyperparam_optimizations=1, evals_per_optimization_step=2)






# Reliability Experiments:
#Standard:
#runExp(cart, reliabilityTest, number_of_tests=50, length_of_tests=50000, path="reliability_new/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=3, number_of_checks_best_runs=5, final_evaluation_runs=15,
#       number_of_hyperparam_optimizations=3, evals_per_optimization_step=3
#       )
#More runs:

#runExp(cart, reliabilityTest, number_of_tests=100, length_of_tests=3000, path="reliability/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=3, number_of_checks_best_runs=5, final_evaluation_runs=15,
#       number_of_hyperparam_optimizations=1, evals_per_optimization_step=1
#       )


# More evals per optimization:
#runExp(cart, reliabilityTest, number_of_tests=30, length_of_tests=3000, path="reliability_more_evals/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=3, number_of_checks_best_runs=5, final_evaluation_runs=15,
#       number_of_hyperparam_optimizations=1, evals_per_optimization_step=2
#       )
# More optimizations:
#runExp(cart, reliabilityTest, number_of_tests=30, length_of_tests=3000, path="reliability_more_optimizations/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=3, number_of_checks_best_runs=5, final_evaluation_runs=15,
#       number_of_hyperparam_optimizations=2, evals_per_optimization_step=1
#       )
       


# Both of the above:
#runExp(cart, reliabilityTest, number_of_tests=15, length_of_tests=3000, path="reliability_more_evals_and_optimizations/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=3, number_of_checks_best_runs=5, final_evaluation_runs=15,
#       number_of_hyperparam_optimizations=2, evals_per_optimization_step=2
#       )

# Checking if running separate TPEs makes sense or if we can just keep going
#runExp(cart, reliabilityTest, number_of_tests=30, length_of_tests=3000, path="reliability_more_evals_and_more_checks/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=6, number_of_checks_best_runs=5, final_evaluation_runs=15,
#       number_of_hyperparam_optimizations=1, evals_per_optimization_step=2
#       )


#runExp(cart, splitShortList, number_of_tests=50, length_of_tests=50000, path="cart_split/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=3, number_of_checks_best_runs=5, final_evaluation_runs=15
#       )

#runExp(lunar, splitShortList, number_of_tests=50, length_of_tests=100000, path="lunar_split/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=3, number_of_checks_best_runs=5, final_evaluation_runs=15
#       )

#runExp(cart, TDEC_basic, number_of_tests=50, length_of_tests=50000, path="cart_TDEC/", on_server=True, optimize="tpe_best", number_of_best_runs_to_check=3, number_of_checks_best_runs=5, #final_evaluation_runs=15
#       )

#runExp(lunar, TDEC_basic, number_of_tests=50, length_of_tests=100000, path="lunar_TDEC/", on_server=True, optimize="tpe_best",
#       number_of_best_runs_to_check=3, number_of_checks_best_runs=5, final_evaluation_runs=15
#       )










# runExp(cart, TDEC_smart_list, number_of_tests=10, length_of_tests=50000, on_server=True, path="TDEC_smart_cart_2nd_exp/")
# runExp(lunar, TDEC_smart_list, number_of_tests=5, length_of_tests=100000, on_server=True, path="TDEC_smart_lunar/")

# runExp(lunar, QVC_list, number_of_tests=50, length_of_tests=20000, on_server=True, path="QVC_lunar/")

# cartDict = {}
# cartDict.update(runExp(cart, QVC_abs, number_of_tests=50, length_of_tests=20000, on_server=True, path="QVC_abs_cart/"))
# cartDict.update(runExp(cart, QVC_scale, number_of_tests=50, length_of_tests=20000, on_server=True, path="QVC_scale_cart/"))


############################# TDEC Experiments #############################################
####### cart
# runExp(cart, TDEC_scaling_abs, number_of_tests=50, length_of_tests=100000, path="cart_TDEC_abs_scaling/", on_server=True)


# runExp(cart, TDEC_basic, number_of_tests=50, length_of_tests=100000, path="cart_TDEC_basic/", on_server=True)
# runExp(cart, TDEC_noTargets, number_of_tests=50, length_of_tests=100000, path="cart_TDEC_no_targets/",  on_server=True)
# runExp(cart, TDEC_train_or_act, number_of_tests=50, length_of_tests=100000, path="cart_TDEC_train_or_act/",  on_server=True)

####### acro
# runExp(acro, TDEC_scaling_abs, number_of_tests=50, length_of_tests=100000, path="acro_TDEC_abs_scaling/", on_server=True)


# runExp(acro, TDEC_basic, number_of_tests=50, length_of_tests=100000, path="acro_TDEC_basic/", on_server=True)
# runExp(acro, TDEC_noTargets, number_of_tests=50, length_of_tests=100000, path="acro_TDEC_no_targets/", on_server=True)
# runExp(acro, TDEC_train_or_act, number_of_tests=50, length_of_tests=100000, path="acro_TDEC_train_or_act/", on_server=True)

####### mountain
# runExp(mountain, TDEC_scaling_abs, number_of_tests=50, length_of_tests=100000, path="mountain_TDEC_abs_scaling/", on_server=True)

# runExp(mountain, TDEC_basic, number_of_tests=50, length_of_tests=100000, path="mountain_TDEC_basic/", on_server=True)
# runExp(mountain, TDEC_noTargets, number_of_tests=50, length_of_tests=100000, path="mountain_TDEC_no_targets/", on_server=True)
# runExp(mountain, TDEC_train_or_act, number_of_tests=50, length_of_tests=100000, path="mountain_TDEC_train_or_act/", on_server=True)

###### lunar
# runExp(lunar, TDEC_scaling_abs, number_of_tests=50, length_of_tests=100000, path="lunar_TDEC_abs_scaling/", on_server=True)


# runExp(lunar, TDEC_basic, number_of_tests=50, length_of_tests=150000, path="lunar_TDEC_basic/", on_server=True)
# runExp(lunar, TDEC_noTargets, number_of_tests=50, length_of_tests=100000, path="lunar_TDEC_no_targets/", on_server=True)
# runExp(lunar, TDEC_train_or_act, number_of_tests=50, length_of_tests=100000, path="lunar_TDEC_train_or_act/", on_server=True)


############################# Split Experiments #############################################
# cartDict = {}
# cartDict.update(runExp(cart, splitShortList, number_of_tests=50, length_of_tests=100000, path="QsplitExp_cart/", on_server=True, randomizeParams=True))
# cartDict.update(runExp(cart, QV_split, number_of_tests=50, length_of_tests=100000, path="QVsplitExp_cart/", on_server=True, randomizeParams=True))
# cartDict.update(runExp(cart, QV_no_target, number_of_tests=50, length_of_tests=100000, path="QVnoTargetExp_cart/", on_server=True, randomizeParams=True))

# acroDict = {}
# acroDict.update(runExp(acro, splitShortList, number_of_tests=50, length_of_tests=100000, path="QsplitExp_acro/",  on_server=True, randomizeParams=True))
# acroDict.update(runExp(acro, QV_split, number_of_tests=50, length_of_tests=100000, path="QVsplitExp_acro/",  on_server=True, randomizeParams=True))
# acroDict.update(runExp(acro, QV_no_target, number_of_tests=50, length_of_tests=100000, path="QVnoTargetExp_acro/", on_server=True, randomizeParams=True))

# mountainDict = {}
# mountainDict.update(runExp(mountain, splitShortList, number_of_tests=50, length_of_tests=100000,  path="QsplitExp_mountain/", on_server=True, randomizeParams=True))
# mountainDict.update(runExp(mountain, QV_split, number_of_tests=50, length_of_tests=100000,  path="QVsplitExp_mountain/",  on_server=True, randomizeParams=True))
# mountainDict.update(runExp(mountain, QV_no_target, number_of_tests=50, length_of_tests=100000,  path="QVnoTargetExp_mountain/", on_server=True, randomizeParams=True))

# lunarDict = {}
# lunarDict.update(runExp(lunar, splitShortList, number_of_tests=50, length_of_tests=150000, path="QsplitExp_lunar/", on_server=True, randomizeParams=True))
# lunarDict.update(runExp(lunar, QV_split, number_of_tests=50, length_of_tests=150000,  path="QVsplitExp_lunar/",  on_server=True, randomizeParams=True))
# lunarDict.update(runExp(lunar, QV_no_target, number_of_tests=50, length_of_tests=150000,  path="QVnoTargetExp_lunar/", on_server=True, randomizeParams=True))
