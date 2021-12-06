#!/usr/bin/env python
# coding: utf-8


from BayesianNetwork import *
from ExactInference import *
from ApproximateInference import *
from datetime import datetime
import matplotlib.pyplot as plt

# global varaibles

T = True
F = False
# Bayes Nets
# AIMA-ALARM example
BN_alarm = BayesNet(
    [
        ("Burglary", [], {(): 0.001}),
        ("Earthquake", [], {(): 0.002}),
        ("Alarm", ["Burglary", "Earthquake"], {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001}),
        ("JohnCalls", ["Alarm"], {(T,): 0.90, (F,): 0.05}),
        ("MaryCalls", ["Alarm"], {(T,): 0.70, (F,): 0.01}),
    ]
)
# our Bayes Net
BN_horse = BayesNet(
    [
        ("SunnyDay", [], {(): 0.7}),
        ("RadioOn", [], {(): 0.6}),
        ("HorseAWins", ["SunnyDay"], {(T,): 0.6, (F,): 0.3}),
        ("HorseBWins", ["SunnyDay"], {(T,): 0.8, (F,): 0.3}),
        ("JohnCalls", ["HorseAWins", "HorseBWins"], {(T, T): 0.9, (T, F): 0.8, (F, T): 0.7, (F, F): 0.4}),
        ("MikeCalls", ["HorseAWins", "HorseBWins"], {(T, T): 0.96, (T, F): 0.7, (F, T): 0.9, (F, F): 0.2}),
        ("AlianKnows", ["JohnCalls"], {(T,): 0.9, (F,): 0.2}),
        ("LilyKnows", ["MikeCalls", "RadioOn"], {(T, T): 0.95, (T, F): 0.8, (F, T): 0.5, (F, F): 0.1}),
    ]
)
# queries
q_alarm = {
    "causal": {"X": "JohnCalls", "e": {"Earthquake": F}},
    "diagnostic": {"X": "Burglary", "e": {"JohnCalls": T}},
    "sanity": {"X": "MaryCalls", "e": {"Alarm": T}},
}
q_horse = {
    "causal": {"X": "AlianKnows", "e": {"SunnyDay": T}},
    "diagnostic": {"X": "HorseBWins", "e": {"LilyKnows": T}},
    "sanity": {"X": "LilyKnows", "e": {"HorseBWins": F, "HorseAWins": F}},
}


def generate_syntax(X, e, bool=True):
    """generate the syntax of a query, e.q. P(SunnyDay=True|AlianKnows=True,LilyKnows=True)"""
    res = "P("
    res += f"{X}=True|" if bool else f"{X}=False|"
    for k, v in e.items():
        res += f"{k}={v},"
    res = res[:-1] + ")"
    return res


def print_res(method, query, p):
    """ print the result of a query"""
    print(f"{method}:")
    print(generate_syntax(query["X"], query["e"]) + f" = {round(p[T], 4)}")
    print(generate_syntax(query["X"], query["e"], F) + f" = {round(p[F], 4)}")
    print("------------------")


# ## Exact Inference


def exact_inference(BN, query):
    """exact inference"""
    # causal reasoning
    causal_q = query["causal"]
    p = enumeration_ask(causal_q["X"], causal_q["e"], BN)
    print_res("causal reasoning", causal_q, p)

    # diagnostic reasoning
    diagnostic_q = query["diagnostic"]
    p = enumeration_ask(diagnostic_q["X"], diagnostic_q["e"], BN)
    print_res("diagnostic reasoning", diagnostic_q, p)

    # sanity check
    sanity_q = query["sanity"]
    p = enumeration_ask(sanity_q["X"], sanity_q["e"], BN)
    print_res("sanity check", sanity_q, p)


exact_inference(BN_alarm, q_alarm)
exact_inference(BN_horse, q_horse)


# ## Approximate Inference


def appro_inference(BN, query, rej_sample_size=10000, gibbs_sample_size=300):
    """ approximate inference """
    X = query["X"]
    e = query["e"]
    p_enum = enumeration_ask(X, e, BN)
    sample = Sample()
    p_rej = sample.rejection_sampling(X, e, BN, rej_sample_size)
    p_gibbs = sample.gibbs_sampling(X, e, BN, gibbs_sample_size)
    print_res("enumeration", query, p_enum)
    print_res("rejection sampling", query, p_rej)
    print_res("gribbs sampling", query, p_gibbs)


appro_inference(BN_alarm, q_alarm["causal"])
appro_inference(BN_alarm, q_alarm["diagnostic"])
appro_inference(BN_alarm, q_alarm["sanity"])
appro_inference(BN_horse, q_horse["causal"])
appro_inference(BN_horse, q_horse["diagnostic"])
appro_inference(BN_horse, q_horse["sanity"])


# ### Convergence test
# - How could you define stability?
#     - abs(last_p-now_p)/last_p < threshold
# - How many samples are needed for rejection sampling to become stable?
# - How many samples are needed for Gibbs sampling to become stable?
# - Do these values depend on the structure of the network or the query itself?
# - Is one algorithm “better” than the other?


def visual(rec_rej, rec_gib, exact_v, title, N_rej, N_gib, x="number of samples", y="estimate by sampling"):
    """ plot function including exact and approximate inference """
    plt.plot(rec_rej[0], rec_rej[1], "r-", alpha=0.6, label="rejection sampling")
    plt.plot(rec_gib[0], rec_gib[1], "b-", alpha=0.6, label="gibbs sampling")
    plt.xlim([10, max(N_rej, N_gib)])
    if y == "estimate by sampling":
        plt.ylim([0, 1]) # ylim for convergence plot
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.axhline(y=exact_v, color="g", alpha=0.2, linestyle="dashed")
    plt.legend()
    plt.show()


# plot the result without convergence test
# sanity_q = q_horse['sanity']
# X = sanity_q['X']
# e = sanity_q['e']

# p = enumeration_ask(X, e, BN_horse)
# conv = {"start_N": 10, "max_N": 10000, "interval": 100, "threshold": 0.01}
# sample = Sample()
# rec_rej = sample.samples(X, e, BN_horse, conv, "rejection")
# rec_gibbs = sample.samples(X, e, BN_horse, conv, "gibbs")
# title = generate_syntax(X, e)
# visual(rec_rej, rec_gibbs, p[T], title, conv)


def test_conv(query, BN, rej_conv, gibbs_conv):
    """ convergence test """
    X = query["X"]
    e = query["e"]

    sample = Sample()
    start_time1 = datetime.now()
    rec_rej, N_rej, res_rej = sample.converge_sampling(X, e, BN, rej_conv, "rejection")
    end_time1 = datetime.now()
    start_time2 = datetime.now()
    rec_gib, N_gib, res_gib = sample.converge_sampling(X, e, BN, gibbs_conv, "gibbs")
    end_time2 = datetime.now()
    exact_v = enumeration_ask(X, e, BN)[T]

    print(f"rejection sampling {res_rej}")
    print(f"total running time is {(end_time1 - start_time1).total_seconds() * 1000} ms")
    print(f"gibbs sampling {res_gib}")
    print(f"total running time is {(end_time2 - start_time2).total_seconds() * 1000} ms")

    title = generate_syntax(X, e)
    visual(rec_rej, rec_gib, exact_v, title, N_rej, N_gib)


# convergence arguments
rej_conv = {"start_N": 10, "max_N": 10000, "interval": 100, "threshold": 0.01}
gibbs_conv = {"start_N": 10, "max_N": 10000, "interval": 100, "threshold": 0.01}

## %%timeit -r5 -n5
# AIAM-ALARM
test_conv(q_alarm["causal"], BN_alarm, rej_conv, gibbs_conv)
test_conv(q_alarm["diagnostic"], BN_alarm, rej_conv, gibbs_conv)
test_conv(q_alarm["sanity"], BN_alarm, rej_conv, gibbs_conv)

# our example
rej_conv = {"start_N": 10, "max_N": 10000, "interval": 100, "threshold": 0.01}
gibbs_conv = {"start_N": 10, "max_N": 10000, "interval": 100, "threshold": 0.01}
test_conv(q_horse["causal"], BN_horse, rej_conv, gibbs_conv)
test_conv(q_horse["diagnostic"], BN_horse, rej_conv, gibbs_conv)
test_conv(q_horse["sanity"], BN_horse, rej_conv, gibbs_conv)


# ### Accuracy test


def test_acc(query, BN, rej_conv, gibbs_conv):
    """ accuracy test """
    X = query["X"]
    e = query["e"]

    exact_v = enumeration_ask(X, e, BN)[T]
    sample = Sample()
    rec_rej, N_rej, res_rej = sample.accuracy_sampling(X, e, BN, rej_conv, "rejection")
    rec_gib, N_gib, res_gib = sample.accuracy_sampling(X, e, BN, gibbs_conv, "gibbs")
    print(f"rejection sampling {res_rej}")
    print(f"gibbs sampling {res_gib}")

    title = generate_syntax(X, e)
    visual(rec_rej, rec_gib, 0, title, N_rej, N_gib, "number of samples", "relative error")  # accuracy


# ### AIMA


# AIMA-ALARM, threshold: 0.005
rej_conv = {"start_N": 10, "max_N": 50000, "interval": 100, "threshold": 0.005}
gibbs_conv = {"start_N": 10, "max_N": 50000, "interval": 100, "threshold": 0.001}
test_acc(q_alarm["causal"], BN_alarm, rej_conv, gibbs_conv)
test_acc(q_alarm["diagnostic"], BN_alarm, rej_conv, gibbs_conv)
test_acc(q_alarm["sanity"], BN_alarm, rej_conv, gibbs_conv)

# AIMA-ALARM, threshold: 0.01
rej_conv = {"start_N": 10, "max_N": 10000, "interval": 100, "threshold": 0.01}
gibbs_conv = {"start_N": 10, "max_N": 10000, "interval": 100, "threshold": 0.01}
test_acc(q_alarm["causal"], BN_alarm, rej_conv, gibbs_conv)
test_acc(q_alarm["diagnostic"], BN_alarm, rej_conv, gibbs_conv)
test_acc(q_alarm["sanity"], BN_alarm, rej_conv, gibbs_conv)

# ### HORSE


# Our example, threshold: 0.01
rej_conv = {"start_N": 10, "max_N": 10000, "interval": 100, "threshold": 0.01}
gibbs_conv = {"start_N": 10, "max_N": 10000, "interval": 100, "threshold": 0.01}
test_acc(q_horse["causal"], BN_horse, rej_conv, gibbs_conv)
test_acc(q_horse["diagnostic"], BN_horse, rej_conv, gibbs_conv)
test_acc(q_horse["sanity"], BN_horse, rej_conv, gibbs_conv)

# Our example, threshold: 0.005
rej_conv = {"start_N": 10, "max_N": 50000, "interval": 100, "threshold": 0.005}
gibbs_conv = {"start_N": 10, "max_N": 50000, "interval": 100, "threshold": 0.01}
test_acc(q_horse["causal"], BN_horse, rej_conv, gibbs_conv)
test_acc(q_horse["diagnostic"], BN_horse, rej_conv, gibbs_conv)
test_acc(q_horse["sanity"], BN_horse, rej_conv, gibbs_conv)

# Our example, threshold: 0.001
rej_conv = {"start_N": 10, "max_N": 50000, "interval": 100, "threshold": 0.001}
gibbs_conv = {"start_N": 10, "max_N": 50000, "interval": 100, "threshold": 0.01}
test_acc(q_horse["causal"], BN_horse, rej_conv, gibbs_conv)
test_acc(q_horse["diagnostic"], BN_horse, rej_conv, gibbs_conv)
test_acc(q_horse["sanity"], BN_horse, rej_conv, gibbs_conv)
