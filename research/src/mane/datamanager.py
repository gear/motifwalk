"""data manager
"""
# Coding: utf-8
# File name: mane.py
# Created: 2016-07-19
# Description: Main file to run the model.
## v0.0: File created. Add argparse

import csv
import random

def read_edges_from_txt(filepath, sep=" ", one_origin=False):
    edges = []
    with open(filepath) as f:
        for line in f:
            n1, n2 = line.strip().split(sep)
            edges.append((int(n1), int(n2)))
    if one_origin:
        edges = [(i-1, j-1) for i, j in edges]
    return edges

def read_coms_from_txt(filepath, sep=" ", one_origin=False, index=True):
    coms = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if index:
                n, c = line.strip().split(sep)
            else:
                c = line.strip()
                n = i
            coms.append((int(n), int(c)))
    if one_origin:
        coms = [(i-1, j-1) for i, j in coms]
    return coms


class DataManager(object):

    def __init__(self, edges, labels):
        """
        :param edges: list of tuple (node_id1, node_id2)
        :param labels: list of tuple (node_id, com_id)
        """
        self.edges = edges
        self.ground_truth = {}
        for i, c in labels:
            self.ground_truth.setdefault(i, set())
            self.ground_truth[i].add(c)
        self.num_nodes = len(self.ground_truth)
        self.num_edges = len(self.edges)
        self.unique_ground_truth = self.get_unique_ground_truth()

    def calculate_fscore(self, list_of_com):
        """
        :param list_of_com: list of community ids
        :return: micro_fscore, macro_fscore
        """
        tp = {}
        fn = {}
        fp = {}
        coms = set()
        for n, c in enumerate(list_of_com):
            if n not in self.ground_truth:
                continue
            tp.setdefault(c, 0)
            coms.add(c)
            if c in self.ground_truth[n]:
                tp[c] += 1
            else:
                fp.setdefault(c, 0)
                fp[c] += 1
                d = list(self.ground_truth[n] - set([c]))[0]
                fn.setdefault(d, 0)
                fn[d] += 1
                coms.add(d)
        tp_sum = sum(tp.values())
        fp_sum = sum(fp.values())
        fn_sum = sum(fn.values())
        micro_prec = tp_sum / float(tp_sum + fp_sum)
        micro_recall = tp_sum / float(tp_sum + fn_sum)
        micro_fscore = 2 * micro_prec * micro_recall / (micro_prec + micro_recall)
        macro_fscores = []
        for c in coms:
            macro_prec = tp.get(c, 0) / float(tp.get(c, 0) + fp.get(c, 0) + 10e-9)
            macro_recall = tp.get(c, 0) / float(tp.get(c, 0) + fn.get(c, 0) + 10e-9)
            score = macro_prec * macro_recall / (macro_prec + macro_recall + 10e-9)
            macro_fscores.append(score)
        macro_fscore = 2 * sum(macro_fscores) / float(len(macro_fscores))
        return micro_fscore, macro_fscore

    def get_unique_ground_truth(self):
        gt = {}
        for n, c in self.ground_truth.items():
            gt[n] = list(c)[0]
        return gt

    def export_edges(self, path, sep=" ", one_origin=False):
        edges = self.edges
        if one_origin:
            edges = [(i+1, j+1) for i, j in edges]
        with open(path, "w") as f:
            writer = csv.writer(f, delimiter=sep)
            writer.writerows(edges)

    def export_ground_truth(self, path, sep=" "):
        with open(path, "w") as f:
            writer = csv.writer(f, delimiter=sep)
            writer.writerows(self.unique_ground_truth.items())

    def sample_training_set(self, sample_ratio):
        n_sample = int(self.num_nodes * sample_ratio)
        nodes = list(range(self.num_nodes))
        random.shuffle(nodes)
        training_set = {}
        for n in nodes[:n_sample]:
            c = self.unique_ground_truth[n]
            training_set[n] = c
        return training_set

    def export_training_set(self, path, sample_ratio, sep=" "):
        training_set = self.sample_training_set(sample_ratio)
        with open(path, "w") as f:
            writer = csv.writer(f, delimiter=sep)
            writer.writerows(training_set)
