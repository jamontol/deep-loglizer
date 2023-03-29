import os
import io
import itertools
import torch
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
import hashlib
import pickle
import re
import logging
from tqdm import tqdm
#from ml_anomaly.preprocessing.semantic_search import Semantic


from deeploglizer.common.utils import (
    json_pretty_dump,
    dump_pickle,
    load_pickle,
)

def load_vectors(fname):
    logging.info("Loading vectors from {}.".format(fname))

    if fname.endswith("parquet"):
        
        try:
            df_cache = pd.read_parquet(fname)
            df_cache = df_cache[~df_cache.index.duplicated(keep="first")]
            df_cache.drop(['batch_id'], axis = 1, inplace=True)
            df_cache.index = df_cache.index.astype(str)  # fix parquet issue
        except MemoryError:
            log.error(
                " Memory error while reading the cache. Disabling cache: The embedding process will take \
                much longer"
            )
            cache = False

        data = df_cache

    elif fname.endswith("pkl"):
        with open(fname, "rb") as fr:
            data = pickle.load(fr)
    else:
        # load fasttext file
        fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin.readlines()[0:1000]:
            tokens = line.rstrip().split(" ")
            data[tokens[0]] = np.array(list(map(float, tokens[1:])))

    return data


class Vocab:
    def __init__(self, max_token_len, min_token_count, use_tfidf=False):
        self.max_token_len = max_token_len
        self.min_token_count = min_token_count
        self.use_tfidf = use_tfidf
        self.word2idx = {"PADDING": 0, "OOV": 1}
        self.token_vocab_size = None

    def __tokenize_log(self, log):
        word_lst_tmp = re.findall(r"[a-zA-Z]+", log)
        word_lst = []
        for word in word_lst_tmp:
            res = list(filter(None, re.split("([A-Z][a-z][^A-Z]*)", word)))
            if len(res) == 0:
                word_lst.append(word.lower())
            else:
                res = [w.lower() for w in res]
                word_lst.extend(res)
        return word_lst

    def gen_pretrain_matrix(self, pretrain_path, type_embeddings="semantics"):

        logging.info("Generating a pretrain matrix.")

        if type_embeddings == "semantics":

            word_vec_dict = load_vectors(pretrain_path)
            vocab_size = len(self.word2idx)
            pretrain_matrix = np.zeros([vocab_size, 300])
            oov_count = 0
            # print(list(self.word2idx.keys()))
            # exit()
            for word, idx in tqdm(self.word2idx.items()):
                if word in word_vec_dict:
                    pretrain_matrix[idx] = word_vec_dict[word]
                else:
                    oov_count += 1
            logging.info(
                "{}/{} words are assigned pretrained vectors.".format(
                    vocab_size - oov_count, vocab_size
                )
            )

        elif type_embeddings == "sentences":

            sentence_vec_dict = load_vectors(pretrain_path)
            vocab_size = len(self.sentence2idx)
            pretrain_matrix = np.zeros([vocab_size, sentence_vec_dict.shape[1]])
            oov_count = 0
            for sentence, idx in tqdm(self.sentence2idx.items()):
                if sentence in sentence_vec_dict.index:
                    pretrain_matrix[idx] = sentence_vec_dict.loc[sentence]
                else:
                    #TODO search for most similar 
                    oov_count += 1
            logging.info(
                "{}/{} sentences are assigned pretrained vectors.".format(
                    vocab_size - oov_count, vocab_size
                )
            )

        return torch.from_numpy(pretrain_matrix)

    def trp(self, l, n):
        """ Truncate or pad a list """
        r = l[:n]
        if len(r) < n:
            r.extend(list([0]) * (n - len(r)))
        return r

    def build_vocab(self, logs, type_embeddings:str = "semantics"):

        if type_embeddings == "semantics":

            token_counter = Counter()
            for log in logs:
                tokens = self.__tokenize_log(log)
                token_counter.update(tokens)
            valid_tokens = set(
                [
                    word
                    for word, count in token_counter.items()
                    if count >= self.min_token_count
                ]
            )

            self.word2idx.update({word: idx for idx, word in enumerate(valid_tokens, 2)})
            self.token_vocab_size = len(self.word2idx)
        
        elif type_embeddings == "sentences":

            self.sentence2idx = logs #training log2id_train
            self.sentence_vocab_size = len(self.sentence2idx)

    def fit_tfidf(self, total_logs):
        logging.info("Fitting tfidf.")
        self.tfidf = TfidfVectorizer(
            tokenizer=lambda x: self.__tokenize_log(x),
            vocabulary=self.word2idx,
            norm="l1",
        )
        self.tfidf.fit(total_logs)

    def transform_tfidf(self, logs):
        return self.tfidf.transform(logs)

    def logs2idx(self, logs):
        idx_list = []
        for log in logs:
            tokens = self.__tokenize_log(log)
            tokens_idx = self.trp(
                [self.word2idx.get(t, 1) for t in tokens], self.max_token_len
            )
            idx_list.append(tokens_idx)
        return idx_list


class FeatureExtractor(BaseEstimator):
    """
    feature_type: "sequentials", "semantics", "quantitatives"
    window_type: "session", "sliding"
    max_token_len: only used for semantics features
    """

    def __init__(
        self,
        label_type="next_log",  # "none", "next_log", "anomaly"
        feature_type="sequentials",
        eval_type="session",
        window_type="sliding",
        window_size=None,
        stride=None,
        max_token_len=50,
        min_token_count=1,
        pretrain_path=None,
        use_tfidf=False,
        cache=False,
        **kwargs,
    ):
        self.label_type = label_type
        self.feature_type = feature_type
        self.eval_type = eval_type
        self.window_type = window_type
        self.window_size = window_size
        self.stride = stride
        self.pretrain_path = pretrain_path
        self.use_tfidf = use_tfidf
        self.max_token_len = max_token_len
        self.min_token_count = min_token_count
        self.cache = cache
        self.vocab = Vocab(max_token_len, min_token_count)
        self.meta_data = {}

        if cache:
            param_json = self.get_params()
            identifier = hashlib.md5(str(param_json).encode("utf-8")).hexdigest()[0:8]
            self.cache_dir = os.path.join("./cache", identifier)
            os.makedirs(self.cache_dir, exist_ok=True)
            json_pretty_dump(
                param_json, os.path.join(self.cache_dir, "feature_extractor.json")
            )

    def __generate_windows(self, session_dict, stride):
        window_count = 0
        for session_id, data_dict in session_dict.items():
            if self.window_type == "sliding":
                i = 0
                templates = data_dict["templates"]
                template_len = len(templates)
                windows = []
                window_labels = []
                window_anomalies = []
                while i + self.window_size < template_len:
                    window = templates[i: i + self.window_size]
                    next_log = self.log2id_train.get(templates[i + self.window_size], 1)

                    if isinstance(data_dict["label"], list):
                        window_anomaly = int(
                            1 in data_dict["label"][i: i + self.window_size + 1]
                        )
                    else:
                        window_anomaly = data_dict["label"]

                    windows.append(window)
                    window_labels.append(next_log)
                    window_anomalies.append(window_anomaly)
                    i += stride
                else:
                    window = templates[i:-1]
                    window.extend(["PADDING"] * (self.window_size - len(window)))
                    next_log = self.log2id_train.get(templates[-1], 1)

                    if isinstance(data_dict["label"], list):
                        window_anomaly = int(1 in data_dict["label"][i:])
                    else:
                        window_anomaly = data_dict["label"]

                    windows.append(window)
                    window_labels.append(next_log)
                    window_anomalies.append(window_anomaly)
                window_count += len(windows)

                session_dict[session_id]["windows"] = windows
                session_dict[session_id]["window_labels"] = window_labels
                session_dict[session_id]["window_anomalies"] = window_anomalies

                if session_id == "all":
                    logging.info(
                        "Total window number {} ({:.2f})".format(
                            len(window_anomalies),
                            sum(window_anomalies) / len(window_anomalies),
                        )
                    )

            elif self.window_type == "session":
                session_dict[session_id]["windows"] = [data_dict["templates"]]
                session_dict[session_id]["window_labels"] = [data_dict["label"]]
                window_count += 1

        logging.info("{} sliding windows generated.".format(window_count))

    def __windows2quantitative(self, windows):
        total_features = []
        for window in windows:
            feature = [0] * len(self.id2log_train)
            window = [self.log2id_train.get(x, 1) for x in window]
            log_counter = Counter(window)
            for logid, log_count in log_counter.items():
                feature[int(logid)] = log_count
            #total_features.append(feature[1:])  # discard the position of padding
            total_features.append(np.array(feature[1:])[:, np.newaxis])  # discard the position of padding

        return np.array(total_features)

    def __windows2sequential(self, windows):
        total_features = []
        for window in windows:
            ids = [self.log2id_train.get(x, 1) for x in window]
            total_features.append(ids)
        return np.array(total_features)

    def __windows2semantics(self, windows, log2idx):
        # input: raw windows
        # output: encoded token matrix,
        total_idx = [list(map(lambda x: log2idx[x], window)) for window in windows]
        return np.array(total_idx)

    def __windows2sentences(self, windows, log2idx):
        # input: raw windows
        # output: encoded token matrix,
        total_idx = [list(map(lambda x: log2idx[x], window)) for window in windows]
        return np.array(total_idx)

    def save(self):
        logging.info("Saving feature extractor to {}.".format(self.cache_dir))
        with open(os.path.join(self.cache_dir, "est.pkl"), "wb") as fw:
            pickle.dump(self, fw)

    def load(self):
        try:
            save_file = os.path.join(self.cache_dir, "est.pkl")
            logging.info("Loading feature extractor from {}.".format(save_file))
            with open(save_file, "rb") as fw:
                obj = pickle.load(fw)
                self.__dict__ = obj.__dict__
                return True
        except Exception as e:
            logging.info("Cannot load cached feature extractor.")
            return False

    def fit(self, session_dict):
        if self.load():
            return
        log_padding = "<pad>"
        log_oov = "<oov>"

        # encode
        total_logs = list(
            itertools.chain(*[v["templates"] for k, v in session_dict.items()])
        )
        self.ulog_train = set(total_logs)
        self.id2log_train = {0: log_padding, 1: log_oov}
        self.id2log_train.update(
            {idx: log for idx, log in enumerate(self.ulog_train, 2)}
        )
        self.log2id_train = {v: k for k, v in self.id2log_train.items()}

        logging.info("{} templates are found.".format(len(self.log2id_train)))

        if self.label_type == "next_log":
            self.meta_data["num_labels"] = len(self.log2id_train)
        elif self.label_type == "anomaly":
            self.meta_data["num_labels"] = 2
        else:
            logging.info('Unrecognized label type "{}"'.format(self.label_type))
            exit()

        if any(map(self.feature_type.__contains__, ["semantics"])):
        # if "semantics" in self.feature_type:
            logging.info("Using semantics.")
            logging.info("Building vocab.")
            self.vocab.build_vocab(self.ulog_train)
            logging.info("Building vocab done.")
            self.meta_data["vocab_size"] = self.vocab.token_vocab_size

            if self.pretrain_path is not None:
                logging.info(
                    "Using pretrain word embeddings from {}".format(self.pretrain_path)
                )
                self.meta_data["pretrain_matrix"] = self.vocab.gen_pretrain_matrix(
                    self.pretrain_path
                )
            if self.use_tfidf:
                self.vocab.fit_tfidf(total_logs)

        elif any(map(self.feature_type.__contains__, ["sentences"])):
            logging.info("Using sentences.")
            logging.info("Building vocab.")
            self.vocab.build_vocab(self.log2id_train, type_embeddings='sentences')
            logging.info("Building vocab done.")
            self.meta_data["vocab_size"] = self.vocab.sentence_vocab_size

            if self.pretrain_path is not None:
                logging.info(
                    "Using pretrain sentence embeddings from {}".format(self.pretrain_path)
                )
                self.meta_data["pretrain_matrix"] = self.vocab.gen_pretrain_matrix(
                    self.pretrain_path, type_embeddings = 'sentences'
                )

        #elif self.feature_type == "sequentials":
        elif any(map(self.feature_type.__contains__, ["sequentials"])):

            self.meta_data["vocab_size"] = len(self.log2id_train)

        else:
            logging.info('Unrecognized feature type "{}"'.format(self.feature_type))
            exit()

        if self.cache:
            self.save()

    def transform(self, session_dict, datatype="train"):
        logging.info("Transforming {} data.".format(datatype))

        ulog = set(itertools.chain(*[v["templates"] for k, v in session_dict.items()]))
        if datatype == "test":
            # handle new logs
            ulog_new = ulog - self.ulog_train
            logging.info(f"{len(ulog_new)} new templates show while testing.")

        if self.cache:
            cached_file = os.path.join(self.cache_dir, datatype + ".pkl")
            if os.path.isfile(cached_file):
                return load_pickle(cached_file)

        # generate windows, each window contains logid only
        if datatype == "train":
            self.__generate_windows(session_dict, self.stride)
        else:
            self.__generate_windows(session_dict, self.stride)

        if "sentences" in self.feature_type:
            #TODO
            #self.id2log_test.update({idx: log for idx, log in enumerate(self.ulog, 2)})
            #indice = {idx: log for idx, log in enumerate(ulog, 2)}
            log2idx = {log: self.vocab.sentence2idx[log] for log in ulog}
            log2idx["PADDING"] = 0
            logging.info("Extracting sentence features.")

        if "semantics" in self.feature_type:
            if self.use_tfidf:
                indice = self.vocab.transform_tfidf(ulog).toarray()
            else:
                indice = np.array(self.vocab.logs2idx(ulog))
            log2idx = {log: indice[idx] for idx, log in enumerate(ulog)}
            log2idx["PADDING"] = np.zeros(indice.shape[1]).reshape(-1)
            logging.info("Extracting semantic features.")

        for session_id, data_dict in session_dict.items():
            feature_dict = defaultdict(list)
            windows = data_dict["windows"]
            # generate sequential features # sliding windows on logid list
            # if self.feature_type == "sequentials":
            if "sequentials" in self.feature_type:
                feature_dict["sequentials"] = self.__windows2sequential(windows)

            # generate semantics features # use logid -> token id list
            if "semantics" in self.feature_type:
                feature_dict["semantics"] = self.__windows2semantics(windows, log2idx)

            if "sentences" in self.feature_type:
                feature_dict["sentences"] = self.__windows2sentences(windows, log2idx) # TODO

            # generate quantitative features # count logid in each window
            #if self.feature_type == "quantitatives":
            if "quantitatives" in self.feature_type:
                feature_dict["quantitatives"] = self.__windows2quantitative(windows)

            session_dict[session_id]["features"] = feature_dict

        logging.info("Finish feature extraction ({}).".format(datatype))
        if self.cache:
            dump_pickle(session_dict, cached_file)
        return session_dict

    def fit_transform(self, session_dict):
        self.fit(session_dict)
        return self.transform(session_dict, datatype="train")
