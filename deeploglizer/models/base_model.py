import os
import time
import torch
import logging
import numpy as np
import pandas as pd
from torch import nn
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from deeploglizer.common.utils import set_device, tensor2flatten_arr


class Embedder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pretrain_matrix=None,
        freeze=False,
        use_tfidf=False,
    ):
        super(Embedder, self).__init__()
        self.use_tfidf = use_tfidf
        if pretrain_matrix is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(
                pretrain_matrix, padding_idx=1, freeze=freeze
            )
        else:
            self.embedding_layer = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=1
            )

    def forward(self, x):
        if self.use_tfidf:
            return torch.matmul(x, self.embedding_layer.weight.double())
        else:
            return self.embedding_layer(x.long())


class ForcastBasedModel(nn.Module):
    def __init__(
        self,
        meta_data,
        model_save_path,
        feature_type,
        label_type,
        eval_type,
        topk,
        use_tfidf,
        embedding_dim,
        freeze=False,
        gpu=-1,
        anomaly_ratio=None,
        patience=3,
        **kwargs,
    ):
        super(ForcastBasedModel, self).__init__()
        self.device = set_device(gpu)
        self.topk = topk
        self.meta_data = meta_data
        self.feature_type = feature_type
        self.label_type = label_type
        self.eval_type = eval_type
        self.anomaly_ratio = anomaly_ratio  # only used for auto encoder
        self.patience = patience
        self.time_tracker = {}
        
        self.log = {"train": {key: [] for key in ["epoch", "loss"]},
                    "eval": {key: [] for key in ["epoch", "loss"]}
                    }
        self.model_save_path = model_save_path
        
        os.makedirs(model_save_path, exist_ok=True)
        self.model_save_file = os.path.join(model_save_path, "model.ckpt")
        if any(map(self.feature_type.__contains__, ["sequentials", "semantics", "sentences"])): #feature_type in ["sequentials", "semantics"]:
            self.embedder = Embedder(
                meta_data["vocab_size"],
                embedding_dim=embedding_dim,
                pretrain_matrix=meta_data.get("pretrain_matrix", None),
                freeze=freeze,
                use_tfidf=use_tfidf,
            )
        else:
            logging.info(f'Unrecognized feature type, except sequentials or semantics, got {feature_type}')

    def evaluate(self, test_loader, epoch=1, dtype="test"):
        logging.info("Evaluating {} data.".format(dtype))

        if self.label_type == "next_log":
            return self.__evaluate_next_log(test_loader, epoch, dtype=dtype)
        elif self.label_type == "anomaly":
            return self.__evaluate_anomaly(test_loader, dtype=dtype)
        elif self.label_type == "none":
            return self.__evaluate_recst(test_loader, dtype=dtype)

    def __evaluate_recst(self, test_loader, dtype="test"):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()
            for batch_input in test_loader:
                return_dict = self.forward(self.__input2device(batch_input))
                y_pred = return_dict["y_pred"]
                store_dict["session_idx"].extend(
                    tensor2flatten_arr(batch_input["session_idx"])
                )
                store_dict["window_anomalies"].extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )
                store_dict["window_preds"].extend(tensor2flatten_arr(y_pred))
            infer_end = time.time()
            logging.info("Finish inference [{:.2f}s]".format(infer_end - infer_start))
            self.time_tracker["test"] = infer_end - infer_start

            store_df = pd.DataFrame(store_dict)

            use_cols = ["session_idx", "window_anomalies", "window_preds"]
            session_df = (
                store_df[use_cols]
                .groupby("session_idx", as_index=False)
                .max()  # most anomalous window
            )
            assert (
                self.anomaly_ratio is not None
            ), "anomaly_ratio should be specified for autoencoder!"
            thre = np.percentile(
                session_df[f"window_preds"].values, 100 - self.anomaly_ratio * 100
            )
            pred = (session_df[f"window_preds"] > thre).astype(int)
            y = (session_df["window_anomalies"] > 0).astype(int)

            eval_results = {
                "f1": f1_score(y, pred),
                "rc": recall_score(y, pred),
                "pc": precision_score(y, pred),
                "acc": accuracy_score(y, pred),
                "spec": recall_score(np.logical_not(y), np.logical_not(pred)),
                "pred": pred
            }

            logging.info({k: f"{v:.3f}" for k, v in eval_results.items() if k != 'pred'})
            return eval_results

    def __evaluate_anomaly(self, test_loader, dtype="test"):

        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()
            for batch_input in test_loader:
                return_dict = self.forward(self.__input2device(batch_input))
                y_prob, y_pred = return_dict["y_pred"].max(dim=1)
                store_dict["session_idx"].extend(
                    tensor2flatten_arr(batch_input["session_idx"])
                )
                store_dict["window_anomalies"].extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )
                store_dict["window_preds"].extend(tensor2flatten_arr(y_pred))
            infer_end = time.time()
            logging.info("Finish inference. [{:.2f}s]".format(infer_end - infer_start))
            self.time_tracker["test"] = infer_end - infer_start

            store_df = pd.DataFrame(store_dict)
            use_cols = ["session_idx", "window_anomalies", "window_preds"]
            session_df = store_df[use_cols].groupby("session_idx", as_index=False).sum()
            pred = (session_df[f"window_preds"] > 0).astype(int)
            y = (session_df["window_anomalies"] > 0).astype(int)

            eval_results = {
                "f1": f1_score(y, pred),
                "rc": recall_score(y, pred),
                "pc": precision_score(y, pred),
                "acc": accuracy_score(y, pred),
                "spec": recall_score(np.logical_not(y), np.logical_not(pred)),
            }
            logging.info({k: f"{v:.3f}" for k, v in eval_results.items()})
            return eval_results

    def __evaluate_next_log(self, test_loader, epoch, dtype="test"):
        model = self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()
            batch_cnt_eval = 0
            epoch_loss_eval = 0
            for batch_input in test_loader:
                return_dict = model.forward(self.__input2device(batch_input))
                ####################################
                loss_eval = return_dict["loss"]
                epoch_loss_eval += loss_eval.item()
                batch_cnt_eval += 1
                ####################################
                y_pred = return_dict["y_pred"]
                y_prob_topk, y_pred_topk = torch.topk(y_pred, self.topk)  # b x topk

                store_dict["session_idx"].extend(
                    tensor2flatten_arr(batch_input["session_idx"])
                )
                store_dict["window_anomalies"].extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )
                store_dict["window_labels"].extend(
                    tensor2flatten_arr(batch_input["window_labels"])
                )
                for ind, feature_input in enumerate(batch_input["features"]):
                    store_dict[f"x{ind}"].extend(feature_input.data.cpu().numpy())

                store_dict["y_pred_topk"].extend(y_pred_topk.data.cpu().numpy())
                store_dict["y_prob_topk"].extend(y_prob_topk.data.cpu().numpy())
            
            epoch_loss_eval = epoch_loss_eval / batch_cnt_eval
            logging.info(
                "Evaluation loss: {:.5f}".format(epoch_loss_eval)
            )
            self.log['eval']['epoch'].append(epoch)
            self.log['eval']['loss'].append(epoch_loss_eval)

            infer_end = time.time()
            logging.info("Finish inference. [{:.2f}s]".format(infer_end - infer_start))
            self.time_tracker["test"] = infer_end - infer_start
            store_df = pd.DataFrame(store_dict)
            best_result = None
            #best_f1 = -float("inf")
            best_spec = -float("inf")

            count_start = time.time()

            topkdf = pd.DataFrame(store_df["y_pred_topk"].tolist())
            logging.info("Calculating acc sum.")
            hit_df = pd.DataFrame()
            for col in sorted(topkdf.columns):
                topk = col + 1
                hit = (topkdf[col] == store_df["window_labels"]).astype(int)
                hit_df[topk] = hit
                if col == 0:
                    acc_sum = 2 ** topk * hit
                else:
                    acc_sum += 2 ** topk * hit
            acc_sum[acc_sum == 0] = 2 ** (1 + len(topkdf.columns))
            hit_df["acc_num"] = acc_sum

            for col in sorted(topkdf.columns):
                topk = col + 1
                check_num = 2 ** topk
                store_df["window_pred_anomaly_{}".format(topk)] = (
                    ~(hit_df["acc_num"] <= check_num)
                ).astype(int)
            # store_df.to_csv("store_{}_2.csv".format(dtype), index=False)

            logging.info("Finish generating store_df.")

            if self.eval_type == "session":
                use_cols = ["session_idx", "window_anomalies"] + [
                    f"window_pred_anomaly_{topk}" for topk in range(1, self.topk + 1)
                ]
                session_df = (
                    store_df[use_cols].groupby("session_idx", as_index=False).sum()
                )

                window_pred_anomaly = store_df[use_cols].groupby("session_idx")[f"window_pred_anomaly_{self.topk}"].apply(list).to_frame(name=f"window_list_anomaly_{self.topk}")

            else:
                session_df = store_df
            # session_df.to_csv("session_{}_2.csv".format(dtype), index=False)

            for topk in range(1, self.topk + 1):
                pred = (session_df[f"window_pred_anomaly_{topk}"] > 0).astype(int)
                y = (session_df["window_anomalies"] > 0).astype(int)
                window_topk_acc = 1 - store_df[f"window_pred_anomaly_{topk}"].sum() / len(store_df)

                #results=None
                #if topk == self.topk: 
                #    results = pd.concat([session_df['session_idx'], pred, window_pred_anomaly],axis=1)

                eval_results = {
                    "f1": f1_score(y, pred),
                    "rc": recall_score(y, pred),
                    "pc": precision_score(y, pred),
                    "spec": recall_score(np.logical_not(y), np.logical_not(pred)),
                    "top{}-acc".format(topk): window_topk_acc,
                    #"pred": results
                }
                logging.info({k: f"{v:.3f}" for k, v in eval_results.items() if k != 'pred'})
                #if eval_results["f1"] >= best_f1:
                if eval_results["spec"] >= best_spec:
                    eval_results["pred"] = pd.concat([session_df['session_idx'], pred, window_pred_anomaly],axis=1)
                    best_result = eval_results
                    best_spec = eval_results["spec"]
                    best_f1 = eval_results["f1"]
                    best_rc = eval_results["rc"]
                    best_pc = eval_results["pc"]
            count_end = time.time()
            logging.info("Finish counting [{:.2f}s]".format(count_end - count_start))
            return best_result

    def __input2device(self, batch_input):
        #dict_batch = {k: v.to(self.device) for k, v in batch_input.items()}
        dict_batch = {k: v.to(self.device) for k, v in batch_input.items() if k != 'features'}
        dict_batch['features'] = [feature.to(self.device) for feature in batch_input['features']] #turned into list of features
        return dict_batch
        
    def save_model(self):
        logging.info("Saving model to {}".format(self.model_save_file))
        try:
            torch.save(
                self.state_dict(),
                self.model_save_file,
                _use_new_zipfile_serialization=False,
            )
        except:
            torch.save(self.state_dict(), self.model_save_file)

    def load_model(self, model_save_file=""):
        logging.info("Loading model from {}".format(self.model_save_file))
        self.load_state_dict(torch.load(model_save_file, map_location=self.device))

    def fit(self, train_loader, test_loader=None, epoches=10, learning_rate=1.0e-3, stop_criteria = "f1"):
        self.to(self.device)
        logging.info(
            "Start training on {} batches with {}.".format(
                len(train_loader), self.device
            )
        )
        #best_f1 = -float("inf")
        best_metric = -float("inf")
        best_results = None
        worse_count = 0
        for epoch in range(1, epoches + 1):
            epoch_time_start = time.time()
            model = self.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            batch_cnt = 0
            epoch_loss = 0
            for batch_input in train_loader:
                loss = model.forward(self.__input2device(batch_input))["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                batch_cnt += 1
            epoch_loss = epoch_loss / batch_cnt
            epoch_time_elapsed = time.time() - epoch_time_start
            logging.info(
                "Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, epoches, epoch_loss, epoch_time_elapsed)
            )
            self.time_tracker["train"] = epoch_time_elapsed
            self.log['train']['epoch'].append(epoch)
            self.log['train']['loss'].append(epoch_loss)

            if test_loader is not None and (epoch % 1 == 0):
                eval_results = self.evaluate(test_loader, epoch)
                #if eval_results["f1"] > best_f1:
                if eval_results[stop_criteria] > best_metric:
                    #best_f1 = eval_results["f1"]
                    best_metric = eval_results[stop_criteria]
                    best_results = eval_results
                    best_results["converge"] = int(epoch)
                    
                    self.save_model()
                    worse_count = 0
                else:
                    worse_count += 1
                    if worse_count >= self.patience:
                        logging.info("Early stop at epoch: {}".format(epoch))
                        break

        self.load_model(self.model_save_file)
        self.save_log() # Save train and eval loss 

        return best_results

    def save_log(self):
        
        try:
            for key, values in self.log.items():
                output_file = os.path.join(self.model_save_path, f"{key}_log.csv")
                pd.DataFrame(values).to_csv(output_file, index=False)
            print("Log saved")
        except:
            print("Failed to save logs")