import torch
from torch import nn

from deeploglizer.models import ForcastBasedModel


class Transformer(ForcastBasedModel):
    def __init__(
        self,
        meta_data,
        embedding_dim=16,
        nhead=4,
        hidden_size=100,
        num_layers=1,
        model_save_path="./transformer_models",
        feature_type="sequentials",
        label_type="next_log",
        eval_type="session",
        topk=5,
        patience= 3,
        use_tfidf=False,
        freeze=False,
        gpu=-1,
        **kwargs
    ):
        super().__init__(
            meta_data=meta_data,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            eval_type=eval_type,
            topk=topk,
            patience=patience,
            use_tfidf=use_tfidf,
            embedding_dim=embedding_dim,
            freeze=freeze,
            gpu=gpu,
        )
        num_labels = meta_data["num_labels"]
        self.feature_type = feature_type
        self.hidden_size = hidden_size
        self.use_tfidf = use_tfidf

        self.cls = torch.zeros(1, 1, embedding_dim).to(self.device)
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.rnn_extra = nn.LSTM(
        input_size=1, #added dimension for quantitative features
        hidden_size=self.hidden_size,
        batch_first=True,
        num_layers=num_layers,
        bidirectional=False,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.prediction_layer = nn.Linear(embedding_dim + (len(self.feature_type)-1)*hidden_size, num_labels)

    def forward(self, input_dict):
        if self.label_type == "anomaly":
            y = input_dict["window_anomalies"].long().view(-1)
        elif self.label_type == "next_log":
            y = input_dict["window_labels"].long().view(-1)
        self.batch_size = y.size()[0]

        features = input_dict["features"]

        x = features[0] #list iof features

        #x = input_dict["features"]
        x = self.embedder(x)

        if "semantics" in self.feature_type:
            if not self.use_tfidf:
                x = x.sum(dim=-2)  # add tf-idf

        x_t = x.transpose(1, 0)
        # cls_expand = self.cls.expand(-1, self.batch_size, -1)
        # embedding_with_cls = torch.cat([cls_expand, x_t], dim=0)

        x_transformed = self.transformer_encoder(x_t.float())
        representation = x_transformed.transpose(1, 0).mean(dim=1)
        # representation = x_transformed[0]

        if len(features) > 1: # count vector or another sequence feature
            x_extra = features[1] # counts
            outputs_extra, _ = self.rnn_extra(x_extra.float())
            representation = torch.cat((representation, outputs_extra[:, -1, :]), -1)

        logits = self.prediction_layer(representation)
        y_pred = logits.softmax(dim=-1)
        loss = self.criterion(logits, y)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict
