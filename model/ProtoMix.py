from model.RNN import RNN
import torch
import random
import numpy as np
from model.model_utils import *
import torch.nn as nn
import torch.nn.functional as F
import math

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class BiaffineAttention(nn.Module):
    def __init__(self, in1_features, in2_features, out_features=1):
        super(BiaffineAttention, self).__init__()
        self.bilinear = nn.Bilinear(in1_features, in2_features, out_features, bias=True)
        self.lin = nn.Linear(in1_features + in2_features, out_features, bias=False)

    def forward(self, x1, x2):
        output = self.bilinear(x1, x2)
        output = output + self.lin(torch.cat([x1, x2], dim=1))
        return output

class ProtoMix(nn.Module):
    def __init__(self, args,
                 **kwargs):
        super(ProtoMix, self).__init__()
        self.device = torch.device(args.cuda_choice if torch.cuda.is_available() else "cpu")
        self.datatset = kwargs["dataset"]
        self.feature_keys=kwargs["feature_keys"]
        self.label_key=kwargs["label_key"]
        self.mode=kwargs["mode"]
        self.train_patient_num = kwargs["train_patient_num"]
        self.positive_frequency = kwargs["positive_frequency"]
        self.negative_frequency = kwargs["negative_frequency"]
        self.sample_ratio = args.sample_ratio
        
        self.train_dropout_rate = args.train_dropout_rate
        self.hidden_dim = args.hidden_dim
        self.embed_dim = args.embed_dim
        self.mixup_alpha = args.mixup_alpha

        self.model = RNN(
            dataset=self.datatset,
            feature_keys=self.feature_keys,
            label_key=self.label_key,
            mode=self.mode,
            embedding_dim = self.embed_dim,
            hidden_dim = self.hidden_dim,
            train_dropout_rate = self.train_dropout_rate,
            rnn_type = "GRU",
            num_layers= args.encoder_layer
        )
        self.patient_dim = len(self.model.feature_keys) * self.model.hidden_dim

        self.num_prototypes = args.num_prototypes

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(self.train_dropout_rate)

        self.register_buffer("train_patient_embed", torch.randn(self.train_patient_num, self.patient_dim))
        self.register_buffer("prototype_embed", torch.randn(self.num_prototypes, self.patient_dim))
        output_size = self.model.get_output_size(self.model.label_tokenizer)
        
        self.proto_fc = nn.Linear(self.patient_dim, self.num_prototypes)
        self.class_fc = nn.Linear(self.num_prototypes, output_size)
        self.leakyrelu = nn.LeakyReLU()

        self.biaffineattention = BiaffineAttention(self.patient_dim,self.patient_dim,1)

    def _init_weight(self):
        self.model.apply(init_weights)
        self.class_fc.apply(init_weights())
        self.proto_fc.apply(init_weights())
        self.biaffineattention.apply(init_weights())
        nn.init.xavier_uniform_(self.train_patient_embed)
        nn.init.xavier_uniform_(self.prototype_embed)

    def class_linear(self,embed):
        return self.class_fc(self.leakyrelu(self.proto_fc(embed)))

    def mixup_process(self, embed, labels, adaptive_alpha=False, mixup_alpha=1.0):
        if adaptive_alpha == False:
            if mixup_alpha > 0:
                lam = torch.distributions.beta.Beta(mixup_alpha, mixup_alpha).sample().item()
            else:
                lam = 1
            batch_size = embed.size(0)
            index = torch.randperm(batch_size).to(self.device)

            mixed_embed = lam * embed + (1 - lam) * embed[index, :]
            labels_a, labels_b = labels, labels[index]
            labels_reweighted = labels_a * lam + labels_b * (1 - lam)
            return mixed_embed, labels_reweighted

        else:
           
            batch_size = embed.size(0)
            index = torch.randperm(batch_size).to(self.device)
            embed_a, embed_b = embed, embed[index, :]

            distances_a = torch.cdist(embed_a, self.prototype_embed.clone().detach())
            labels_a = torch.argmin(distances_a, dim=1)
            prototypes_a = self.prototype_embed.clone().detach()[labels_a]
            distances_b = torch.cdist(embed_b, self.prototype_embed.clone().detach())
            labels_b = torch.argmin(distances_b, dim=1)
            prototypes_b = self.prototype_embed.clone().detach()[labels_b]
            lam = self.sigmoid(self.biaffineattention(prototypes_a,prototypes_b))

            mixed_embed = lam * embed_a + (1 - lam) * embed_b
            labels_a, labels_b = labels, labels[index]
            labels_reweighted = labels_a * lam + labels_b * (1 - lam)

            return mixed_embed, labels_reweighted

    def forward_train(self,**kwargs):
        patient_embed, mask, patient_emb_all_step = self.model(**kwargs)
        patient_embed = self.dropout(patient_embed)
        patient_y_true = self.model.prepare_labels(kwargs[self.model.label_key], self.model.label_tokenizer)
        
        distances = torch.cdist(patient_embed, self.prototype_embed.clone().detach())
        labels = torch.argmin(distances, dim=1)
        prototype_labels_one_hot = one_hot(labels, self.num_prototypes).to(self.device)

        mixed_embed, labels_reweighted = self.mixup_process(patient_embed, prototype_labels_one_hot,
                                                       adaptive_alpha=True, mixup_alpha=self.mixup_alpha)
        proto_logits = self.proto_fc(mixed_embed)
        loss_protolabel = F.cross_entropy(proto_logits, labels_reweighted)

        sample_indices = (self.prototype_embed.clone().detach(), self.positive_frequency, self.negative_frequency, patient_embed,
                                          kwargs[self.model.label_key], self.sample_ratio)
        patient_embed_sample = patient_embed[sample_indices]
        patient_y_true_sample = patient_y_true[sample_indices]

        mixed_embed, labels_reweighted = self.mixup_process(patient_embed_sample, patient_y_true_sample,
                                                       adaptive_alpha=True, mixup_alpha=self.mixup_alpha)
        cls_logits = self.class_linear(mixed_embed)
        loss_protocls = F.binary_cross_entropy_with_logits(cls_logits,labels_reweighted)

        patient_embed_final = patient_embed
        origin_cls_logits = self.class_linear(patient_embed_final)
        patient_y_prob = self.model.prepare_y_prob(origin_cls_logits)
        loss_origincls = F.binary_cross_entropy_with_logits(origin_cls_logits, patient_y_true)

        return loss_protolabel, loss_protocls, loss_origincls, patient_y_true, patient_y_prob, patient_embed

    def forward_test(self, **kwargs):
        patient_embed, mask, patient_emb_all_step = self.model(**kwargs)
        patient_y_true = self.model.prepare_labels(kwargs[self.model.label_key], self.model.label_tokenizer)
        patient_embed_final = patient_embed
        origin_cls_logits = self.class_linear(patient_embed_final)
        patient_y_prob = self.model.prepare_y_prob(origin_cls_logits)
        loss_origincls = F.binary_cross_entropy_with_logits(origin_cls_logits, patient_y_true)
        return loss_origincls, origin_cls_logits, patient_y_true, patient_y_prob, patient_embed

    def patientEmbedUpdate(self, patiet_embed_train_all):
        self.train_patient_embed[:, :] = patiet_embed_train_all

    def prototypeEmbedUpdate(self,prototype_embed_new):
        self.prototype_embed[:, :] = prototype_embed_new

    def forward(self, mode, **kwargs):
        if mode == 'train':
            return self.forward_train(**kwargs)
        elif mode == 'test':
            return self.forward_test(**kwargs)