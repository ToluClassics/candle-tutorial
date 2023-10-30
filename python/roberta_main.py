import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaEmbeddings, RobertaConfig
from transformers import RobertaForSequenceClassification

model = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli')

# input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2], [0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
# with torch.no_grad():
#     output = model(input_ids)


print(model)


# # output = output.squeeze(0)[0][:10]

# # print(model.embeddings(input_ids).squeeze(0))
# # print(model.embeddings(input_ids).shape)

# a = output[0][:, 0] 

# print(a)
# print(output[0].shape)

# b = output[0][:, 0, :]

# print("========================")

# print(b)

# # assert(== )

# print(output[0][:, 0].shape)

# # print(model.embeddings.position_embeddings.weight[0])

# # print(model.embeddings.position_embeddings.weight[0])

# def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
#     """
#     Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
#     are ignored. This is modified from fairseq's `utils.make_positions`.

#     Args:
#         x: torch.Tensor x:

#     Returns: torch.Tensor
#     """
#     # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
#     mask = input_ids.ne(padding_idx).int()
#     incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
#     return incremental_indices.long() + padding_idx

# # print(create_position_ids_from_input_ids(input_ids, 1))


# x = torch.Tensor([[ 2.3611, -0.8813, -0.5006, -0.2178],
#         [ 0.0419,  0.0763, -1.0457, -1.6692],
#         [-1.0494,  0.8111,  1.5723,  1.2315],
#         [ 1.3081,  0.6641,  1.1802, -0.2547],
#         [ 0.5292,  0.7636,  0.3692, -0.8318]])

# # print(x)

# y = torch.Tensor([[0., 1., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 0., 1.],
#         [1., 0., 0., 0.],
#         [0., 0., 1., 0.]])

# def sigmoid(x): 
#     return (1 + (-x).exp()).reciprocal()


# def binary_cross_entropy(input, y): 

#     print(pred.log()*y)
#     print((1-y)*(1-pred).log())
#     return -(pred.log()*y + (1-y)*(1-pred).log()).mean()

# pred = sigmoid(x)
# print(pred)

# print(binary_cross_entropy(pred, y))


# import torch
# import torch.nn.functional as F

# inp = torch.Tensor([[ 2.3611, -0.8813, -0.5006, -0.2178],
#     [ 0.0419,  0.0763, -1.0457, -1.6692],
#     [-1.0494,  0.8111,  1.5723,  1.2315],
#     [ 1.3081,  0.6641,  1.1802, -0.2547],
#     [ 0.5292,  0.7636,  0.3692, -0.8318]])

# target = torch.Tensor([[0., 1., 0., 0.],
#     [0., 1., 0., 0.],
#     [0., 0., 0., 1.],
#     [1., 0., 0., 0.],
#     [0., 0., 1., 0.]])

# print(F.binary_cross_entropy_with_logits(inp, target))