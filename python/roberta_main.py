import torch
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaEmbeddings, RobertaConfig

model = RobertaModel.from_pretrained('roberta-base')

input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
with torch.no_grad():
    output = model(input_ids)[0]


# output = output.squeeze(0)[0][:10]

print(model.embeddings(input_ids).squeeze(0))
print(model.embeddings(input_ids).shape)

print(output[0])

# print(model.embeddings.position_embeddings.weight[0])

# print(model.embeddings.position_embeddings.weight[0])

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

print(create_position_ids_from_input_ids(input_ids, 1))