use candle_core::Tensor;


#[derive(Debug)]
#[records::record]
pub struct SequenceClassifierOutput {
    loss: Option<Tensor>,
    logits: Tensor,
    hidden_states: Option<Tensor>,
    attentions: Option<Tensor>
}