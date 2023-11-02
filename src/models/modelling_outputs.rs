use candle_core::Tensor;


#[derive(Debug)]
#[records::record]
pub struct SequenceClassifierOutput {
    loss: Option<Tensor>,
    logits: Tensor,
    hidden_states: Option<Tensor>,
    attentions: Option<Tensor>
}

#[derive(Debug)]
#[records::record]
pub struct TokenClassifierOutput {
    loss: Option<Tensor>,
    logits: Tensor,
    hidden_states: Option<Tensor>,
    attentions: Option<Tensor>
}

#[derive(Debug)]
#[records::record]
pub struct QuestionAnsweringModelOutput {
    loss: Option<Tensor>,
    start_logits: Tensor,
    end_logits: Tensor,
    hidden_states: Option<Tensor>,
    attentions: Option<Tensor>
}