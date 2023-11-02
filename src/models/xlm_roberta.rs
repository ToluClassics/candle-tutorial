use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};

use crate::models::modelling_outputs::{SequenceClassifierOutput, TokenClassifierOutput, QuestionAnsweringModelOutput};
use crate::models::model_utils::{Dropout, HiddenAct, Linear, HiddenActLayer, LayerNorm, PositionEmbeddingType};
use crate::models::model_utils::binary_cross_entropy_with_logit;
use serde::Deserialize;

pub const FLOATING_DTYPE: DType = DType::F32;
pub const LONG_DTYPE: DType = DType::I64;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct XLMRobertaConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    hidden_act: HiddenAct,
    hidden_dropout_prob: f64,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    initializer_range: f64,
    layer_norm_eps: f64,
    pad_token_id: usize,
    bos_token_id: usize,
    eos_token_id: usize,
    #[serde(default)]
    position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    use_cache: bool,
    classifier_dropout: Option<f64>,
    model_type: Option<String>,
    problem_type: Option<String>,
    _num_labels: Option<usize>,
    id2label: Option<HashMap<String, String>>,
    label2id: Option<HashMap<String, usize>>
}

impl Default for XLMRobertaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50265,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("xlm-roberta".to_string()),
            problem_type: None,
            _num_labels: Some(3),
            id2label: None,
            label2id: None
        }
    }
}

fn cumsum_2d(mask: &Tensor, dim: u8, device: &Device) -> Result<Tensor> {
    let mask = mask.to_vec2::<u8>()?;

    let rows = mask.len();
    let cols = mask[0].len();

    let mut result = mask.clone();

    match dim {
        0 => {
            // Cumulative sum along rows
            for i in 0..rows {
                for j in 1..cols {
                    result[i][j] += result[i][j - 1];
                }
            }
        }
        1 => {
            // Cumulative sum along columns
            for j in 0..cols {
                for i in 1..rows {
                    result[i][j] += result[i - 1][j];
                }
            }
        }
        _ => panic!("Dimension not supported"),
    }

    let result = Tensor::new(result, &device)?;

    Ok(result)
}

pub fn create_position_ids_from_input_ids(
    input_ids: &Tensor,
    padding_idx: u32,
    past_key_values_length: u8,
) -> Result<Tensor> {
    let mask = input_ids.ne(padding_idx)?;
    let incremental_indices = cumsum_2d(&mask, 0, input_ids.device())?;

    let incremental_indices = incremental_indices
        .broadcast_add(&Tensor::new(&[past_key_values_length], input_ids.device())?)?;

    Ok(incremental_indices)
}

fn embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((vocab_size, hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, hidden_size))
}

fn linear(size1: usize, size2: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((size2, size1), "weight")?;
    let bias = vb.get(size2, "bias")?;
    Ok(Linear::new(weight, Some(bias)))
}

fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let (weight, bias) = match (vb.get(size, "weight"), vb.get(size, "bias")) {
        (Ok(weight), Ok(bias)) => (weight, bias),
        (Err(err), _) | (_, Err(err)) => {
            if let (Ok(weight), Ok(bias)) = (vb.get(size, "gamma"), vb.get(size, "beta")) {
                (weight, bias)
            } else {
                return Err(err);
            }
        }
    };
    Ok(LayerNorm::new(weight, bias, eps))
}

pub struct XLMRobertaEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    pub padding_idx: u32,
}

impl XLMRobertaEmbeddings {
    pub fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let padding_idx = config.pad_token_id as u32;

        Ok(Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            token_type_embeddings,
            layer_norm,
            dropout: Dropout::new(config.hidden_dropout_prob),
            padding_idx,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: Option<&Tensor>,
        inputs_embeds: Option<&Tensor>,
    ) -> Result<Tensor> {
        let position_ids = match position_ids {
            Some(ids) => ids.to_owned(),
            None => {
                if Option::is_some(&inputs_embeds) {
                    let position_ids =
                        self.create_position_ids_from_input_embeds(inputs_embeds.unwrap())?;
                    position_ids
                } else {
                    let position_ids =
                        create_position_ids_from_input_ids(input_ids, self.padding_idx, 1)?;
                    position_ids
                }
            }
        };

        let inputs_embeds: Tensor = match inputs_embeds {
            Some(embeds) => embeds.to_owned(),
            None => {
                let embeds = self.word_embeddings.forward(input_ids)?;
                embeds
            }
        };

        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let mut embeddings = (inputs_embeds + token_type_embeddings)?;

        if let Some(position_embeddings) = &self.position_embeddings {
            embeddings = embeddings.broadcast_add(&position_embeddings.forward(&position_ids)?)?
        }

        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings)?;

        Ok(embeddings)
    }

    pub fn create_position_ids_from_input_embeds(&self, input_embeds: &Tensor) -> Result<Tensor> {
        let input_shape = input_embeds.dims3()?;
        let seq_length = input_shape.1;

        println!("seq_length: {:?}", seq_length);
        let mut position_ids = Tensor::arange(
            self.padding_idx + 1,
            seq_length as u32 + self.padding_idx + 1,
            &Device::Cpu,
        )?;

        println!("position_ids: {:?}", position_ids);

        position_ids = position_ids
            .unsqueeze(0)?
            .expand((input_shape.0, input_shape.1))?;
        Ok(position_ids)
    }
}

struct XLMRobertaSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl XLMRobertaSelfAttention {
    fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let hidden_size = config.hidden_size;
        let query = linear(hidden_size, all_head_size, vb.pp("query"))?;
        let value = linear(hidden_size, all_head_size, vb.pp("value"))?;
        let key = linear(hidden_size, all_head_size, vb.pp("key"))?;
        Ok(Self {
            query,
            key,
            value,
            dropout,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs.contiguous()
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_probs =
            { candle_nn::ops::softmax(&attention_scores, candle_core::D::Minus1)? };
        let attention_probs = self.dropout.forward(&attention_probs)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(candle_core::D::Minus2)?;
        Ok(context_layer)
    }
}

struct XLMRobertaSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl XLMRobertaSelfOutput {
    fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

struct XLMRobertaAttention {
    self_attention: XLMRobertaSelfAttention,
    self_output: XLMRobertaSelfOutput,
}

impl XLMRobertaAttention {
    fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let self_attention = XLMRobertaSelfAttention::load(vb.pp("self"), config)?;
        let self_output = XLMRobertaSelfOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            self_attention,
            self_output,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let self_outputs = self.self_attention.forward(hidden_states)?;
        let attention_output = self.self_output.forward(&self_outputs, hidden_states)?;
        Ok(attention_output)
    }
}

struct XLMRobertaIntermediate {
    dense: Linear,
    intermediate_act: HiddenActLayer,
}

impl XLMRobertaIntermediate {
    fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let dense = linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            intermediate_act: HiddenActLayer::new(config.hidden_act),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let ys = self.intermediate_act.forward(&hidden_states)?;
        Ok(ys)
    }
}

struct XLMRobertaOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl XLMRobertaOutput {
    fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let dense = linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

struct XLMRobertaLayer {
    attention: XLMRobertaAttention,
    intermediate: XLMRobertaIntermediate,
    output: XLMRobertaOutput,
}

impl XLMRobertaLayer {
    fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let attention = XLMRobertaAttention::load(vb.pp("attention"), config)?;
        let intermediate = XLMRobertaIntermediate::load(vb.pp("intermediate"), config)?;
        let output = XLMRobertaOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states)?;

        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;
        Ok(layer_output)
    }
}

struct XLMRobertaEncoder {
    layers: Vec<XLMRobertaLayer>,
}

impl XLMRobertaEncoder {
    fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| XLMRobertaLayer::load(vb.pp(&format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        Ok(XLMRobertaEncoder { layers })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states)?
        }
        Ok(hidden_states)
    }
}

pub struct XLMRobertaPooler{
    dense: Linear,
    activation: HiddenActLayer,
}

impl XLMRobertaPooler{
    pub fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        Ok( Self {
            dense,
            activation: HiddenActLayer::new(HiddenAct::Tanh),
        })

    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // We "pool" the model by simply taking the hidden state corresponding
        // to the first token.

        let first_token_sensor = hidden_states.i((.., 0))?;
        let pooled_output = self.dense.forward(&first_token_sensor)?;
        let pooled_output = self.activation.forward(&pooled_output)?;
        
        Ok(pooled_output)
    }
}

pub struct XLMRobertaModel {
    embeddings: XLMRobertaEmbeddings,
    encoder: XLMRobertaEncoder,
    pub device: Device,
}

impl XLMRobertaModel {
    pub fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        
        let (embeddings, encoder) = match (
            XLMRobertaEmbeddings::load(vb.pp("embeddings"), config),
            XLMRobertaEncoder::load(vb.pp("encoder"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                if let Some(_model_type) = &config.model_type {
                    if let (Ok(embeddings), Ok(encoder)) = (
                        XLMRobertaEmbeddings::load(vb.pp(&format!("roberta.embeddings")), config),
                        XLMRobertaEncoder::load(vb.pp(&format!("roberta.encoder")), config),
                    ) {
                        (embeddings, encoder)
                    } else {
                        return Err(err);
                    }
                } else {
                    return Err(err);
                }
            }
        };
        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let embedding_output = self
            .embeddings
            .forward(input_ids, token_type_ids, None, None)?;
        let sequence_output = self.encoder.forward(&embedding_output)?;
        Ok(sequence_output)
    }
}

pub struct XLMRobertaModelWithPooler {
    embeddings: XLMRobertaEmbeddings,
    encoder: XLMRobertaEncoder,
    pooler: XLMRobertaPooler, 
    pub device: Device,
}

impl XLMRobertaModelWithPooler {
    pub fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let (embeddings, encoder, pooler) = match (
            XLMRobertaEmbeddings::load(vb.pp("embeddings"), config),
            XLMRobertaEncoder::load(vb.pp("encoder"), config),
            XLMRobertaPooler::load(vb.pp("pooler"), config)
        ) {
            (Ok(embeddings), Ok(encoder), Ok(pooler)) => (embeddings, encoder, pooler),
            (Err(err), _, _) | (_, Err(err), _) | (_, _, Err(err)) => {
                if let Some(model_type) = &config.model_type {
                    if let (Ok(embeddings), Ok(encoder), Ok(pooler)) = (
                        XLMRobertaEmbeddings::load(vb.pp(&format!("{model_type}.embeddings")), config),
                        XLMRobertaEncoder::load(vb.pp(&format!("{model_type}.encoder")), config),
                        XLMRobertaPooler::load(vb.pp(&format!("{model_type}.pooler")), config),
                    ) {
                        (embeddings, encoder, pooler)
                    } else {
                        return Err(err);
                    }
                } else {
                    return Err(err);
                }
            }
        };
        Ok(Self {
            embeddings,
            encoder,
            pooler,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let embedding_output = self
            .embeddings
            .forward(input_ids, token_type_ids, None, None)?;
        let sequence_output = self.encoder.forward(&embedding_output)?;
        let pooled_output = self.pooler.forward(&sequence_output)?;
        Ok(pooled_output)
    }
}

struct XLMRobertaClassificationHead{
    dense: Linear,
    dropout: Dropout,
    out_proj: Linear
}

impl XLMRobertaClassificationHead {

    fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let classifier_dropout = config.classifier_dropout;

        let classifier_dropout: f64 = match classifier_dropout {
            Some(classifier_dropout) => classifier_dropout,
            None => config.hidden_dropout_prob, 
        };
        let out_proj = linear(config.hidden_size, config._num_labels.unwrap(), vb.pp("out_proj"))?;

        Ok( Self {
            dense,
            dropout: Dropout::new(classifier_dropout),
            out_proj
        })

    }

    fn forward(&self, features: &Tensor) -> Result<Tensor> {

        let x = features.i((.., 0))?;
        let x = self.dropout.forward(&x)?;
        let x = self.dense.forward(&x)?;
        let x = x.tanh()?;
        let x = self.dropout.forward(&x)?;
        let x = self.out_proj.forward(&x)?;

        Ok(x)
    }
}

pub struct XLMRobertaForSequenceClassification {
    xlmroberta: XLMRobertaModel,
    classifier: XLMRobertaClassificationHead,
    pub device: Device,
    config: XLMRobertaConfig
}

impl  XLMRobertaForSequenceClassification {
    pub fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let (xlmroberta, classifier) = match (
            XLMRobertaModel::load(vb.pp("roberta"), config),
            XLMRobertaClassificationHead::load(vb.pp("classifier"), config),
        ) {
            (Ok(xlmroberta), Ok(classifier)) => (xlmroberta, classifier),
            (Err(err), _) | (_, Err(err)) => {
                return Err(err);
            }
        };
        Ok(Self {
            xlmroberta,
            classifier,
            device: vb.device().clone(),
            config: config.clone()
        })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor, labels: Option<&Tensor>) -> Result<SequenceClassifierOutput> {
        let outputs = self
            .xlmroberta
            .forward(input_ids, token_type_ids)?;
        let mut problem_type: String = String::from("");

        let logits = self.classifier.forward(&outputs)?;
        let mut loss: Tensor = Tensor::new(vec![0.0], &self.device)?;

        match labels {
            Some(labels) => {
                let labels = labels.to_device(&input_ids.device())?;

                if self.config.problem_type == None {
                    if self.config._num_labels == Some(1) {
                        problem_type = String::from("regression");
                    } else if self.config._num_labels > Some(1) && (labels.dtype() == LONG_DTYPE || labels.dtype() == DType::U32) {
                        problem_type = String::from("single_label_classification");
                    } else {
                        problem_type = String::from("multi_label_classification");
                    }
                }

                if problem_type == String::from("single_label_classification") {
                    loss = candle_nn::loss::cross_entropy(&logits.flatten_to(1)?, &labels.flatten_to(1)?)?;
                } else if problem_type == String::from("multi_label_classification") {
                    let labels_logits: Tensor =  logits.zeros_like()?;
                    let mut label_logits = labels_logits.to_vec2::<f32>()?;

                    let label = vec![0, 1, 2, 3, 2];

                    for vec_i in 0..label_logits.len() {
                            label_logits[vec_i][label[vec_i]] = 1.0;
                    }

                    let label_logits = Tensor::new(label_logits, &self.device)?;

                    loss = binary_cross_entropy_with_logit(&logits, &label_logits)?;
                }

            }

            None => {}
        }

        Ok(SequenceClassifierOutput {
            loss :Some(loss),
            logits,
            hidden_states :None,
            attentions : None
        })


    }

}

pub struct XLMRobertaForTokenClassification {
    xlmroberta: XLMRobertaModel,
    dropout: Dropout,
    classifier: Linear,
    pub device: Device,
}

impl XLMRobertaForTokenClassification {
    pub fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let classifier_dropout = config.classifier_dropout;

        println!("{:?}", config);

        let (xlmroberta, classifier) = match (
            XLMRobertaModel::load(vb.pp("roberta"), config),

            if Option::is_some(&config._num_labels) {
                linear(config.hidden_size, config._num_labels.unwrap(), vb.pp("classifier"))
            } else if Option::is_some(&config.id2label) {
                let num_labels = &config.id2label.as_ref().unwrap().len();
                linear(config.hidden_size, num_labels.clone(), vb.pp("classifier"))
            } else {
                candle_core::bail!("cannnot find the number of classes to map to")
            }
            
        ) {
            (Ok(xlmroberta), Ok(classifier)) => (xlmroberta, classifier),
            (Err(err), _) | (_, Err(err)) => {
                return Err(err);
            }
        };
        Ok(Self {
            xlmroberta,
            dropout: Dropout::new(classifier_dropout.unwrap_or_else(|| 0.2)),
            classifier,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor, labels: Option<&Tensor>) -> Result<TokenClassifierOutput> {
        let outputs = self
            .xlmroberta
            .forward(input_ids, token_type_ids)?;
        let outputs = self.dropout.forward(&outputs)?;

        let logits = self.classifier.forward(&outputs)?;

        println!("{:?}", logits);
        let mut loss: Tensor = Tensor::new(vec![0.0], &self.device)?;

        match labels {
            Some(labels) => {
                loss = candle_nn::loss::cross_entropy(&logits.flatten_to(1)?, &labels.flatten_to(1)?)?;
            }
            None => {}
        }
       
        Ok(TokenClassifierOutput {
            loss :Some(loss),
            logits,
            hidden_states :None,
            attentions : None
        })


    }

}


pub struct XLMRobertaForQuestionAnswering {
    xlmroberta: XLMRobertaModel,
    dropout: Dropout,
    qa_outputs: Linear,
    pub device: Device,
}


impl XLMRobertaForQuestionAnswering {
    pub fn load(vb: VarBuilder, config: &XLMRobertaConfig) -> Result<Self> {
        let classifier_dropout = config.classifier_dropout;

        println!("{:?}", config);

        let (xlmroberta, qa_outputs) = match (
            XLMRobertaModel::load(vb.pp("roberta"), config),
            linear(config.hidden_size, 2, vb.pp("classifier"))
            
        ) {
            (Ok(xlmroberta), Ok(qa_outputs)) => (xlmroberta, qa_outputs),
            (Err(err), _) | (_, Err(err)) => {
                return Err(err);
            }
        };
        Ok(Self {
            xlmroberta,
            dropout: Dropout::new(classifier_dropout.unwrap_or_else(|| 0.2)),
            qa_outputs,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor, start_positions: Option<&Tensor>, end_positions:  Option<&Tensor>) -> Result<QuestionAnsweringModelOutput> {
        let outputs = self
            .xlmroberta
            .forward(input_ids, token_type_ids)?;
        let outputs = self.dropout.forward(&outputs)?;

        let logits = self.qa_outputs.forward(&outputs)?;

        let start_logits = logits.i((.., 0))?;
        let end_logits = logits.i((.., 1))?;

        println!("{:?}", logits);
        let mut loss: Tensor = Tensor::new(vec![0.0], &self.device)?;

        match (start_positions, end_positions) {
            (Some(start_positions), Some(end_positions)) => {
                let start_loss = candle_nn::loss::cross_entropy(&start_logits.flatten_to(1)?, &start_positions.flatten_to(1)?)?;
                let end_loss = candle_nn::loss::cross_entropy(&end_logits.flatten_to(1)?, &end_positions.flatten_to(1)?)?;

                loss = ((start_loss + end_loss)? / 2.0)?;
            }
            _ => {}
        }
       
        Ok(QuestionAnsweringModelOutput {
            loss :Some(loss),
            start_logits,
            end_logits,
            hidden_states :None,
            attentions : None
        })

    }

}

