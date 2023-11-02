mod tests {
    use candle_tutorial::models::xlm_roberta::XLMRobertaModel;
    use candle_tutorial::models::xlm_roberta::XLMRobertaForTokenClassification ; 
    use candle_tutorial::utils::{build_roberta_model_and_tokenizer, ModelType};

    use anyhow::Result;
    use candle_core::Tensor;


    // https://github.com/huggingface/transformers/blob/46092f763d26eb938a937c2a9cc69ce1cb6c44c2/tests/models/xlm_roberta/test_modeling_xlm_roberta.py#L32
    #[test]
    fn test_modeling_xlm_roberta_base () -> Result<()> {
        let model_type = "XLMRobertaModel";
        let (model, _tokenizer) =  build_roberta_model_and_tokenizer("xlm-roberta-base", false, model_type).unwrap();

        let model: XLMRobertaModel = match model {
            ModelType::XLMRobertaModel {model} => model,
            _ => panic!("Invalid model_type")
        };

        let input_ids = &[[0u32, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2]];
        let input_ids = Tensor::new(input_ids, &model.device).unwrap();

        let token_ids = input_ids.zeros_like().unwrap();
        let output = model.forward(&input_ids, &token_ids)?;

        let expected_shape = [1, 12, 768];

        assert_eq!(output.shape().dims(), &expected_shape);

        Ok(())

    }


    #[test]
    fn test_inference_token_classification_head() -> Result<()> {

        let model_type = "XLMRobertaForTokenClassification";
        let (model, _tokenizer) =  build_roberta_model_and_tokenizer("Davlan/xlm-roberta-base-wikiann-ner", false, model_type).unwrap();

        let model: XLMRobertaForTokenClassification = match model {
            ModelType::XLMRobertaForTokenClassification {model} => model,
            _ => panic!("Invalid model_type")
        };

        let input_ids = &[[0u32, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]];
        let input_ids = Tensor::new(input_ids, &model.device).unwrap();

        let token_ids = input_ids.zeros_like().unwrap();
        let output = model.forward(&input_ids, &token_ids, None)?;

        println!("Output: {:?}",candle_nn::ops::softmax(&output.logits, candle_core::D::Minus1)?.to_vec3::<f32>()?);
        println!("Output: {:?}", output.logits.to_vec3::<f32>()?);

        Ok(())

    }


}