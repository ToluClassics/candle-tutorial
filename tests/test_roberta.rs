mod tests {
    use candle_tutorial::models::roberta::{RobertaEmbeddings,RobertaModel, RobertaConfig, create_position_ids_from_input_ids};
    use candle_tutorial::models::roberta::{RobertaForSequenceClassification, RobertaForTokenClassification }; 
    use candle_tutorial::utils::{build_roberta_model_and_tokenizer, ModelType, round_to_decimal_places};

    use anyhow::Result;
    use candle_nn::VarBuilder;
    use candle_core::{DType, Device, Tensor};

    // Regression_test = https://github.com/huggingface/transformers/blob/21dc5859421cf0d7d82d374b10f533611745a8c5/tests/models/xlm_roberta_xl/test_modeling_xlm_roberta_xl.py#L496
    #[test]
    fn test_create_position_ids_from_input_embeds() -> Result<()> {

        let config = RobertaConfig::default();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let embeddings_module = RobertaEmbeddings::load(vb, &config).unwrap();

        let input_embeds = Tensor::randn(0f32, 1f32, (2, 4, 30), &Device::Cpu).unwrap();
        let position_ids = embeddings_module.create_position_ids_from_input_embeds(&input_embeds);

        let expected_tensor: &[[u32; 4]; 2] = &[
            [0 + embeddings_module.padding_idx + 1, 1 + embeddings_module.padding_idx + 1, 2 + embeddings_module.padding_idx + 1, 3 + embeddings_module.padding_idx + 1,],
            [0 + embeddings_module.padding_idx + 1, 1 + embeddings_module.padding_idx + 1, 2 + embeddings_module.padding_idx + 1, 3 + embeddings_module.padding_idx + 1,]
        ];

        assert_eq!(position_ids.unwrap().to_vec2::<u32>()?, expected_tensor);

        Ok(())

    }

    #[test]
    fn test_create_position_ids_from_input_ids() -> Result<()> {

        let config = RobertaConfig::default();

        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let embeddings_module = RobertaEmbeddings::load(vb, &config).unwrap();

        let input_ids = &[[0u32, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]];
        let input_ids = Tensor::new(input_ids, &Device::Cpu)?;

        let position_ids = create_position_ids_from_input_ids(&input_ids, embeddings_module.padding_idx, 1)?;

        let expected_tensor = &[[2u8, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]];

        assert_eq!(position_ids.to_vec2::<u8>()?, expected_tensor);



        Ok(())
        

    }

    // https://github.com/huggingface/transformers/blob/e1cec43415e72c9853288d4e9325b734d36dd617/tests/models/roberta/test_modeling_roberta.py#L548
    #[test]
    fn test_modeling_roberta_base () -> Result<()> {
        let model_type = "RobertaModel";
        let (model, _tokenizer) =  build_roberta_model_and_tokenizer("roberta-base", false, model_type).unwrap();

        let model: RobertaModel = match model {
            ModelType::RobertaModel {model} => model,
            _ => panic!("Invalid model_type")
        };

        let input_ids = &[[0u32, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]];
        let input_ids = Tensor::new(input_ids, &model.device).unwrap();

        let token_ids = input_ids.zeros_like().unwrap();
        let output = model.forward(&input_ids, &token_ids)?;

        let expected_shape = [1, 11, 768];

        assert_eq!(output.shape().dims(), &expected_shape);

        let output = output.squeeze(0)?;
        let output = output.to_vec2::<f32>()?;
        let output: Vec<Vec<f32>> = output.iter().take(3).map(|nested_vec| nested_vec.iter().take(3).map(|&x| round_to_decimal_places(x, 4)).collect()).collect();

        let expected_output = [[-0.0231, 0.0782, 0.0074], [-0.1854, 0.0540, -0.0175], [0.0548, 0.0799, 0.1687]];

        assert_eq!(output, expected_output);

        Ok(())

    }


    // https://github.com/huggingface/transformers/blob/46092f763d26eb938a937c2a9cc69ce1cb6c44c2/tests/models/roberta/test_modeling_roberta.py#L567
    #[test]
    fn test_roberta_sequence_classification() -> Result<()> {

        let model_type = "RobertaForSequenceClassification";
        let (model, _tokenizer) =  build_roberta_model_and_tokenizer("roberta-large-mnli", false, model_type).unwrap();

        let model: RobertaForSequenceClassification = match model {
            ModelType::RobertaForSequenceClassification {model} => model,
            _ => panic!("Invalid model_type")
        };

        let input_ids = &[[0u32, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]];
        let input_ids = Tensor::new(input_ids, &model.device).unwrap();

        let token_ids = input_ids.zeros_like().unwrap();
        let output = model.forward(&input_ids, &token_ids, None)?;

        let expected_shape = [1, 3];
        let expected_output = [[-0.9469, 0.3913, 0.5118]];


        assert_eq!(output.logits.shape().dims(), &expected_shape);

        let output = output.logits.to_vec2::<f32>()?;
        let output: Vec<Vec<f32>> = output.iter().take(3).map(|nested_vec| nested_vec.iter().take(3).map(|&x| round_to_decimal_places(x, 4)).collect()).collect();

        assert_eq!(output, expected_output);

        Ok(())

    }

    #[test]
    fn test_roberta_token_classification() -> Result<()> {

        let model_type = "RobertaForTokenClassification";
        let (model, _tokenizer) =  build_roberta_model_and_tokenizer("Davlan/xlm-roberta-base-wikiann-ner", false, model_type).unwrap();

        let model: RobertaForTokenClassification = match model {
            ModelType::RobertaForTokenClassification {model} => model,
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

    #[test]
    fn test_roberta_question_answering() -> Result<()> {

        let model_type = "RobertaForTokenClassification";
        let (model, _tokenizer) =  build_roberta_model_and_tokenizer("deepset/roberta-base-squad2", false, model_type).unwrap();

        let model: RobertaForTokenClassification = match model {
            ModelType::RobertaForTokenClassification {model} => model,
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