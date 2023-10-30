mod tests {
    use candle_tutorial::models::roberta::{RobertaEmbeddings,RobertaModel, RobertaConfig,FLOATING_DTYPE, create_position_ids_from_input_ids};
    use candle_tutorial::models::roberta::RobertaForSequenceClassification; 
    use anyhow::{anyhow, Error as E, Result};
    use hf_hub::{api::sync::Api, Cache, Repo, RepoType};
    use tokenizers::Tokenizer;
    use candle_nn::VarBuilder;
    use candle_core::{DType, Device, Tensor};

    enum RobertaModelType {
        RobertaModel {model: RobertaModel},
        RobertaForSequenceClassification {model: RobertaForSequenceClassification}
    }

    fn round_to_decimal_places(n: f32, places: u32) -> f32 {
        let multiplier: f32 = 10f32.powi(places as i32);
        (n * multiplier).round() / multiplier
    }

    fn build_roberta_model_and_tokenizer(model_name_or_path: impl Into<String>, offline: bool, model_type: &str) -> Result<(RobertaModelType, Tokenizer)> {
        let device = Device::Cpu;
        let (model_id, revision) = (model_name_or_path.into(), "main".to_string());
        let repo = Repo::with_revision(model_id, RepoType::Model, revision);

        let (config_filename, tokenizer_filename, weights_filename) = if offline {
            let cache = Cache::default().repo(repo);
            (
                cache
                    .get("config.json")
                    .ok_or(anyhow!("Missing config file in cache"))?,
                cache
                    .get("tokenizer.json")
                    .ok_or(anyhow!("Missing tokenizer file in cache"))?,
                cache
                    .get("model.safetensors")
                    .ok_or(anyhow!("Missing weights file in cache"))?,
            )
        } else {
            let api = Api::new()?;
            let api = api.repo(repo);
            (
                api.get("config.json")?,
                api.get("tokenizer.json")?,
                api.get("model.safetensors")?,
            )
        };

        println!("config_filename: {}", config_filename.display());

        let config = std::fs::read_to_string(config_filename)?;
        let config: RobertaConfig = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], FLOATING_DTYPE, &device)? };

        let model = match model_type {
            "RobertaModel" => {
                let model = RobertaModel::load(vb, &config)?;
                RobertaModelType::RobertaModel {model}
            }
            "RobertaForSequenceClassification" => {
                let model = RobertaForSequenceClassification::load(vb, &config)?;
                RobertaModelType::RobertaForSequenceClassification {model}
            }
            _ => panic!("Invalid model_type")
        };

        Ok((model, tokenizer))
    }

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
            RobertaModelType::RobertaModel {model} => model,
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
    fn test_inference_classification_head() -> Result<()> {

        let model_type = "RobertaForSequenceClassification";
        let (model, _tokenizer) =  build_roberta_model_and_tokenizer("roberta-large-mnli", false, model_type).unwrap();

        let model: RobertaForSequenceClassification = match model {
            RobertaModelType::RobertaForSequenceClassification {model} => model,
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


}