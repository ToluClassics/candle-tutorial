use anyhow::{anyhow, Error as E, Result};
use hf_hub::{api::sync::Api, Cache, Repo, RepoType};
use tokenizers::Tokenizer;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use candle_tutorial::models::roberta::{RobertaModel, RobertaConfig, FLOATING_DTYPE};


fn build_model_and_tokenizer() -> Result<(RobertaModel, Tokenizer)> {
        let device = Device::Cpu;
        let default_model = "roberta-base".to_string();
        let default_revision = "main".to_string();
        let (model_id, revision) = (default_model, default_revision);
        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let offline = false;

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
        let model = RobertaModel::load(vb, &config)?;
        Ok((model, tokenizer))
    }


fn main() -> Result<()> {
    let (model, _tokenizer) = build_model_and_tokenizer()?;
    let device = &model.device;

    let input_ids = &[[0u32, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2], [0u32, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]];
    let input_ids = Tensor::new(input_ids, &device)?;

    let token_ids = input_ids.zeros_like()?;

    println!("token_ids: {:?}", token_ids.to_vec2::<u32>()?);
    println!("input_ids: {:?}", input_ids.to_vec2::<u32>()?);

    let output = model.forward(&input_ids, &token_ids)?;
    let output = output.squeeze(0)?;


    // println!("output: {:?}", output.shape());

    //print the first hidden state
    println!("output: {:?}", output.to_vec3::<f32>()?[0]);



    Ok(())
}
