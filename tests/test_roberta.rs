mod tests {
    use candle_tutorial::models::roberta::{RobertaEmbeddings, RobertaConfig};
    use candle_nn::VarBuilder;
    use candle_core::{DType, Device, Tensor, Result};

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
}