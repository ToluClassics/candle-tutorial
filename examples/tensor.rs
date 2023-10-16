use anyhow::Result;
use candle_core::{DType, Device, Tensor};

fn tensor_from_data() -> Result<()> {
    let data: [u32; 3] = [1u32, 2, 3];
    let tensor = Tensor::new(&data, &Device::Cpu)?;
    println!("tensor: {:?}", tensor.to_vec1::<u32>()?);

    let nested_data: [[u32; 3]; 3] = [[1u32, 2, 3], [4, 5, 6], [7, 8, 9]];
    let nested_tensor = Tensor::new(&nested_data, &Device::Cpu)?;
    println!("nested_tensor: {:?}", nested_tensor.to_vec2::<u32>()?);

    Ok(())
}

fn tensor_from_another_tensor() -> Result<()> {
    let data: [u32; 3] = [1u32, 2, 3];
    let tensor = Tensor::new(&data, &Device::Cpu)?;
    let zero_tensor = tensor.zeros_like()?;

    println!("zero_tensor: {:?}", zero_tensor.to_vec1::<u32>()?);

    let ones_tensor = tensor.ones_like()?;
    println!("ones_tensor: {:?}", ones_tensor.to_vec1::<u32>()?);

    let random_tensor = tensor.rand_like(0.0, 1.0)?;
    println!(
        "random_tensor: {:?}",
        random_tensor.to_vec1::<f64>().unwrap()
    );

    Ok(())
}

fn main() {
    let _ = tensor_from_data();

    let _ = tensor_from_another_tensor();
}
