// use anyhow::Result;
use candle_core::{DType, Device, Tensor, Result};

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

pub fn sigmoid(xs: &Tensor) -> Result<Tensor> {
    // TODO: Should we have a specialized op for this?
    (xs.neg()?.exp()? + 1.0)?.recip()
}

pub fn binary_cross_entropy(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    let inp = sigmoid(inp)?;

    let one_tensor = Tensor::new(1.0, &inp.device())?;

    let left_side = target * inp.log()?;
    let right_side = (one_tensor.broadcast_sub(&target)?) * (one_tensor.broadcast_sub(&inp)?.log()?);

    let loss = left_side? + right_side?;
    let loss = loss?.neg()?.mean_all()?;
    
    Ok(loss)
}

fn main() {
    let _ = tensor_from_data();

    let _ = tensor_from_another_tensor();

    let inp = [[ 2.3611f64, -0.8813, -0.5006, -0.2178],
    [ 0.0419,  0.0763, -1.0457, -1.6692],
    [-1.0494,  0.8111,  1.5723,  1.2315],
    [ 1.3081,  0.6641,  1.1802, -0.2547],
    [ 0.5292,  0.7636,  0.3692, -0.8318],
    [ 0.5100,  0.9849, -1.2905,  0.2821],
    [ 1.4662,  0.4550,  0.9875,  0.3143],
    [-1.2121,  0.1262,  0.0598, -1.6363],
    [ 0.3214, -0.8689,  0.0689, -2.5094],
    [ 1.1320, -0.6824,  0.1657, -0.0687]];

    let target = [[0.0f64, 1., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
    [1., 0., 0., 0.],
    [0., 0., 1., 0.],
    [1., 0., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 1., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.]];

    let device = Device::Cpu;

    let inp = Tensor::new(&inp, &device).unwrap();
    let target = Tensor::new(&target, &device).unwrap();

    let loss = binary_cross_entropy(&inp, &target).unwrap();

    println!("{:?}", loss)
}
