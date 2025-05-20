pub mod tensor;
pub mod nn;
use tensor::Tensor;
use nn::*;
fn main(){
    let input = Tensor::ones(&vec![2, 3]);
    let linear = Linear::new(3, 1);
    let mut output = linear.forward(&input).reshape(&vec![2]);
    println!("input: {}", input);
    println!("weight: {}", linear.weight);
    println!("bias: {}", linear.bias);
    println!("output: {}", output);
    output.init_grad();
    output.set_grad(&Tensor::ones(&vec![2]));
    output.back_prop();
    input.print_grad();
}