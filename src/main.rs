pub mod tensor;
pub mod nn;
use std::vec;

use tensor::Tensor;
fn main(){
    let mut a = Tensor::randn(&vec![3,3]); // initialize a 3x3 matrix from normal distribution
    let mut b = Tensor::randn(&vec![3,3]); // initialize a 3x3 matrix from normal distribution
    let mut c = &a * &b;
    println!("a = {}",a);
    println!("b = {}",b);
    c.init_grad(); // initialize all the Tensors in upstream
    c.set_grad(&Tensor::ones(&vec![3,3])); // initialize itself
    c.back_prop(); // propagation
    a.print_grad();
    b.print_grad();
}