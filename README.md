# AutoTensor : A Tensor library featured auto differentiation written in rust
**Note:this repository was only used to store my rust homework**


## Inspiration
At the beginning, I was intended to write a neural network framework using ndarray crate.However, I felt confused (and almost disgusted) by the ndarray's design(maybe fundamentally due to rust's design),so I turn to think that if I can design a ndarray library which is easy to use(Unfortunately, after coding, I find that maybe it's really hard in the context of Rust).
## Warning
Some of the functions are the same as numpy and pytorch,BUT SOME OF THEM CANNOT OR MAYBE CANNOT FUNCTION AS EXPECTED.
## Basic Structure
### TensorRaw
Stores raw data (raw), shape (shape), strides (strides), offset (offset), and other low-level details.
Implements fundamental tensor operations (e.g., transpose, reshape, slicing) but is not directly exposed to users.
### Tensor
Wraps TensorRaw and provides high-level interfaces with automatic differentiation support.(Basically just Arc<RefCell<Tensor>>)
## Basic Usage
### Some Basic Operations
``` rust
    let mut a = Tensor::randn(&vec![3,3]); // initialize a 3x3 matrix from normal distribution
    let mut b = Tensor::randn(&vec![3,3]); // initialize a 3x3 matrix from normal distribution
    let mut c = &a + &b; // add two tensors
    let mut d = &a - &b; // subtract two tensors
    let mut e = &a * &b; // element-wise multiply two tensors
    let mut f = &a / &b; // element-wise divide two tensors
    let mut g = &a * Tensor::from(2.0); // multiply a tensor by a scalar
    let mut h = a.matmul(&b); // matrix multiplication
    let mut i = a.transpose(&vec![1,0]); // transpose a tensor
    let mut j = a.reshape(&vec![1,9]); // reshape a tensor
    let mut k = a.sum(0); // sum over the first dimension
    let mut l = a.exp();
    let mut m = a.get(s![(0,2),(1,2)]); // get a slice of the tensor
```

### Slicing
Because of the striction of index operator in Rust, we cannot using *[]* operator.
You can use the macro *s![]* to get a SliceIndex:
- ... for spaceholder
- (a , b) for [a , b)
- (a , b , c) for [a , b) with a stride of c
- a for just number a
```rust
    let a = Tensor::randn(&vec![3 , 3]);
    let b = a.get(s![(0 , 2) , (0 , 2)]);
    println!("{}" , a);
    println!("{}" , b);
```
### Auto Differentiation
(Warning: There may exists some bugs when tackling complicated operations)
``` rust
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
```

## Perceptron as an Example

```rust
pub mod tensor;

use tensor::*;
use crate::tensor::Tensor;

/// 单层感知机结构体
pub struct Perceptron {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Perceptron {
    /// 初始化，输入特征数 input_dim
    pub fn new(input_dim: usize) -> Self {
        let weight = Tensor::randn(&vec![input_dim, 1]); // 随机初始化权重
        let bias = Tensor::zeros(&vec![1]); // 偏置初始化为0
        Perceptron { weight, bias }
    }

    /// 前向传播
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let z = x.matmul(&self.weight) + &self.bias;
        z.sigmoid()
    }
    pub fn train(&mut self, x: &Tensor, y: &Tensor, learning_rate: f64) -> f64{
        let output = self.forward(x);
        let mut loss = (output-y).pow(2.0).mean(0); // 均方误差损失
        loss.init_grad();
        loss.set_grad(&Tensor::ones(&loss.shape())); // 初始化梯度为1
        loss.back_prop(); // 反向传播

        // 更新权重和偏置
        self.weight = &self.weight - &self.weight.grad() * Tensor::from(learning_rate);
        self.bias = &self.bias - &self.bias.grad() * Tensor::from(learning_rate);
        return loss.fetch(&vec![0]); // 返回当前损失值
    }
}
fn main(){
    let x = Tensor::randn(&vec![2, 2]); 
    let y = Tensor::randn(&vec![2, 1]); 


    let mut model = Perceptron::new(2); // 初始化感知机，输入特征数为2
    for epoch in 0..100 {
        let loss = model.train(&x, &y, 0.1); // 训练模型，学习率为0.1
        println!("Epoch {}: Loss = {}", epoch, loss);
    }

    
}
```