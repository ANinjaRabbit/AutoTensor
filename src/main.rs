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


        let x = Tensor::new(&vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ]);
    let rx = x.reshape(&vec![4 , 2]);
    let y = Tensor::new(&vec![
        0.0,
        0.0,
        0.0,
        1.0,
    ]);
    let ry = y.reshape(&vec![4 , 1]);


    let mut model = Perceptron::new(2); // 初始化感知机，输入特征数为3
    for epoch in 0..200 {
        let loss = model.train(&rx, &ry, 1.0); // 训练模型，学习率为1.0
        println!("Epoch {}: Loss = {}", epoch, loss);
    }
    println!("Final result:");
    println!("{}",model.forward(&rx));
    
}