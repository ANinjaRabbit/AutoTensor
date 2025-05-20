
use crate::tensor::{MultiDimIterator, Tensor};

pub trait Layer {
    /// 前向传播
    fn forward(&self, input: &Tensor) -> Tensor;
    /// 可选：返回参数（如权重和偏置），便于优化器统一管理
    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}
pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let inputshape = input.shape()[0..input.dim()-1].to_vec();
        let output = Tensor::zeros(&[inputshape.clone(), vec![self.weight.shape()[0]]].concat());
        for i in MultiDimIterator::new(&inputshape){
            let out = self.weight.dot(&input.get(&i)) + &self.bias; 
            output.get(&i).assign(&out);
        }
        output
    }
    fn parameters(&self) -> Vec<&Tensor> {
        let params = vec![&self.weight,&self.bias];
        params
    }
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let weight = Tensor::randn(&vec![output_dim, input_dim]);
        let bias =   Tensor::zeros(&vec![output_dim]);
        Linear { weight, bias }
    }
}
