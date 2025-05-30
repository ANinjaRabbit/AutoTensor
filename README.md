# AutoTensor : A Tensor library featured auto differentiation written in rust
**Note:this repository was only used to store my rust homework**


## Inspiration
At the beginning, I was intended to write a neural network framework using ndarray crate.However, I felt confused (and almost disgusted) by the ndarray's design(maybe fundamentally due to rust's design),so I turn to think that if I can design a ndarray library which is easy to use(Unfortunately, after coding, I find that maybe it's really hard in the context of Rust).
## Warning
Some of the functions are the same as numpy and pytorch,BUT SOME OF THEM ARE NOT OR MAYBE CANNOT FUNCTION AS EXPECTED.
## Basic Structure
### TensorRaw
Stores raw data (raw), shape (shape), strides (strides), offset (offset), and other low-level details.
Implements fundamental tensor operations (e.g., transpose, reshape, slicing) but is not directly exposed to users.
### Tensor
Wraps TensorRaw and provides high-level interfaces with automatic differentiation support.(Basically just Arc<RefCell<Tensor>>)
## Basic Usage

### Auto Differentiation
(Warning: There may exists some bugs when tackling complicated operations)
```
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