use core::{num, panic};
use std::{ cell::RefCell, collections::HashMap, fmt::Display, ops::{Add, Div, Mul, Neg, Sub}, sync::Arc, vec};
use rand::{self, fill};
struct MultiDimIterator {
    current: Vec<usize>,
    bounds: Vec<usize>,
    finished: bool,
}

impl MultiDimIterator {
    fn new(bounds: &[usize]) -> Self {
        let dims = bounds.len();
        let current = vec![0; dims];
        let finished = bounds.iter().any(|&b| b == 0);
        
        Self { current, bounds: bounds.to_vec(), finished }
    }
}

impl Iterator for MultiDimIterator {
    type Item = Vec<usize>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let result = self.current.clone();
        let mut dim = self.current.len();
        while let Some(d) = dim.checked_sub(1) {
            self.current[d] += 1;
            
            if self.current[d] < self.bounds[d] {
                break; 
            }
            
            self.current[d] = 0; 
            dim = d;
        }
        if dim == 0 && self.current.iter().all(|&v| v == 0) {
            self.finished = true;
        }
        
        Some(result)
    }
}


type SliceUnit = (usize,usize,usize);
#[derive(Debug,Copy,Clone,PartialEq)]
pub enum SliceIndexUnit{
    Num(usize),
    Slice(SliceUnit),
    Filler
}
pub type SliceIndex = Vec<SliceIndexUnit>;

macro_rules! s {
    ( $($tokens:tt),* $(,)? ) => {
        {
            let mut slice_index = Vec::new();
            $(
                s!(@parse_unit $tokens, slice_index);
            )*
            slice_index
        }
    };
    (@parse_unit $unit:tt, $slice_index:ident) => {
        match s!(@tokenize $unit) {
            SliceIndexUnit::Num(n) => $slice_index.push(SliceIndexUnit::Num(n)),
            SliceIndexUnit::Slice(s) => $slice_index.push(SliceIndexUnit::Slice(s)),
            SliceIndexUnit::Filler => $slice_index.push(SliceIndexUnit::Filler),
        }
    };

    (@tokenize ($start:literal , $end:literal , $step:literal)) => {
        SliceIndexUnit::Slice(($start, $end, $step))
    };

    (@tokenize ($start:literal , $end:literal)) => {
        SliceIndexUnit::Slice(($start, $end, 1))
    };

    (@tokenize ..) => {
        SliceIndexUnit::Filler
    };

    (@tokenize $num:literal) => {
        SliceIndexUnit::Num($num as usize)
    };
}


pub fn slice_from_vec(v : &Vec<usize>) -> Vec<SliceIndexUnit> {
    let mut slice_index = Vec::new();
    for i in v {
        slice_index.push(SliceIndexUnit::Num(*i));
    }
    slice_index
}


enum Op{
    Add(Arc<RefCell<TensorRaw>> , Arc<RefCell<TensorRaw>>),
    Sub(Arc<RefCell<TensorRaw>> , Arc<RefCell<TensorRaw>>),
    Mul(Arc<RefCell<TensorRaw>> , Arc<RefCell<TensorRaw>>),
    Div(Arc<RefCell<TensorRaw>> , Arc<RefCell<TensorRaw>>),
    MatMul(Arc<RefCell<TensorRaw>> , Arc<RefCell<TensorRaw>>),
    Exp(Arc<RefCell<TensorRaw>>),
    Neg(Arc<RefCell<TensorRaw>>),
    Sum(Arc<RefCell<TensorRaw>> , usize),
    Reshape(Arc<RefCell<TensorRaw>> ),
    Transpose(Arc<RefCell<TensorRaw>> , Vec<usize>),
    Assign(Arc<RefCell<TensorRaw>>),
    Slice(Arc<RefCell<TensorRaw>> , SliceIndex),
    Dot(Arc<RefCell<TensorRaw>> , Arc<RefCell<TensorRaw>>),
}
pub struct TensorRaw{
    raw : Arc<RefCell<Vec<f64>>>,
    shape : Vec<usize>,
    strides : Vec<usize>,
    offset : usize,
    op : Option<Op>,
    grad : Option<Vec<f64>>
}


fn permuate(shape : & Vec<usize> , axes : & Vec<usize>)-> Vec<usize>{
    let mut newshape = vec![0; shape.len()];
    for i in 0..shape.len(){
        newshape[i] = shape[axes[i]];
    }
    newshape
}
fn inverse_permuate(axes : & Vec<usize>)-> Vec<usize>{
    let mut newaxes = vec![0; axes.len()];
    for i in 0..axes.len(){
        newaxes[axes[i]] = i;
    }
    newaxes
}

impl TensorRaw{
    pub fn new( data : & Vec<f64>)-> TensorRaw{
        TensorRaw { raw: Arc::new(RefCell::new(data.clone())),
             shape: vec![data.len()], strides: vec![1], offset: 0 ,
             op : None,
                grad : None
            }
    }
    pub fn shape(self : & Self)-> Vec<usize>{
        self.shape.clone()
    }
    pub fn ones(shape : & Vec<usize>) -> TensorRaw{
        TensorRaw{
            raw : Arc::new(RefCell::new(vec![1.;shape.iter().product()])),
            shape : shape.clone(),
            strides : {
                match shape.len(){
                    0 => vec![],
                    _ => {
                        let mut strides = vec![1];
                        let mut prod = 1;
                        let mut rev_out_shape = shape.clone();
                        rev_out_shape.reverse();
                        for i in &rev_out_shape[0..(shape.len()-1)]  {
                            prod *= i;
                            strides.push(prod);
                        }
                        strides.reverse();
                        strides
                    }
                }
            },
            offset : 0,
             op : None,
                grad : None
        }
    }
        pub fn arange(start: f64, end: f64, step: f64) -> TensorRaw {
        assert!(step != 0.0, "step 不能为0");
        let mut data = Vec::new();
        let mut val = start;
        if step > 0.0 {
            while val < end {
                data.push(val);
                val += step;
            }
        } else {
            while val > end {
                data.push(val);
                val += step;
            }
        }
        let shape = vec![data.len()];
        TensorRaw {
            raw: Arc::new(RefCell::new(data)),
            shape,
            strides: vec![1],
            offset: 0,
             op : None,
                grad : None
        }
    }
    pub fn zeros(shape : & Vec<usize>) -> TensorRaw{
        TensorRaw{
            raw : Arc::new(RefCell::new(vec![0.;shape.iter().product()])),
            shape : shape.clone(),
            strides : {
                match shape.len(){
                    0 => vec![],
                    _ => {
                        let mut strides = vec![1];
                        let mut prod = 1;
                        let mut rev_out_shape = shape.clone();
                        rev_out_shape.reverse();
                        for i in &rev_out_shape[0..(shape.len()-1)]  {
                            prod *= i;
                            strides.push(prod);
                        }
                        strides.reverse();
                        strides
                    }
                }
            },
            offset : 0,
             op : None,
                grad : None
        }
    }
    fn pick(raw : & [f64],shape : & [usize],strides : & [usize], result : & mut Vec<f64>){
        if shape.len() == 1{
            for i in 0..shape[0]{
                result.push(raw[i*strides[0]]);
            }
            return
        }
        let dim = shape[0];
        let step = strides[0];
        for i in 0..dim{
            Self::pick(&raw[i*step..],&shape[1..],&strides[1..],result);
        }
    }
    pub fn raw(self : & Self) -> f64{
        self.raw.borrow()[0]
    }
    pub fn transpose(self : & Self , axes : Vec<usize>)-> TensorRaw{
        // [1 ,2 , 0] [k , m , n] -> [m , n , k]
        assert_eq!(self.shape.len(),axes.iter().collect::<std::collections::HashSet<_>>().len());
        let newshape = permuate(&self.shape , &axes);
        let mut newraw = Vec::with_capacity(newshape.iter().product());
        for i in MultiDimIterator::new(&newshape){
            let mut raw_index = 0;
            for j in 0..self.shape.len(){
                raw_index += i[axes[j]] * self.strides[j];
            }
            newraw.push(self.raw.borrow()[raw_index]);
        }
        TensorRaw {
            raw : Arc::new(RefCell::new(newraw)),
            strides : trivial_strides(&newshape),
            shape : newshape,
            offset : 0,
            op : None,
            grad : None
        }
    }
    pub fn reshape(self : & Self , shape : & Vec<usize>) -> TensorRaw{
        let mut res = self.clone();
        res.reshape_in_place(shape);
        res
    }
    pub fn reshape_in_place(self : & mut Self,shape : & Vec<usize>){ 
        assert_eq!(shape.iter().product::<usize>(),self.shape.iter().product());
        if self.raw.borrow().len() != self.shape.iter().product(){
            let mut newraw = Vec::new();
            Self::pick(&self.raw.borrow()[self.offset..], &self.shape, &self.strides, & mut newraw);
            self.raw = Arc::new(RefCell::new(newraw));
            self.offset = 0;
        }
        self.shape = shape.clone();
        self.strides = match shape.len(){
                    0 => vec![],
                    _ => {
                        let mut strides = vec![1];
                        let mut prod = 1;
                        let mut rev_out_shape = shape.clone();
                        rev_out_shape.reverse();
                        for i in &rev_out_shape[0..(shape.len()-1)]  {
                            prod *= i;
                            strides.push(prod);
                        }
                        strides.reverse();
                        strides
                    }
                };
    }
    pub fn views(self : & Self)-> TensorRaw{
        TensorRaw{
            raw : self.raw.clone(),
            shape : self.shape.clone(),
            strides : self.strides.clone(),
            offset:self.offset,
            op : None,
            grad : None
        }
    }

    pub fn get(self : & Self , index : SliceIndex) -> TensorRaw{
        assert!(index.len() <= self.shape.len());
        assert!(index.iter().filter(|&x| *x == SliceIndexUnit::Filler).count() <= 1);
        if index.is_empty(){
            self.views()
        }
        else{
            match index.contains(&SliceIndexUnit::Filler){
                true => { // expansion
                    let fillerpos = index.iter().position(|&x| x == SliceIndexUnit::Filler).unwrap();
                    let rindex = &index[fillerpos..];
                    let restdim = rindex.len() - 1;
                    let fillerdim = self.shape.len() - restdim - fillerpos;
                    self.get(
                        {
                            let mut res = index[0..fillerpos].to_vec();
                            for i in 0..fillerdim{
                                res.push(SliceIndexUnit::Slice((0,self.shape[i] , 1)))
                            }
                            res.extend_from_slice(&rindex[1..]);
                            res
                        }
                    )
                }
                false => {
                    let mut nshape = Vec :: new();
                    for i in 0..index.len(){
                        match index[i]{
                            SliceIndexUnit::Filler => {},
                            SliceIndexUnit::Num(_) => {},
                            SliceIndexUnit::Slice((start , end , step))=>{
                                assert_ne!(start , end);
                                assert!(end <= self.shape[i]);
                                let mut dim = 1 + (end - start) / step;
                                if start + step * (dim-1) == end {
                                    dim -= 1;
                                }
                                nshape.push(dim);
                            }
                        }
                    }
                    if index.len() < self.shape.len(){
                        nshape.extend_from_slice(&self.shape[index.len()..]);
                    }
                    TensorRaw{
                        raw : self.raw.clone(),
                        shape : nshape,
                        strides : {
                            let mut rstrides = Vec :: new();
                            let restdim = self.shape.len() - index.len();
                            for i in (self.shape.len()-restdim..self.shape.len()).rev(){
                                rstrides.push(self.strides[i]);
                            }
                            for i in (0..index.len()).rev(){
                                match index[i]{
                                    SliceIndexUnit::Filler => (),
                                    SliceIndexUnit::Num(_) => (),
                                    SliceIndexUnit::Slice((_ , _ , step))=>{
                                        rstrides.push(self.strides[i] * step);
                                    }
                                }
                            }
                            rstrides.reverse();
                            rstrides
                        },
                        offset : {
                            let mut noffset = self.offset;
                            for i in 0..index.len(){
                                match index[i]{
                                    SliceIndexUnit::Filler => (),
                                    SliceIndexUnit::Num(x) => {
                                        noffset += x * self.strides[i];
                                    }
                                    SliceIndexUnit::Slice((start , _ ,_))=>{
                                        noffset += start* self.strides[i];
                                    }
                                }
                            }
                            noffset
                        },
                    op : None,
                    grad : None

                        
                    }
                }
            }
        }
    }
    pub fn fetch(self : & Self , index : & Vec<usize>) -> f64{
        let mut raw_index = self.offset;
        for i in 0..self.shape.len(){
            raw_index += (index[index.len() - i-1] % self.shape[self.shape.len()-i - 1]) * self.strides[self.shape.len() - i -1];
        }
        self.raw.borrow()[raw_index]
    }
    fn set(self : & mut Self , index : & Vec<usize> , value : f64){
        let mut raw_index = self.offset;
        for i in 0..self.shape.len(){
            raw_index += (index[index.len() - i-1] % self.shape[self.shape.len()-i - 1]) * self.strides[self.shape.len() - i -1];
        }
        self.raw.borrow_mut()[raw_index] = value;
    }
    pub fn exp(self : & Self)-> TensorRaw{
        let mut new_raw = self.raw.borrow().clone();
        for v in &mut new_raw {
            *v = v.exp();
        }
        TensorRaw {
            raw: Arc::new(RefCell::new(new_raw)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            op : None,
            grad : None

        }
    }
}
impl Clone for TensorRaw {
    fn clone(&self) -> Self {
        let mut newraw = Vec::new();
        Self::pick(&self.raw.borrow()[self.offset..], &self.shape, &self.strides, & mut newraw);
        TensorRaw{
            raw : Arc::new(RefCell::new(newraw)),
            shape : self.shape.clone(),
            strides : {
                let shape = self.shape.clone();
                 match shape.len(){
                    0 => vec![],
                    _ => {
                        let mut strides = vec![1];
                        let mut prod = 1;
                        let mut rev_out_shape = shape.clone();
                        rev_out_shape.reverse();
                        for i in &rev_out_shape[0..(shape.len()-1)]  {
                            prod *= i;
                            strides.push(prod);
                        }
                        strides.reverse();
                        strides
                    }
                }
            },
            offset : 0,
            op : None,
            grad : None
        }
    }
}

impl Display for TensorRaw{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = &self.raw.borrow()[self.offset..];
        match self.shape.len(){
            0 => write!(f, "{:.4}" , data[0]), // 零维张量

            1 => {
                // 一维张量
                write!(f, "[")?;
                for i in  0..self.shape[0] {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:.4}", data[i*self.strides[0]])?;
                }
                write!(f, "]")
            },

            2 => {
                // 二维张量(矩阵)
                write!(f, "[")?;
                for i in 0..self.shape[0]{
                    write!(f,"[ ")?;
                    for j in 0..self.shape[1]{
                        if j > 0{
                            write!(f, ", ")?;
                        }
                        write!(f,"{:.4}",data[i * self.strides[0] + j * self.strides[1]])?;
                    }
                    write!(f," ]")?;
                    if i < self.shape[0]-1{
                        write!(f, ",\n")?;
                    }
                }
                write!(f,"]")
            }
            
            _ => {
                // 多维张量(递归处理)
                let dim = self.shape[0];

                write!(f, "[")?;
                for i in 0..dim {
                    let sub_tensor = TensorRaw {
                        raw: self.raw.clone(),
                        shape: self.shape[1..].to_vec(),
                        strides: self.strides[1..].to_vec(),
                        offset : self.offset+i*self.strides[0],
                        op : None,
                        grad : None
                    };
                    
                    // 递归格式化子张量
                    write!(f, "{}", sub_tensor)?;
                    if i != dim-1{
                        write!(f,",\n")?;
                    }
                }
                write!(f, "]")
            }
        }
        
    }
}


impl Neg for &TensorRaw {
    type Output = TensorRaw;
    fn neg(self) -> Self::Output {
        let mut new_raw = self.raw.borrow().clone();
        for v in &mut new_raw {
            *v = -*v;
        }
        TensorRaw {
            raw: Arc::new(RefCell::new(new_raw)),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            op : None,
            grad : None
        }
    }
}

fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
    let n = shape1.len().max(shape2.len());
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let dim1 = *shape1.get(shape1.len().wrapping_sub(i + 1)).unwrap_or(&1);
        let dim2 = *shape2.get(shape2.len().wrapping_sub(i + 1)).unwrap_or(&1);
        if dim1 % dim2 != 0 && dim2 % dim1 != 0 {
            panic!("Cannot broadcast shapes {:?} and {:?}", shape1, shape2);
        }
        result.push(dim1.max(dim2));
    }
    result.reverse();
    result
}


impl Add for &TensorRaw {
    type Output = TensorRaw;
    fn add(self, rhs: Self) -> Self::Output {
        let out_shape = broadcast_shapes(&self.shape, &rhs.shape);
        let out_len = out_shape.iter().product();
        let mut data = Vec::with_capacity(out_len);
        for i in MultiDimIterator::new(&out_shape) {
            data.push(self.fetch(&i) + rhs.fetch(&i));
        }
        TensorRaw {
            raw: Arc::new(RefCell::new(data)),
            shape: out_shape.clone(),
            strides: {
                match out_shape.len(){
                    0 => vec![],
                    _ => {
                        let mut strides = vec![1];
                        let mut prod = 1;
                        let mut rev_out_shape = out_shape.clone();
                        rev_out_shape.reverse();
                        for i in &rev_out_shape[0..(out_shape.len()-1)]  {
                            prod *= i;
                            strides.push(prod);
                        }
                        strides.reverse();
                        strides
                    }
                }
            },
            offset: 0,
            op : None,
            grad : None
        }
    }
}

impl Div for &TensorRaw {
    type Output = TensorRaw;
    fn div(self, rhs: Self) -> Self::Output {
        let out_shape = broadcast_shapes(&self.shape, &rhs.shape);
        let out_len = out_shape.iter().product();
        let mut data = Vec::with_capacity(out_len);
        for i in MultiDimIterator::new(&out_shape) {
            data.push(self.fetch(&i) / rhs.fetch(&i));
        }
        TensorRaw {
            raw: Arc::new(RefCell::new(data)),
            shape: out_shape.clone(),
            strides: {
                match out_shape.len(){
                    0 => vec![],
                    _ => {
                        let mut strides = vec![1];
                        let mut prod = 1;
                        let mut rev_out_shape = out_shape.clone();
                        rev_out_shape.reverse();
                        for i in &rev_out_shape[0..(out_shape.len()-1)]  {
                            prod *= i;
                            strides.push(prod);
                        }
                        strides.reverse();
                        strides
                    }
                }
            },
            offset: 0,
            op : None,
            grad : None
        }
    }
}


impl Sub for &TensorRaw {
    type Output = TensorRaw;
    fn sub(self, rhs: Self) -> Self::Output {
        let out_shape = broadcast_shapes(&self.shape, &rhs.shape);
        let out_len = out_shape.iter().product();
        let mut data = Vec::with_capacity(out_len);
        for i in MultiDimIterator::new(&out_shape) {
            data.push(self.fetch(&i) - rhs.fetch(&i));
        }
        TensorRaw {
            raw: Arc::new(RefCell::new(data)),
            shape: out_shape.clone(),
            strides: {

                match out_shape.len(){
                    0 => vec![],
                    _ => {
                        let mut strides = vec![1];
                        let mut prod = 1;
                        let mut rev_out_shape = out_shape.clone();
                        rev_out_shape.reverse();
                        for i in &rev_out_shape[0..(out_shape.len()-1)]  {
                            prod *= i;
                            strides.push(prod);
                        }
                        strides.reverse();
                        strides
                    }
                }
            },
            offset: 0,
            op : None,
            grad : None
        }
    }
}


impl Mul for &TensorRaw {
    type Output = TensorRaw;
    fn mul(self, rhs: Self) -> Self::Output {
        let out_shape = broadcast_shapes(&self.shape, &rhs.shape);
        let out_len = out_shape.iter().product();
        let mut data = Vec::with_capacity(out_len);
        for i in MultiDimIterator::new(&out_shape) {
            data.push(self.fetch(&i) * rhs.fetch(&i));
        }
        TensorRaw {
            raw: Arc::new(RefCell::new(data)),
            shape: out_shape.clone(),
            strides: {
                match out_shape.len(){
                    0 => vec![],
                    _ => {
                        let mut strides = vec![1];
                        let mut prod = 1;
                        let mut rev_out_shape = out_shape.clone();
                        rev_out_shape.reverse();
                        for i in &rev_out_shape[0..(out_shape.len()-1)]  {
                            prod *= i;
                            strides.push(prod);
                        }
                        strides.reverse();
                        strides
                    }
                }
            },
            offset: 0,
            op : None,
            grad : None
        }
    }
}

impl From<f64> for TensorRaw {
    fn from(val: f64) -> Self {
        TensorRaw {
            raw: Arc::new(RefCell::new(vec![val])),
            shape: vec![],
            strides: vec![],
            offset: 0,
            op : None,
            grad : None
        }
    }
}

impl TensorRaw {
    pub fn dot(&self , rhs : &TensorRaw) -> TensorRaw{
        assert!(self.shape.len() == rhs.shape.len()+1 && self.shape.len() >= 2 , "dot Axes Error.");
        assert_eq!(self.shape[0..self.shape.len()-2] , rhs.shape[0..rhs.shape.len()-1] , "dot Shape Error.");
        assert_eq!(self.shape[self.shape.len()-1] , rhs.shape[rhs.shape.len()-1] , "dot Matrix Not Match.");
        let mut out_shape = self.shape[0..self.shape.len()-1].to_vec();
        let mut out_raw = Vec::with_capacity(out_shape.iter().product());
        for idx in MultiDimIterator::new(&out_shape[0..out_shape.len() - 1]){
            for i in 0..self.shape[self.shape.len() - 2]{
                let mut val = 0.;
                for j in 0..self.shape[self.shape.len()-1]{
                    let mut aidx = idx.clone();
                    aidx.extend(vec![i , j]);
                    let mut bidx = idx.clone();
                    bidx.extend(vec![j]);
                    val += self.fetch(&aidx) * rhs.fetch(&bidx);
                }
                out_raw.push(val);
            }
        }
        TensorRaw { raw : Arc::new(RefCell::new(out_raw)) , shape : out_shape.clone(), 
            strides : {
                match out_shape.len(){
                    0 => vec![],
                    _ => {
                        let mut strides = vec![1];
                        let mut prod = 1;
                        let mut rev_out_shape = out_shape.clone();
                        rev_out_shape.reverse();
                        for i in &rev_out_shape[0..(out_shape.len()-1)]  {
                            prod *= i;
                            strides.push(prod);
                        }
                        strides.reverse();
                        strides
                    }
                }
            }
            , offset : 0 ,
            op : None,
            grad : None
        }

    }
    pub fn matmul(&self, rhs: &TensorRaw) -> TensorRaw {
        assert!(self.shape.len() == rhs.shape.len() && self.shape.len() >= 2 , "matmul Not Enough Axes.");
        assert_eq!(self.shape[0..self.shape.len() - 2] , rhs.shape[0..self.shape.len() - 2] , "matmul Axes Not Equal.");
        assert_eq!(self.shape[self.shape.len() - 1] , rhs.shape[self.shape.len() - 2] , "matmul Matrix Not Match.");
        let mut out_shape = self.shape[0..self.shape.len() - 2].to_vec();
        out_shape.extend(vec![self.shape[self.shape.len() - 2] , rhs.shape[rhs.shape.len() - 1]]);
        let mut out_raw = Vec::with_capacity(out_shape.iter().product());
        for idx in MultiDimIterator::new(&out_shape[0..out_shape.len() - 2]){
            for i in 0..self.shape[self.shape.len() - 2]{
                for j in 0..rhs.shape[rhs.shape.len() - 1]{
                    let mut val = 0.;
                    for k in 0..self.shape[self.shape.len() - 1]{
                        let mut aidx = idx.clone();
                        aidx.extend(vec![i , k]);
                        let mut bidx = idx.clone();
                        bidx.extend(vec![k , j]);
                        val += self.fetch(&aidx) * rhs.fetch(&bidx);
                    }
                    out_raw.push(val);
                }
            }
        }
        TensorRaw { raw: Arc::new(RefCell::new(out_raw)) , shape: out_shape.clone(), 
            strides: 
            {

                match out_shape.len(){
                    0 => vec![],
                    _ => {
                        let mut strides = vec![1];
                        let mut prod = 1;
                        let mut rev_out_shape = out_shape.clone();
                        rev_out_shape.reverse();
                        for i in &rev_out_shape[0..(out_shape.len()-1)]  {
                            prod *= i;
                            strides.push(prod);
                        }
                        strides.reverse();
                        strides
                    }
                }
            }
            , offset: 0 ,
            op : None,
            grad : None
        }
    }
}
impl TensorRaw {
    pub fn iden(n: usize) -> TensorRaw {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        TensorRaw {
            raw: Arc::new(RefCell::new(data)),
            shape: vec![n, n],
            strides: vec![n, 1],
            offset: 0,
            op : None,
            grad : None
        }
    }
}

impl TensorRaw{
    fn sum(self : & Self , axis : usize) -> TensorRaw{
        assert!(axis < self.shape.len() , "Axis Out Of Range.");
        let mut out_shape = self.shape.clone();
        out_shape.remove(axis);
        let mut out_raw = Vec::with_capacity(out_shape.iter().product());
        for idx in MultiDimIterator::new(&out_shape){
            let mut val = 0.;
            for i in 0..self.shape[axis]{
                let mut aidx = idx.clone();
                aidx.insert(axis , i);
                val += self.fetch(&aidx);
            }
            out_raw.push(val);
        }
        TensorRaw {
            raw: Arc::new(RefCell::new(out_raw)),
            shape: out_shape.clone(),
            strides: {
                match out_shape.len(){
                    0 => vec![],
                    _ => {
                        let mut strides = vec![1];
                        let mut prod = 1;
                        let mut rev_out_shape = out_shape.clone();
                        rev_out_shape.reverse();
                        for i in &rev_out_shape[0..(out_shape.len()-1)]  {
                            prod *= i;
                            strides.push(prod);
                        }
                        strides.reverse();
                        strides
                    }
                }

            },
            offset: 0,
            op : None,
            grad : None
        }


    }
}
impl TensorRaw{
    pub fn assign(self : & mut Self , rhs : & TensorRaw){
        assert_eq!(self.shape , self.shape);
        for i in MultiDimIterator::new(&self.shape){
            self.set(&i , rhs.fetch(&i));
        }
    }
    pub fn rand(shape : & Vec<usize>)-> TensorRaw{
        let mut data = Vec::new();
        for _ in 0..shape.iter().product::<usize>(){
            data.push(rand::random::<f64>());
        }
        TensorRaw{
            raw : Arc::new(RefCell::new(data)),
            shape : shape.clone(),
            strides : {
                match shape.len(){
                    0 => vec![],
                    _ => {
                        let mut strides = vec![1];
                        let mut prod = 1;
                        let mut rev_out_shape = shape.clone();
                        rev_out_shape.reverse();
                        for i in &rev_out_shape[0..(shape.len()-1)]  {
                            prod *= i;
                            strides.push(prod);
                        }
                        strides.reverse();
                        strides
                    }
                }
            },
            offset : 0,
             op : None,
                grad : None
        }
    }
    pub fn randbetween(shape : & Vec<usize> , start : f64 , end : f64)-> TensorRaw{
        let mut data = Vec::new();
        for _ in 0..shape.iter().product::<usize>(){
            data.push(rand::random::<f64>() * (end - start) + start);
        }
        TensorRaw{
            raw : Arc::new(RefCell::new(data)),
            shape : shape.clone(),
            strides : {
                match shape.len(){
                    0 => vec![],
                    _ => {
                        let mut strides = vec![1];
                        let mut prod = 1;
                        let mut rev_out_shape = shape.clone();
                        rev_out_shape.reverse();
                        for i in &rev_out_shape[0..(shape.len()-1)]  {
                            prod *= i;
                            strides.push(prod);
                        }
                        strides.reverse();
                        strides
                    }
                }
            },
            offset : 0,
             op : None,
                grad : None
        }
    }
}

struct Tensor(Arc<RefCell<TensorRaw>>);

//impl all the functions from tensorraw for tensor
impl Clone for Tensor {
    fn clone(&self) -> Self {
        let res = Tensor(Arc::new(RefCell::new(self.0.borrow().clone())));
        res.0.borrow_mut().op = None;// Clear the operation history
        res.0.borrow_mut().grad = None;
        res
    }
}
impl Display for Tensor{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.borrow())
    }
}
impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((&*self.0.borrow() + &*rhs.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Add(self.0.clone(), rhs.0.clone()));
        res
    }
}
impl Add<Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((&*self.0.borrow() + &*rhs.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Add(self.0.clone(), rhs.0.clone()));
        res
    }
}
impl Add<&Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((&*self.0.borrow() + &*rhs.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Add(self.0.clone(), rhs.0.clone()));
        res
    }
}
impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((&*self.0.borrow() + &*rhs.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Add(self.0.clone(), rhs.0.clone()));
        res
    }
}
impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Self) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((&*self.0.borrow() - &*rhs.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Sub(self.0.clone(), rhs.0.clone()));
        res
    }
}
impl Sub<Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((&*self.0.borrow() - &*rhs.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Sub(self.0.clone(), rhs.0.clone()));
        res
    }
}
impl Sub<&Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((&*self.0.borrow() - &*rhs.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Sub(self.0.clone(), rhs.0.clone()));
        res
    }
}
impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((&*self.0.borrow() - &*rhs.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Sub(self.0.clone(), rhs.0.clone()));
        res
    }
}
impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((&*self.0.borrow() * &*rhs.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Mul(self.0.clone(), rhs.0.clone()));
        res
    }
}
impl Mul<Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((&*self.0.borrow() * &*rhs.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Mul(self.0.clone(), rhs.0.clone()));
        res
    }
}
impl Mul<&Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((&*self.0.borrow() * &*rhs.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Mul(self.0.clone(), rhs.0.clone()));
        res
    }
}
impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((&*self.0.borrow() * &*rhs.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Mul(self.0.clone(), rhs.0.clone()));
        res
    }
}
impl Div for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Self) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((&*self.0.borrow() / &*rhs.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Div(self.0.clone(), rhs.0.clone()));
        res
    }
}
impl Div<Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((&*self.0.borrow() / &*rhs.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Div(self.0.clone(), rhs.0.clone()));
        res
    }
}
impl Div<&Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((&*self.0.borrow() / &*rhs.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Div(self.0.clone(), rhs.0.clone()));
        res
    }
}
impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((&*self.0.borrow() / &*rhs.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Div(self.0.clone(), rhs.0.clone()));
        res
    }
}

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((-&*self.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Neg(self.0.clone()));
        res
    }
}
impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        let res = Tensor(Arc::new(RefCell::new((-&*self.0.borrow()).clone())));
        res.0.borrow_mut().op = Some(Op::Neg(self.0.clone()));
        res
    }
}
impl Tensor{
    pub fn new(data : & Vec<f64>)-> Tensor{
        Tensor(Arc::new(RefCell::new(TensorRaw::new(data))))
    }
    pub fn shape(self : & Self)-> Vec<usize>{
        self.0.borrow().shape()
    }
    pub fn raw(self : & Self)-> f64{
        self.0.borrow().raw()
    }
    pub fn fetch(self : & Self , index : & Vec<usize>) -> f64{
        self.0.borrow().fetch(index)
    }
}
impl Tensor {
    pub fn matmul(self : & Self , rhs : & Tensor) -> Tensor{
        let res = Tensor(Arc::new(RefCell::new((self.0.borrow().matmul(&rhs.0.borrow())).clone())));
        res.0.borrow_mut().op = Some(Op::MatMul(self.0.clone(), rhs.0.clone()));
        res
    }
    pub fn dot(self : & Self , rhs : & Tensor) -> Tensor{
        let res = Tensor(Arc::new(RefCell::new((self.0.borrow().dot(&rhs.0.borrow())).clone())));
        res.0.borrow_mut().op = Some(Op::Dot(self.0.clone(), rhs.0.clone()));
        res
    }
}
impl Tensor {
    pub fn exp(self : & Self) -> Tensor{
        let res = Tensor(Arc::new(RefCell::new((self.0.borrow().exp()).clone())));
        res.0.borrow_mut().op = Some(Op::Exp(self.0.clone()));
        res
    }
}
impl Tensor {
    pub fn iden(n: usize) -> Tensor {
        let res = Tensor(Arc::new(RefCell::new(TensorRaw::iden(n))));
        res
    }
}
impl Tensor {
    pub fn sum(self : & Self , axis : usize) -> Tensor{
        let res = Tensor(Arc::new(RefCell::new((self.0.borrow().sum(axis)).clone())));
        res.0.borrow_mut().op = Some(Op::Sum(self.0.clone(), axis));
        res
    }
}
impl Tensor {
    pub fn reshape(self : & mut Self , shape : & Vec<usize>) -> Tensor{
        let res = Tensor(Arc::new(RefCell::new((self.0.borrow().reshape(shape)).clone())));
        res.0.borrow_mut().op = Some(Op::Reshape(self.0.clone()));
        res
    }
}
impl Tensor {
    pub fn transpose(self : & mut Self , axes : Vec<usize>) -> Tensor{
        let res = Tensor(Arc::new(RefCell::new((self.0.borrow().transpose(axes.clone())).clone())));
        res.0.borrow_mut().op = Some(Op::Transpose(self.0.clone() , axes));
        res
    }
}
impl Tensor {
    pub fn assign(self : & mut Self , rhs : & Tensor) {
        self.0.borrow_mut().assign(&rhs.0.borrow());
        self.0.borrow_mut().op = Some(Op::Assign(rhs.0.clone()));
    }
}
impl Tensor{
    pub fn get(self : & mut Self , index : SliceIndex) -> Tensor{
        let res = Tensor(Arc::new(RefCell::new(self.0.borrow().get(index.clone()))));
        res.0.borrow_mut().op = Some(Op::Slice(self.0.clone(), index));
        res
    }
    fn set(self : & mut Self , index : & Vec<usize> , value : f64){
        self.0.borrow_mut().set(index , value);
    }
    pub fn views(self : & mut Self)-> Tensor{
        let res = Tensor(Arc::new(RefCell::new((self.0.borrow().views()).clone())));
        res.0.borrow_mut().op = Some(Op::Slice(self.0.clone(), vec![]));
        res
    }
    pub fn arange(start: f64, end: f64, step: f64) -> Tensor {
        let res = Tensor(Arc::new(RefCell::new(TensorRaw::arange(start, end, step))));
        res
    }
    pub fn zeros(shape : & Vec<usize>) -> Tensor{
        let res = Tensor(Arc::new(RefCell::new(TensorRaw::zeros(shape))));
        res
    }
    pub fn ones(shape : & Vec<usize>) -> Tensor{
        let res = Tensor(Arc::new(RefCell::new(TensorRaw::ones(shape))));
        res
    }
    pub fn rand(shape : & Vec<usize>)-> Tensor{
        let res = Tensor(Arc::new(RefCell::new(TensorRaw::rand(shape))));
        res
    }
    pub fn randbetween(shape : & Vec<usize> , start : f64 , end : f64)-> Tensor{
        let res = Tensor(Arc::new(RefCell::new(TensorRaw::randbetween(shape , start , end))));
        res
    }
}

fn trivial_strides(shape : & Vec<usize>)-> Vec<usize>{
    match shape.len(){
        0 => vec![],
        _ => {
            let mut strides = vec![1];
            let mut prod = 1;
            let mut rev_out_shape = shape.clone();
            rev_out_shape.reverse();
            for i in &rev_out_shape[0..(shape.len()-1)]  {
                prod *= i;
                strides.push(prod);
            }
            strides.reverse();
            strides
        }
    }
}

impl TensorRaw{
    pub fn init_grad(self : & mut Self){
        // clear all the childrens of op using recursion
        match &self.op{
            None => (),
            Some(Op :: Assign( a ))=>{
                a.borrow_mut().grad = Some(vec![0.;self.shape.iter().product()]);
                a.borrow_mut().init_grad();
            }
            Some(Op :: Add( a , b))=>{ 
                a.borrow_mut().grad = Some(vec![0.;self.shape.iter().product()]);
                b.borrow_mut().grad = Some(vec![0.;self.shape.iter().product()]);
                a.borrow_mut().init_grad();
                b.borrow_mut().init_grad();
            }
            Some(Op :: Sub( a , b))=>{ a.borrow_mut().grad = Some(vec![0.;self.shape.iter().product()]);
                b.borrow_mut().grad = Some(vec![0.;self.shape.iter().product()]);
                a.borrow_mut().init_grad();
                b.borrow_mut().init_grad();
            }
            Some(Op :: Mul( a , b))=>{
                a.borrow_mut().grad = Some(vec![0.;self.shape.iter().product()]);
                b.borrow_mut().grad = Some(vec![0.;self.shape.iter().product()]);
                a.borrow_mut().init_grad();
                b.borrow_mut().init_grad();
            }
            Some(Op :: Div( a , b))=>{
                a.borrow_mut().grad = Some(vec![0.;self.shape.iter().product()]);
                b.borrow_mut().grad = Some(vec![0.;self.shape.iter().product()]);
                a.borrow_mut().init_grad();
                b.borrow_mut().init_grad();
            }
            Some(Op :: Neg( a ))=>{
                a.borrow_mut().grad = Some(vec![0.;self.shape.iter().product()]);
                a.borrow_mut().init_grad();
            }
            Some(Op :: MatMul( a , b))=>{
                let alen: usize = a.borrow().shape.iter().product();
                let blen: usize = b.borrow().shape.iter().product();
                a.borrow_mut().grad = Some(vec![0.;alen]);
                b.borrow_mut().grad = Some(vec![0.;blen]);
                a.borrow_mut().init_grad();
                b.borrow_mut().init_grad();
            }
            Some(Op :: Exp( a ))=>{
                a.borrow_mut().grad = Some(vec![0.;self.shape.iter().product()]);
                a.borrow_mut().init_grad();
            }
            Some(Op :: Sum( a , _))=>{
                let alen: usize = a.borrow().shape.iter().product();
                a.borrow_mut().grad = Some(vec![0.;alen]);
                a.borrow_mut().init_grad();
            }
            Some(Op :: Reshape( a ))=>{
                let alen: usize = a.borrow().shape.iter().product();
                a.borrow_mut().grad = Some(vec![0.;alen]);
                a.borrow_mut().init_grad();
            }
            Some(Op :: Transpose( a  , _))=>{
                let alen: usize = a.borrow().shape.iter().product();
                a.borrow_mut().grad = Some(vec![0.;alen]);
                a.borrow_mut().init_grad();
            }
            Some(Op :: Slice(a , index)) => {
                a.borrow_mut().init_grad();
                if(a.borrow().grad.is_none()){
                    let alen: usize = a.borrow().shape.iter().product();
                    a.borrow_mut().grad = Some(vec![0.;alen]);
                }
                else{
                    //only clear the portion of the tensor that is used in the slice
                    let shape = a.borrow().shape.clone();
                    let strides = {
                        match shape.len(){
                        0 => vec![],
                        _ => {
                            let mut strides = vec![1];
                            let mut prod = 1;
                            let mut rev_out_shape: Vec<usize> = shape.clone();
                            rev_out_shape.reverse();
                            for i in &rev_out_shape[0..(shape.len()-1)]  {
                                prod *= i;
                                strides.push(prod);
                            }
                            strides.reverse();
                            strides
                        }
                        }
                    };
                    let nstrides = {
                        let mut rstrides = Vec :: new();
                        let restdim = shape.len() - index.len();
                        for i in (shape.len()-restdim..shape.len()).rev(){
                            rstrides.push(strides[i]);
                        }
                        for i in (0..index.len()).rev(){
                            match index[i]{
                                SliceIndexUnit::Filler => (),
                                SliceIndexUnit::Num(_) => (),
                                SliceIndexUnit::Slice((_ , _ , step))=>{
                                    rstrides.push(strides[i] * step);
                                }
                            }
                        }
                        rstrides.reverse();
                        rstrides
                    };
                    let noffset = {
                        let mut noffset = 0;
                        for i in 0..index.len(){
                            match index[i]{
                                SliceIndexUnit::Filler => (),
                                SliceIndexUnit::Num(x) => {
                                    noffset += x * strides[i];
                                }
                                SliceIndexUnit::Slice((start , _ ,_))=>{
                                    noffset += start* strides[i];
                                }
                            }
                        }
                        noffset
                    };
                    let mut nshape = Vec :: new();
                    for i in 0..index.len(){
                        match index[i]{
                            SliceIndexUnit::Filler => {},
                            SliceIndexUnit::Num(_) => {},
                            SliceIndexUnit::Slice((start , end , step))=>{
                                assert_ne!(start , end);
                                assert!(end <= shape[i]);
                                let mut dim = 1 + (end - start) / step;
                                if start + step * (dim-1) == end {
                                    dim -= 1;
                                }
                                nshape.push(dim);
                            }
                        }
                    }
                    if index.len() < shape.len(){
                        nshape.extend_from_slice(&shape[index.len()..]);
                    }
                    for i in MultiDimIterator::new(&nshape){
                        let mut raw_index = noffset;
                        for j in 0..nshape.len(){
                            raw_index += (i[nshape.len() - j-1] % nshape[nshape.len()-j - 1]) * nstrides[nshape.len() - j -1];
                        }
                        a.borrow_mut().grad.as_mut().unwrap()[raw_index] = 0.;
                    }
                }
            },
            Some(Op::Dot( a, b)) => {
                a.borrow_mut().grad = Some(vec![0.;a.borrow().shape.iter().product()]);
                b.borrow_mut().grad = Some(vec![0.;b.borrow().shape.iter().product()]);
                a.borrow_mut().init_grad();
                b.borrow_mut().init_grad();
            }
        }
    }
    pub fn back_prop(self : & mut Self){
        match &self.op{
            None => (),
            Some(Op::Add(a, b)) => {
                a.borrow_mut().grad.as_mut().unwrap().iter_mut().zip(self.grad.as_ref().unwrap()).for_each(|(x, y)| *x += *y);
                b.borrow_mut().grad.as_mut().unwrap().iter_mut().zip(self.grad.as_ref().unwrap()).for_each(|(x, y)| *x += *y);
                a.borrow_mut().back_prop();
                b.borrow_mut().back_prop();
            },
            Some(Op::Sub(a, b)) => {
                a.borrow_mut().grad.as_mut().unwrap().iter_mut().zip(self.grad.as_ref().unwrap()).for_each(|(x, y)| *x += *y);
                b.borrow_mut().grad.as_mut().unwrap().iter_mut().zip(self.grad.as_ref().unwrap()).for_each(|(x, y)| *x -= *y);
                a.borrow_mut().back_prop();
                b.borrow_mut().back_prop();
            },
            Some(Op::Mul(a, b)) => {
                let strides = trivial_strides(&self.shape);
                for i in MultiDimIterator::new(&self.shape){
                    let mut raw_index = 0;
                    for j in 0..self.shape.len(){
                        raw_index += (i[self.shape.len() - j-1] % self.shape[self.shape.len()-j - 1]) * strides[self.shape.len() - j -1];
                    }
                    let aval = a.borrow().fetch(&i);
                    let bval = b.borrow().fetch(&i);
                    a.borrow_mut().grad.as_mut().unwrap()[raw_index] += bval * self.grad.as_ref().unwrap()[raw_index];
                    b.borrow_mut().grad.as_mut().unwrap()[raw_index] += aval * self.grad.as_ref().unwrap()[raw_index];
                }
                a.borrow_mut().back_prop();
                b.borrow_mut().back_prop();
            },
            Some(Op::Div(a, b)) => {
                // z = (x/y) dz/dy = -x/(y^2) * dz/dx
                let strides = trivial_strides(&self.shape);
                for i in MultiDimIterator::new(&self.shape){
                    let mut raw_index = 0;
                    for j in 0..self.shape.len(){
                        raw_index += (i[self.shape.len() - j-1] % self.shape[self.shape.len()-j - 1]) * strides[self.shape.len() - j -1];

                    }
                    a.borrow_mut().grad.as_mut().unwrap()[raw_index] += 1.0 / b.borrow().fetch(&i) * self.grad.as_ref().unwrap()[raw_index];
                    let bval = b.borrow().fetch(&i);
                    b.borrow_mut().grad.as_mut().unwrap()[raw_index] -= a.borrow().fetch(&i) / (bval * bval) * self.grad.as_ref().unwrap()[raw_index];
                }
                a.borrow_mut().back_prop();
                b.borrow_mut().back_prop();
            },
            Some(Op::Neg(a)) => {
                a.borrow_mut().grad.as_mut().unwrap().iter_mut().zip(self.grad.as_ref().unwrap()).for_each(|(x, y)| *x -= *y);
                a.borrow_mut().back_prop();
            },
            Some(Op::MatMul(a,b)) => {
                let ashape = a.borrow().shape.clone();
                let bshape = b.borrow().shape.clone();
                if self.shape.len() == 2{
                    for i in 0..ashape[0]{
                        for j in 0..ashape[1]{
                            for k in 0..bshape[1]{
                                a.borrow_mut().grad.as_mut().unwrap()[i * ashape[1]+ j] += b.borrow().fetch(&vec![j,k]) * self.grad.as_ref().unwrap()[i + k * self.shape[1]];
                            }
                        }
                    }
                    for i in 0..bshape[0]{
                        for j in 0..bshape[1]{
                            for k in 0..ashape[0]{
                                b.borrow_mut().grad.as_mut().unwrap()[i * bshape[1]+j] += a.borrow().fetch(&vec![k,i]) * self.grad.as_ref().unwrap()[k + j * self.shape[1]];
                            }
                        }
                    }
                }
                else{
                    let strides = trivial_strides(&self.shape[0..self.shape.len()-2].to_vec());
                    for idx in MultiDimIterator::new(&ashape[0..ashape.len()-2]){
                        let mut raw_offset = 0;
                        for l in 0..ashape.len()-2{
                            raw_offset += (idx[l] % ashape[l]) * strides[l];
                        }
                        let aoffset = raw_offset*ashape[ashape.len()-2]*ashape[ashape.len()-1];
                        let selfoffset = raw_offset*self.shape[self.shape.len()-2]*self.shape[self.shape.len()-1];
                        let boffset = raw_offset*bshape[bshape.len()-2]*bshape[bshape.len()-1];
                        for i in 0..ashape[ashape.len()-2]{
                            for j in 0..ashape[ashape.len()-1]{
                                for k in 0..bshape[bshape.len()-1]{
                                    let mut bidx = idx.clone();
                                    bidx.extend_from_slice(&vec![j,k]);
                                    a.borrow_mut().grad.as_mut().unwrap()[aoffset + i * ashape[(ashape.len())-1]+ j] += b.borrow().fetch(&bidx) * self.grad.as_ref().unwrap()[selfoffset+ i + k * self.shape[self.shape.len()-1]];
                                }
                            }
                        }
                        for i in 0..bshape[b.borrow().shape.len()-2]{
                            for j in 0..bshape[bshape.len()-1]{
                                for k in 0..ashape[bshape.len()-2]{
                                    let mut aidx = idx.clone();
                                    aidx.extend_from_slice(&vec![k,i]);
                                    b.borrow_mut().grad.as_mut().unwrap()[boffset + i * bshape[bshape.len()-1]+j] += a.borrow().fetch(&aidx) * self.grad.as_ref().unwrap()[selfoffset+k + j * self.shape[self.shape.len()-1]];
                                }
                            }
                        }
                    }
                }
                a.borrow_mut().back_prop();
                b.borrow_mut().back_prop();
            },
            Some(Op::Exp(a)) => {
                for i in MultiDimIterator::new(&a.borrow().shape){
                    let mut raw_index = 0;
                    for j in 0..a.borrow().shape.len(){
                        raw_index += (i[a.borrow().shape.len() - j-1] % a.borrow().shape[a.borrow().shape.len()-j - 1]) * a.borrow().strides[j];
                    }
                    a.borrow_mut().grad.as_mut().unwrap()[raw_index] += self.fetch(&i) * self.grad.as_ref().unwrap()[raw_index];
                }
                a.borrow_mut().back_prop();
            },
            Some(Op::Sum(a, axis)) => {
                let astrides = trivial_strides(a.borrow().shape.as_ref());
                let ashape = a.borrow().shape.clone();
                let selfstrides = trivial_strides(self.shape.as_ref());
                for i in MultiDimIterator::new(&ashape){
                    let mut raw_index = 0;
                    for j in 0..ashape.len(){
                        raw_index += (i[ashape.len() - j-1] % ashape[ashape.len()-j - 1]) * astrides[j];
                    }
                    let mut idx = i.clone();
                    idx.remove(*axis);
                    let mut raw_index2 = 0;
                    for j in 0..idx.len(){
                        raw_index2 += (idx[idx.len() - j-1] % self.shape[idx.len()-j - 1]) * selfstrides[j];
                    }
                    a.borrow_mut().grad.as_mut().unwrap()[raw_index] += self.grad.as_ref().unwrap()[raw_index2];
                }
                a.borrow_mut().back_prop();
            },
            Some(Op::Reshape(a)) => {
                a.borrow_mut().grad.as_mut().unwrap().iter_mut().zip(self.grad.as_ref().unwrap()).for_each(|(x, y)| *x += *y);
                a.borrow_mut().back_prop();
            },
            Some(Op::Transpose(a , axes)) => {
                // [1 ,2 , 0] [k , m , n] -> [m , n , k]
                let ashape = a.borrow().shape.clone();
                let mut astrides = trivial_strides(&ashape);
                let mut selfstrides = trivial_strides(&self.shape);
                for i in MultiDimIterator::new(&ashape){
                    let mut raw_index = 0;
                    for j in 0..ashape.len(){
                        raw_index += (i[ashape.len() - j-1] % ashape[ashape.len()-j - 1]) * astrides[j];
                    }
                    let mut idx = i.clone();
                    for j in 0..idx.len(){
                        idx[j] = i[axes[j]];
                    }
                    let mut raw_index2 = 0;
                    for j in 0..idx.len(){
                        raw_index2 += (idx[idx.len() - j-1] % self.shape[idx.len()-j - 1]) * selfstrides[j];
                    }
                    a.borrow_mut().grad.as_mut().unwrap()[raw_index] += self.grad.as_ref().unwrap()[raw_index2];
                }

                a.borrow_mut().back_prop();


            },
            Some(Op::Slice(a,index)) => {
                let ashape = a.borrow().shape.clone();
                let astrides = trivial_strides(&ashape);
                let selfstrides = trivial_strides(&self.shape);
                for i in MultiDimIterator::new(&self.shape){
                    let mut raw_index = 0;
                    for j in 0..self.shape.len(){
                        raw_index += (i[self.shape.len() - j - 1] % self.shape[self.shape.len()-j - 1]) * selfstrides[j];
                    }
                    let mut idx = Vec::new();
                    let mut tot = 0;
                    for j in 0..ashape.len(){
                        match index[j]{
                            SliceIndexUnit::Filler => (),
                            SliceIndexUnit::Num(x) => idx.push(x),
                            SliceIndexUnit::Slice((start , end , step))=>{
                                let mut dim = 1 + (end - start) / step;
                                if start + step * (dim-1) == end {
                                    dim -= 1;
                                }
                                idx.push(start + step * (i[tot] % dim));
                                tot += 1;
                            }
                        }
                    }
                    let mut raw_index2 = 0;
                    for j in 0..idx.len(){
                        raw_index2 += (idx[idx.len() - j-1] % ashape[idx.len()-j - 1]) * astrides[idx.len() - j - 1];
                    }
                    a.borrow_mut().grad.as_mut().unwrap()[raw_index2] += self.grad.as_ref().unwrap()[raw_index];
                }
                a.borrow_mut().back_prop();
            },
            Some(Op::Assign(a)) => {
                let shape = a.borrow().shape.clone();
                let strides = trivial_strides(&shape);
                for i in MultiDimIterator::new(&shape){
                    let mut raw_index = 0;
                    for j in 0..shape.len(){
                        raw_index += (i[shape.len() - j-1] % shape[shape.len()-j - 1]) * strides[shape.len() - j -1];
                    }
                    a.borrow_mut().grad.as_mut().unwrap()[raw_index] += self.grad.as_ref().unwrap()[raw_index];
                }
                a.borrow_mut().back_prop();
            },
            Some(Op::Dot(a, b)) => {
                let ashape = a.borrow().shape.clone();
                let bshape = b.borrow().shape.clone();
                if ashape.len() == 2{
                    for i in 0..ashape[0]{
                        for j in 0..ashape[1]{
                            a.borrow_mut().grad.as_mut().unwrap()[i * ashape[1]+ j] += b.borrow().fetch(&vec![j]) * self.grad.as_ref().unwrap()[i];
                        }
                    }
                    for i in 0..bshape[0]{
                        for j in 0..ashape[0]{
                            b.borrow_mut().grad.as_mut().unwrap()[i] += a.borrow().fetch(&vec![j,i]) * self.grad.as_ref().unwrap()[j];
                        }
                    }
                }
                else{
                    let strides = trivial_strides(&self.shape[0..self.shape.len()-1].to_vec());
                    for idx in MultiDimIterator::new(&ashape[0..ashape.len()-2]){
                        let mut raw_offset = 0;
                        for l in 0..ashape.len()-2{
                            raw_offset += (idx[l] % ashape[l]) * strides[l];
                        }
                        let aoffset = raw_offset*ashape[ashape.len()-2]*ashape[ashape.len()-1];
                        let selfoffset = raw_offset*self.shape[self.shape.len()-1];
                        let boffset = raw_offset*bshape[bshape.len()-1];
                        for i in 0..ashape[ashape.len()-2]{
                            for j in 0..ashape[ashape.len()-1]{
                                let mut bidx = idx.clone();
                                bidx.extend_from_slice(&vec![j]);
                                a.borrow_mut().grad.as_mut().unwrap()[aoffset + i * ashape[(ashape.len())-1]+ j] += b.borrow().fetch(&bidx) * self.grad.as_ref().unwrap()[selfoffset + i];
                            }
                        }
                        for i in 0..bshape[b.borrow().shape.len()-1]{
                            for j in 0..ashape[ashape.len()-2]{
                                let mut aidx = idx.clone();
                                aidx.extend_from_slice(&vec![j,i]);
                                b.borrow_mut().grad.as_mut().unwrap()[boffset + i] = a.borrow().fetch(&aidx) * self.grad.as_ref().unwrap()[selfoffset + j];
                            }
                        }
                    }
                }
                a.borrow_mut().back_prop();
                b.borrow_mut().back_prop();

            },
        }
    }
    fn print_grad(self : & Self){
        let tmp = TensorRaw{
            raw : Arc::new(RefCell:: new(self.grad.as_ref().unwrap().clone())),
            shape : self.shape.clone(),
            strides : trivial_strides(self.shape.as_ref()),
            offset : 0,
            op : None,
            grad : None
        };
        println!("{}", tmp);
    }
    fn set_grad(self : & mut Self , grad : & TensorRaw){
        assert_eq!(self.shape , grad.shape);
        let mut result = Vec::new();
        Self::pick(&*grad.raw.borrow_mut(), &grad.shape, &grad.strides, & mut result);
        self.grad = Some(result);
    }
}

impl Tensor{
    pub fn init_grad(self : & mut Self){
        self.0.borrow_mut().init_grad();
    }
    pub fn back_prop(self : & mut Self){
        self.0.borrow_mut().init_grad();
        self.0.borrow_mut().back_prop();
    }
    pub fn print_grad(self : & Self){
        self.0.borrow().print_grad();
    }
    pub fn set_grad(self : & mut Self , grad : & Tensor){
        self.0.borrow_mut().set_grad(&grad.0.borrow());
    }
}



fn main(){
    let a = Tensor::iden(2);
    let b = Tensor::iden(2);
    let mut c = a.matmul(&b);
    println!("{}", a);
    println!("{}", b);
    println!("{}", c);
    c.set_grad(&Tensor::ones(&vec![2 , 2]));
    c.init_grad();
    c.back_prop();
    c.print_grad();
    a.print_grad();
}
