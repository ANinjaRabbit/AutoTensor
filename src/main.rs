use std::{ cell::RefCell, fmt::Display, ops::{Add, Div, Mul, Neg, Sub}, sync::Arc, vec};
use rand;
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
    Transpose(Arc<RefCell<TensorRaw>>),
    Assign(Arc<RefCell<TensorRaw>> , Arc<RefCell<TensorRaw>>),
    Slice(Arc<RefCell<TensorRaw>> , SliceIndex),
}
pub struct TensorRaw{
    raw : Arc<RefCell<Vec<f64>>>,
    shape : Vec<usize>,
    strides : Vec<usize>,
    offset : usize,
    op : Option<Op>,
    grad : Option<Vec<f64>>
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
    pub fn transpose(self : & Self)-> TensorRaw{
        let mut newraw = Vec::new();
        let mut newshape = self.shape.clone();
        newshape.reverse();
        for i in MultiDimIterator::new(&newshape){
            let mut newi = i.clone();
            newi.reverse();
            newraw.push(self.fetch(&newi));
        }
        TensorRaw{
            raw : Arc::new(RefCell::new(newraw)),
            shape : newshape.clone(),
            strides : {
                match newshape.len(){
                    0 => vec![],
                    _ => {
                        let mut strides = vec![1];
                        let mut prod = 1;
                        let mut rev_out_shape = newshape.clone();
                        rev_out_shape.reverse();
                        for i in &rev_out_shape[0..(newshape.len()-1)]  {
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
impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Self) -> Self::Output {
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
impl Div for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Self) -> Self::Output {
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
    pub fn transpose(self : & mut Self) -> Tensor{
        let res = Tensor(Arc::new(RefCell::new((self.0.borrow().transpose()).clone())));
        res.0.borrow_mut().op = Some(Op::Transpose(self.0.clone()));
        res
    }
}
impl Tensor {
    pub fn assign(self : & mut Self , rhs : & Tensor) {
        self.0.borrow_mut().assign(&rhs.0.borrow());
        self.0.borrow_mut().op = Some(Op::Assign(self.0.clone(), rhs.0.clone()));
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
    pub fn rand(shape : & Vec<usize>)-> Tensor{
        let res = Tensor(Arc::new(RefCell::new(TensorRaw::rand(shape))));
        res
    }
    pub fn randbetween(shape : & Vec<usize> , start : f64 , end : f64)-> Tensor{
        let res = Tensor(Arc::new(RefCell::new(TensorRaw::randbetween(shape , start , end))));
        res
    }
}

impl Tensor{
    pub fn init_grad(self : & mut Self){
        // clear all the childrens of op using recursion
        match &self.0.borrow().op{
            None => (),
            Some(Op :: Assign( a , b))=>{
                a.borrow_mut().grad = Some(vec![0.;a.borrow().shape.iter().product()]);
                b.borrow_mut().grad = Some(vec![0.;b.borrow().shape.iter().product()]);
            }
            Some(Op :: Add( a , b))=>{ a.borrow_mut().grad = Some(vec![0.;a.borrow().shape.iter().product()]);
                b.borrow_mut().grad = Some(vec![0.;b.borrow().shape.iter().product()]);
            }
            Some(Op :: Sub( a , b))=>{
                a.borrow_mut().grad = Some(vec![0.;a.borrow().shape.iter().product()]);
                b.borrow_mut().grad = Some(vec![0.;b.borrow().shape.iter().product()]);
            }
            Some(Op :: Mul( a , b))=>{
                a.borrow_mut().grad = Some(vec![0.;a.borrow().shape.iter().product()]);
                b.borrow_mut().grad = Some(vec![0.;b.borrow().shape.iter().product()]);
            }
            Some(Op :: Div( a , b))=>{
                a.borrow_mut().grad = Some(vec![0.;a.borrow().shape.iter().product()]);
                b.borrow_mut().grad = Some(vec![0.;b.borrow().shape.iter().product()]);
            }
            Some(Op :: Neg( a ))=>{
                a.borrow_mut().grad = Some(vec![0.;a.borrow().shape.iter().product()]);
            }
            Some(Op :: MatMul( a , b))=>{
                a.borrow_mut().grad = Some(vec![0.;a.borrow().shape.iter().product()]);
                b.borrow_mut().grad = Some(vec![0.;b.borrow().shape.iter().product()]);
            }
            Some(Op :: Exp( a ))=>{
                a.borrow_mut().grad = Some(vec![0.;a.borrow().shape.iter().product()]);
            }
            Some(Op :: Sum( a , _))=>{
                a.borrow_mut().grad = Some(vec![0.;a.borrow().shape.iter().product()]);
            }
            Some(Op :: Reshape( a ))=>{
                a.borrow_mut().grad = Some(vec![0.;a.borrow().shape.iter().product()]);
            }
            Some(Op :: Transpose( a ))=>{
                a.borrow_mut().grad = Some(vec![0.;a.borrow().shape.iter().product()]);
            }
            Some(Op :: Slice(a , index)) => {
                if(a.borrow().op.is_none()){
                    a.borrow_mut().grad = Some(vec![0.;a.borrow().shape.iter().product()]);
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
            }
        }
    }
}




fn main(){
    let a = Tensor::arange(0., 10., 1.);
    let b = Tensor::arange(0., 12., 1.).reshape(&vec![12 , 1]);
    println!("{}" , a.fetch(&vec![10 , 0]));
    println!("{}" , b.fetch(&vec![10 , 0]));
    let mut c = &a * &b;
    let d = c.exp();
    println!("{}" , c.fetch(&vec![10 , 0]));
    println!("{}", c);
    c.get(s![(0 ,2) , (0 , 5)]).assign(&Tensor::rand(&vec![2 , 5]));
    println!("{}" , c.get(s![]));
    println!("{}" , c.get(s![(0 , 10) , (0 , 10)]).transpose().matmul(&Tensor::iden(10)));
}
