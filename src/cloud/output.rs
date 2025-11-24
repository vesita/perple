use crate::utils::boxes::Box3D;



/// 固定容量的3D边界框容器
/// 
/// 这是一个类似于Vec的容器，但具有固定的最大容量，避免了动态分配内存的开销。
/// 它实现了常用的集合操作，如push、clear、len等，并支持迭代器。
#[derive(Clone)]
pub struct CldBud {
    pub the_box: Box3D,
    pub class_id: u32,
    pub class_name: String,
}

impl CldBud {
    /// 创建一个新的空Bounds容器
    pub fn new() -> Self {
        Self {
            the_box: Box3D::empty_box(),
            class_id: 0,
            class_name: String::new(),
        }
    }
}

impl Default for CldBud {
    fn default() -> Self {
        Self::new()
    }
}

