use crate::utils::Box3D;



#[derive(Clone)]
pub struct Target {
    pub the_box: Box3D,
    pub class_type: String,
    pub id: usize,      // 全局id
}


impl Target {
    pub fn new() -> Self {
        Self {
            the_box: Box3D::empty_box(),
            class_type: "".to_string(),
            id: 0,
        }
    }
}