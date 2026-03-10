use crate::utils::boxes::Box3D;

#[derive(Clone)]
pub struct Target {
    pub the_box: Box3D,
    pub class_type: String,
    pub id: usize, // 全局id
}

impl Target {
    pub fn new(the_box: Box3D, class_type: String, id: usize) -> Self {
        Self {
            the_box,
            class_type,
            id,
        }
    }

    pub fn default() -> Self {
        Self {
            the_box: Box3D::empty_box(),
            class_type: String::new(),
            id: 0,
        }
    }
}
