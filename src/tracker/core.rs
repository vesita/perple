use std::sync::{Arc, Mutex};

use crate::{cloud::CldBud, color::ClrBud, utils::stream::Stream};




pub struct Tracker {
    tar2d: Arc<Mutex<Stream<Vec<ClrBud>>>>,
    tar3d: Arc<Mutex<Stream<Vec<CldBud>>>>,
}

impl Tracker {
    pub fn new(
        tar2d: Arc<Mutex<Stream<Vec<ClrBud>>>>,
        tar3d: Arc<Mutex<Stream<Vec<CldBud>>>>,
    ) -> Self {
        Self {
            tar2d,
            tar3d
        }
    }
}